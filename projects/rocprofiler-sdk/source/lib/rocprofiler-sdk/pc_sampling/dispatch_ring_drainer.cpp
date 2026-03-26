// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "lib/rocprofiler-sdk/pc_sampling/dispatch_ring_drainer.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

#    include "lib/common/utility.hpp"
#    include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#    include "lib/rocprofiler-sdk/context/context.hpp"
#    include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#    include "lib/rocprofiler-sdk/hsa/dispatch_ring_buffer_support.hpp"
#    include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/parser/rocr.h"
#    include "lib/rocprofiler-sdk/pc_sampling/types.hpp"
#    include "lib/rocprofiler-sdk/registration.hpp"
#    include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#    include <rocprofiler-sdk/buffer_tracing.h>
#    include <rocprofiler-sdk/callback_tracing.h>

#    include <hsa/hsa.h>

#    include <atomic>
#    include <chrono>
#    include <cstdint>
#    include <cstring>
#    include <mutex>
#    include <thread>
#    include <unordered_map>

namespace rocprofiler
{
namespace pc_sampling
{
namespace
{
#pragma pack(push, 1)
struct mec_dispatch_record_40
{
    uint64_t start_ts;
    uint64_t end_ts;
    uint64_t kernel_object;
    uint32_t doorbell_id;
    uint32_t ring_index;
    uint32_t queue_size;
    uint32_t flags;
};
struct mec_dispatch_record_32
{
    uint64_t start_ts;
    uint64_t end_ts;
    uint32_t doorbell_id;
    uint32_t ring_index;
    uint32_t queue_size;
    uint32_t flags;
};
#pragma pack(pop)

struct queue_ring_state_t
{
    hsa_queue_t*         queue{};
    void*                buf{};
    uint32_t             ring_bytes{};
    volatile uint32_t*   wptr{};
    uint64_t             read_idx{};
    uint32_t             record_size{};
    hsa_agent_t          agent{};
};

std::mutex                                             g_ring_mu;
std::unordered_map<uint64_t, queue_ring_state_t>       g_queue_rings;
std::atomic<bool>                                      g_drainer_stop{true};
std::thread                                            g_drainer_thread;
context::pc_sampling_service*                          g_pcs_service = nullptr;
std::atomic<rocprofiler_dispatch_id_t>               g_next_dispatch_id{1};

uint32_t
infer_record_size(uint32_t ring_bytes)
{
    if(ring_bytes >= sizeof(mec_dispatch_record_40) && ring_bytes % sizeof(mec_dispatch_record_40) == 0)
        return sizeof(mec_dispatch_record_40);
    if(ring_bytes >= sizeof(mec_dispatch_record_32) && ring_bytes % sizeof(mec_dispatch_record_32) == 0)
        return sizeof(mec_dispatch_record_32);
    return 0;
}

PCSAgentSession*
find_agent_session(hsa_agent_t hag, context::pc_sampling_service* svc)
{
    if(!svc) return nullptr;
    for(const auto& [_, session] : svc->agent_sessions)
    {
        if(session->hsa_agent.has_value() && session->hsa_agent->handle == hag.handle)
            return session.get();
    }
    return nullptr;
}

void
convert_record(const uint8_t* base, uint32_t rec_sz, uint64_t* kernel_object, mec_dispatch_record_40* out)
{
    if(rec_sz >= sizeof(mec_dispatch_record_40))
    {
        std::memcpy(out, base, sizeof(mec_dispatch_record_40));
        *kernel_object = out->kernel_object;
    }
    else
    {
        mec_dispatch_record_32 r32{};
        std::memcpy(&r32, base, sizeof(r32));
        out->start_ts      = r32.start_ts;
        out->end_ts        = r32.end_ts;
        out->doorbell_id   = r32.doorbell_id;
        out->ring_index    = r32.ring_index;
        out->queue_size    = r32.queue_size;
        out->flags         = r32.flags;
        out->kernel_object = 0;
        *kernel_object     = 0;
    }
}

void
emit_kernel_dispatch_tracing(PCSAgentSession*              session,
                               hsa_queue_t*                queue,
                               const mec_dispatch_record_40& rec,
                               uint64_t                    kernel_object,
                               rocprofiler_dispatch_id_t   dispatch_id,
                               context::correlation_id*    cid)
{
    tracing::tracing_data td{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                               td);
    if(td.callback_contexts.empty() && td.buffered_contexts.empty()) return;

    hsa_agent_t hag = session->hsa_agent.value();
    uint64_t    start_ns = rec.start_ts;
    uint64_t    end_ns   = rec.end_ts;
    auto*       ext      = hsa::get_amd_ext_table();
    if(ext && ext->hsa_amd_profiling_convert_tick_to_system_domain_fn)
    {
        ext->hsa_amd_profiling_convert_tick_to_system_domain_fn(hag, rec.start_ts, &start_ns);
        ext->hsa_amd_profiling_convert_tick_to_system_domain_fn(hag, rec.end_ts, &end_ns);
    }

    constexpr auto kernel_dispatch_info_rt_size =
        common::compute_runtime_sizeof<rocprofiler_kernel_dispatch_info_t>();
    auto dispatch_info = common::init_public_api_struct(rocprofiler_kernel_dispatch_info_t{});
    dispatch_info.size                 = kernel_dispatch_info_rt_size;
    dispatch_info.agent_id             = session->agent->id;
    dispatch_info.queue_id             = rocprofiler_queue_id_t{.handle = queue->id};
    dispatch_info.kernel_id            = code_object::get_kernel_id(kernel_object);
    dispatch_info.dispatch_id          = dispatch_id;
    dispatch_info.workgroup_size       = {0, 0, 0};
    dispatch_info.grid_size            = {0, 0, 0};
    dispatch_info.private_segment_size = 0;
    dispatch_info.group_segment_size   = 0;

    auto tracer_data = rocprofiler_callback_tracing_kernel_dispatch_data_t{};
    tracer_data.size             = sizeof(tracer_data);
    tracer_data.start_timestamp  = start_ns;
    tracer_data.end_timestamp    = end_ns;
    tracer_data.dispatch_info    = dispatch_info;

    auto thr_id             = common::get_tid();
    auto internal_corr_id   = cid->internal;
    auto ancestor_corr_id   = cid->ancestor;
    auto& extern_corr       = td.external_correlation_ids;

    if(!td.callback_contexts.empty())
    {
        tracing::execute_phase_none_callbacks(td.callback_contexts,
                                              thr_id,
                                              internal_corr_id,
                                              extern_corr,
                                              ancestor_corr_id,
                                              ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                              ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                              tracer_data);
    }

    if(!td.buffered_contexts.empty())
    {
        auto record = rocprofiler_buffer_tracing_kernel_dispatch_record_t{
            sizeof(rocprofiler_buffer_tracing_kernel_dispatch_record_t),
            ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
            ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
            rocprofiler_async_correlation_id_t{},
            thr_id,
            start_ns,
            end_ns,
            dispatch_info};
        tracing::execute_buffer_record_emplace(td.buffered_contexts,
                                               thr_id,
                                               internal_corr_id,
                                               extern_corr,
                                               ancestor_corr_id,
                                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                               ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                               std::move(record));
    }
}

void
process_dispatch_record(queue_ring_state_t*           st,
                        PCSAgentSession*              session,
                        const mec_dispatch_record_40& rec,
                        uint64_t                      kernel_object)
{
    if(!(rec.flags & 1U)) return;  // barrier
    if(rec.queue_size == 0 || rec.queue_size >= (1U << 25)) return;

    constexpr uint32_t init_pc_sampling_ref = 5;
    auto*              cid =
        context::correlation_tracing_service::construct(init_pc_sampling_ref);
    if(!cid) return;

    auto dispatch_id = g_next_dispatch_id.fetch_add(1, std::memory_order_relaxed);
    auto*  parser    = session->parser.get();

    dispatch_pkt_id_t pkt{};
    pkt.type            = AMD_DISPATCH_PKT_ID;
    pkt.device          = device_handle{static_cast<uint32_t>(session->agent->id.handle)};
    pkt.doorbell_id     = rec.doorbell_id;
    pkt.queue_size      = rec.queue_size;
    pkt.write_index     = rec.ring_index;
    pkt.read_index      = 0;
    pkt.correlation_id  = rocprofiler_async_correlation_id_t{.internal = cid->internal,
                                                            .external = rocprofiler_user_data_t{}};
    pkt.dispatch_id     = dispatch_id;

    if(parser->shouldFlipRocrBuffer(pkt))
    {
        auto* pct = rocprofiler::hsa::get_table().pc_sampling_ext_;
        if(pct && pct->hsa_ven_amd_pcs_flush_fn)
            pct->hsa_ven_amd_pcs_flush_fn(session->hsa_pc_sampling);
    }

    parser->newDispatch(pkt);
    emit_kernel_dispatch_tracing(session, st->queue, rec, kernel_object, dispatch_id, cid);
    session->cid_manager->cid_async_activity_completed(cid);
}

void
drain_queue_entry(queue_ring_state_t& qs, context::pc_sampling_service* svc)
{
    if(!qs.buf || !qs.wptr || qs.record_size == 0 || qs.ring_bytes < qs.record_size) return;

    PCSAgentSession* session = find_agent_session(qs.agent, svc);
    if(!session || !session->parser || !session->cid_manager) return;

    const uint32_t num_slots = qs.ring_bytes / qs.record_size;

    for(;;)
    {
        uint32_t w = *qs.wptr;
        if(qs.read_idx >= w) break;

        const auto* base  = static_cast<const uint8_t*>(qs.buf);
        const auto  slot  = qs.read_idx % num_slots;
        const auto* rec_p = base + static_cast<size_t>(slot) * qs.record_size;

        mec_dispatch_record_40 rec{};
        uint64_t               kernel_object = 0;
        convert_record(rec_p, qs.record_size, &kernel_object, &rec);

        process_dispatch_record(&qs, session, rec, kernel_object);
        qs.read_idx++;
    }
}

void
register_or_refresh_queue(hsa_queue_t* queue, void*)
{
    if(rocprofiler::registration::get_fini_status() > 0) return;

    {
        std::lock_guard<std::mutex> lk(g_ring_mu);
        if(g_queue_rings.count(queue->id)) return;
    }

    auto* ext = hsa::get_amd_ext_table();
    if(!ext || !ext->hsa_amd_queue_get_info_fn || !ext->hsa_amd_profiling_set_profiler_enabled_fn)
        return;

    hsa_agent_t agent{};
    if(ext->hsa_amd_queue_get_info_fn(queue, HSA_AMD_QUEUE_INFO_AGENT, &agent) != HSA_STATUS_SUCCESS)
        return;

    hsa_device_type_t dt = HSA_DEVICE_TYPE_CPU;
    if(hsa::get_core_table().hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &dt) !=
           HSA_STATUS_SUCCESS ||
       dt != HSA_DEVICE_TYPE_GPU)
        return;

    PCSAgentSession* session = find_agent_session(agent, g_pcs_service);
    if(!session) return;

    if(ext->hsa_amd_profiling_set_profiler_enabled_fn(queue, true) != HSA_STATUS_SUCCESS) return;

    void*               buf  = nullptr;
    uint32_t            sz   = 0;
    volatile uint32_t* wptr = nullptr;
    auto get_fn = dispatch_ring_buffer_get_dispatch_records_fn_v();
    if(!get_fn || get_fn(queue, &buf, &sz, &wptr) != HSA_STATUS_SUCCESS) return;

    uint32_t rec_sz = infer_record_size(sz);
    if(rec_sz == 0 || !buf || !wptr) return;

    std::lock_guard<std::mutex> lk(g_ring_mu);
    auto&                       ent    = g_queue_rings[queue->id];
    ent.queue                          = queue;
    ent.buf                            = buf;
    ent.ring_bytes                     = sz;
    ent.wptr                           = wptr;
    ent.agent                          = agent;
    ent.record_size                    = rec_sz;
    if(ent.read_idx > *wptr) ent.read_idx = *wptr;
}

void
drain_all_registered()
{
    auto* svc = g_pcs_service;
    if(!svc) return;
    std::lock_guard<std::mutex> lk(g_ring_mu);
    for(auto& [_, qs] : g_queue_rings)
        drain_queue_entry(qs, svc);
}

void
discover_queues()
{
    auto it_fn = dispatch_ring_buffer_queue_iterate_fn_v();
    if(it_fn) it_fn(register_or_refresh_queue, nullptr);
}

void
drainer_loop()
{
    while(!g_drainer_stop.load(std::memory_order_acquire))
    {
        if(g_pcs_service && hsa::firmware_dispatch_ring_available())
        {
            discover_queues();
            drain_all_registered();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
}  // namespace

void
start_firmware_dispatch_ring_drainer(context::pc_sampling_service* service)
{
    if(!service || !hsa::firmware_dispatch_ring_available()) return;

    std::lock_guard<std::mutex> lk(g_ring_mu);
    g_pcs_service = service;

    g_drainer_stop.store(false, std::memory_order_release);
    if(g_drainer_thread.joinable()) g_drainer_thread.join();
    g_drainer_thread = std::thread{drainer_loop};

    discover_queues();
    drain_all_registered();
}

void
stop_firmware_dispatch_ring_drainer()
{
    g_drainer_stop.store(true, std::memory_order_release);
    if(g_drainer_thread.joinable()) g_drainer_thread.join();

    std::lock_guard<std::mutex> lk(g_ring_mu);
    g_queue_rings.clear();
    g_pcs_service = nullptr;
}

void
drain_firmware_dispatch_rings_before_pcs_batch()
{
    if(!hsa::firmware_dispatch_ring_available()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    drain_all_registered();
}

}  // namespace pc_sampling
}  // namespace rocprofiler

#endif
