// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "lib/rocprofiler-sdk/pc_sampling/dispatch_ring_drainer.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

#    include "lib/common/utility.hpp"
#    include "lib/rocprofiler-sdk/agent.hpp"
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
#    include <hsa/amd_hsa_queue.h>

#    include <algorithm>
#    include <atomic>
#    include <chrono>
#    include <cstdint>
#    include <cstring>
#    include <mutex>
#    include <thread>
#    include <unordered_map>
#    include <unordered_set>
#    include <vector>

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
struct mec_dispatch_record_16
{
    uint32_t ts_lo;
    uint32_t ts_hi;
    uint32_t record_type;   // 1 = dispatch-start, 2 = dispatch-end (EOP)
    uint32_t dispatch_idx;  // read_dispatch_id[31:0] from firmware
};
#pragma pack(pop)

struct timed_record_t
{
    uint64_t ts;
    uint32_t record_type;
    uint32_t slot;
    uint32_t dispatch_idx;
};

struct queue_ring_state_t
{
    hsa_queue_t*         queue{};
    void*                buf{};
    uint32_t             ring_bytes{};
    volatile uint32_t*   wptr{};
    uint64_t             read_idx{};
    uint32_t             record_size{};
    hsa_agent_t          agent{};
    uint64_t             pending_start_ts{0};
    uint32_t             pending_dispatch_idx{0};
    bool                 has_pending_start{false};
    uint64_t             dispatch_count{0};
    uint64_t             last_seen_ts{0};
    uint32_t             last_processed_record_count{0};
};

std::mutex                                             g_ring_mu;
std::unordered_map<uint64_t, queue_ring_state_t>       g_queue_rings;
std::atomic<bool>                                      g_drainer_stop{true};
std::thread                                            g_drainer_thread;
context::pc_sampling_service*                          g_pcs_service = nullptr;
bool                                                   g_standalone_mode{false};
std::atomic<rocprofiler_dispatch_id_t>               g_next_dispatch_id{1};

uint32_t
infer_record_size(uint32_t ring_bytes)
{
    if(g_standalone_mode)
    {
        if(ring_bytes >= sizeof(mec_dispatch_record_16) &&
           ring_bytes % sizeof(mec_dispatch_record_16) == 0)
            return sizeof(mec_dispatch_record_16);
    }
    if(ring_bytes >= sizeof(mec_dispatch_record_40) && ring_bytes % sizeof(mec_dispatch_record_40) == 0)
        return sizeof(mec_dispatch_record_40);
    if(ring_bytes >= sizeof(mec_dispatch_record_32) && ring_bytes % sizeof(mec_dispatch_record_32) == 0)
        return sizeof(mec_dispatch_record_32);
    if(ring_bytes >= sizeof(mec_dispatch_record_16) && ring_bytes % sizeof(mec_dispatch_record_16) == 0)
        return sizeof(mec_dispatch_record_16);
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
convert_paired_16b(uint64_t            start_ts,
                   uint64_t            end_ts,
                   uint64_t            kernel_obj,
                   uint32_t            queue_size,
                   uint32_t            ring_index,
                   uint64_t*           kernel_object_out,
                   mec_dispatch_record_40* out)
{
    out->start_ts      = start_ts;
    out->end_ts        = end_ts;
    out->kernel_object = kernel_obj;
    out->doorbell_id   = 0;
    out->ring_index    = ring_index;
    out->queue_size    = queue_size;
    out->flags         = 1;  // mark as valid dispatch
    *kernel_object_out = kernel_obj;
}

uint64_t
lookup_kernel_object(queue_ring_state_t& qs, uint32_t dispatch_idx)
{
    if(!qs.queue || !qs.queue->base_address || qs.queue->size == 0) return 0;
    uint32_t q_size = qs.queue->size;
    uint32_t slot   = dispatch_idx % q_size;
    const auto* pkts =
        static_cast<const hsa_kernel_dispatch_packet_t*>(qs.queue->base_address);
    uint64_t ko = 0;
    std::memcpy(&ko, &pkts[slot].kernel_object, sizeof(ko));
    return ko;
}

void
emit_kernel_dispatch_tracing_standalone(hsa_agent_t                   hag,
                                        hsa_queue_t*                  queue,
                                        const mec_dispatch_record_40& rec,
                                        uint64_t                      kernel_object,
                                        rocprofiler_dispatch_id_t     dispatch_id,
                                        context::correlation_id*      cid)
{
    tracing::tracing_data td{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                               td);
    if(td.callback_contexts.empty() && td.buffered_contexts.empty()) return;

    uint64_t start_ns = rec.start_ts;
    uint64_t end_ns   = rec.end_ts;
    auto*    ext      = hsa::get_amd_ext_table();
    if(ext && ext->hsa_amd_profiling_convert_tick_to_system_domain_fn)
    {
        ext->hsa_amd_profiling_convert_tick_to_system_domain_fn(hag, rec.start_ts, &start_ns);
        ext->hsa_amd_profiling_convert_tick_to_system_domain_fn(hag, rec.end_ts, &end_ns);
    }

    const auto* rocp_agent = rocprofiler::agent::get_rocprofiler_agent(hag);
    if(!rocp_agent) { fprintf(stderr, "[DRAINER DBG]   early return: no rocp_agent\n"); return; }

    auto kid = rocprofiler::code_object::get_kernel_id(kernel_object);
    // With late-attach, the code object tracking hooks may not have
    // seen the kernel registration.  Still emit the record so the
    // tool can attribute timing.  Use kernel_object as a synthetic
    // kernel_id when the real kid is unknown.
    if(kid == 0 && kernel_object != 0)
        kid = kernel_object;

    constexpr auto kernel_dispatch_info_rt_size =
        common::compute_runtime_sizeof<rocprofiler_kernel_dispatch_info_t>();
    auto dispatch_info = common::init_public_api_struct(rocprofiler_kernel_dispatch_info_t{});
    dispatch_info.size                 = kernel_dispatch_info_rt_size;
    dispatch_info.agent_id             = rocp_agent->id;
    dispatch_info.queue_id             = rocprofiler_queue_id_t{.handle = queue->id};
    dispatch_info.kernel_id            = rocprofiler_kernel_id_t{kid};
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

    auto  thr_id            = common::get_tid();
    auto  internal_corr_id  = cid->internal;
    auto  ancestor_corr_id  = cid->ancestor;
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
    dispatch_info.kernel_id            = rocprofiler::code_object::get_kernel_id(kernel_object);
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
process_dispatch_record_standalone(queue_ring_state_t*           st,
                                   const mec_dispatch_record_40& rec,
                                   uint64_t                      kernel_object)
{
    constexpr uint32_t init_ref = 2;
    auto*              cid      =
        context::correlation_tracing_service::construct(init_ref);
    if(!cid)
    {
        // During finalization, correlation service may be shut down.
        // Create a minimal correlation ID to allow record emission.
        static thread_local context::correlation_id fallback_cid{};
        cid = &fallback_cid;
    }

    auto dispatch_id = g_next_dispatch_id.fetch_add(1, std::memory_order_relaxed);
    emit_kernel_dispatch_tracing_standalone(
        st->agent, st->queue, rec, kernel_object, dispatch_id, cid);
}

void
drain_queue_entry_standalone(queue_ring_state_t& qs)
{
    if(!qs.buf || qs.record_size == 0 || qs.ring_bytes < qs.record_size) return;

    const bool is_16b = (qs.record_size == sizeof(mec_dispatch_record_16));
    if(!is_16b) return;

    const auto*    base      = static_cast<const uint8_t*>(qs.buf);
    const uint32_t num_slots = qs.ring_bytes / qs.record_size;

    std::vector<timed_record_t> records;
    for(uint32_t i = 0; i < num_slots; i++)
    {
        mec_dispatch_record_16 r16{};
        std::memcpy(&r16, base + i * sizeof(r16), sizeof(r16));
        uint64_t ts = (static_cast<uint64_t>(r16.ts_hi) << 32) | r16.ts_lo;
        if(ts != 0 && (r16.record_type == 1 || r16.record_type == 2))
            records.push_back({ts, r16.record_type, i, r16.dispatch_idx});
    }

    if(records.size() <= qs.last_processed_record_count) return;

    std::sort(records.begin(), records.end(),
              [](const timed_record_t& a, const timed_record_t& b) { return a.ts < b.ts; });

    uint32_t skip = qs.last_processed_record_count;
    qs.last_processed_record_count = static_cast<uint32_t>(records.size());

    for(size_t idx = skip; idx < records.size(); idx++)
    {
        auto& r = records[idx];
        if(r.record_type == 1)
        {
            qs.pending_start_ts      = r.ts;
            qs.pending_dispatch_idx  = r.dispatch_idx;
            qs.has_pending_start     = true;
        }
        else if(r.record_type == 2)
        {
            uint64_t start_ts      = qs.has_pending_start ? qs.pending_start_ts : 0;
            uint32_t disp_idx      = qs.has_pending_start ? qs.pending_dispatch_idx
                                                          : r.dispatch_idx;
            uint64_t kernel_obj    = lookup_kernel_object(qs, disp_idx);

            mec_dispatch_record_40 rec{};
            rec.start_ts      = start_ts;
            rec.end_ts        = r.ts;
            rec.kernel_object = kernel_obj;

            process_dispatch_record_standalone(&qs, rec, kernel_obj);
            qs.has_pending_start = false;
            qs.dispatch_count++;
        }
    }
}

void
drain_queue_entry(queue_ring_state_t& qs, context::pc_sampling_service* svc)
{
    if(!qs.buf || !qs.wptr || qs.record_size == 0 || qs.ring_bytes < qs.record_size) return;

    PCSAgentSession* session = find_agent_session(qs.agent, svc);
    if(!session || !session->parser || !session->cid_manager) return;

    const uint32_t num_slots = qs.ring_bytes / qs.record_size;
    const bool     is_16b    = (qs.record_size == sizeof(mec_dispatch_record_16));

    for(;;)
    {
        uint32_t w = *qs.wptr;
        if(qs.read_idx >= w) break;

        const auto* base  = static_cast<const uint8_t*>(qs.buf);
        const auto  slot  = qs.read_idx % num_slots;
        const auto* rec_p = base + static_cast<size_t>(slot) * qs.record_size;

        if(is_16b)
        {
            mec_dispatch_record_16 r16{};
            std::memcpy(&r16, rec_p, sizeof(r16));

            if(r16.record_type == 1)
            {
                qs.pending_start_ts     = (static_cast<uint64_t>(r16.ts_hi) << 32) | r16.ts_lo;
                qs.pending_dispatch_idx = r16.dispatch_idx;
                qs.has_pending_start    = true;
                qs.read_idx++;
                continue;
            }

            if(r16.record_type == 2 && qs.has_pending_start)
            {
                uint64_t end_ts     = (static_cast<uint64_t>(r16.ts_hi) << 32) | r16.ts_lo;
                uint64_t kernel_obj = lookup_kernel_object(qs, qs.pending_dispatch_idx);
                uint32_t q_sz       = qs.queue ? qs.queue->size : 0;
                uint32_t ring_idx   = qs.pending_dispatch_idx % (q_sz ? q_sz : 1);

                mec_dispatch_record_40 rec{};
                uint64_t               kernel_object = 0;
                convert_paired_16b(qs.pending_start_ts, end_ts, kernel_obj,
                                   q_sz, ring_idx, &kernel_object, &rec);

                process_dispatch_record(&qs, session, rec, kernel_object);
                qs.has_pending_start = false;
                qs.dispatch_count++;
                qs.read_idx++;
                continue;
            }

            qs.read_idx++;
            continue;
        }

        mec_dispatch_record_40 rec{};
        uint64_t               kernel_object = 0;
        convert_record(rec_p, qs.record_size, &kernel_object, &rec);

        process_dispatch_record(&qs, session, rec, kernel_object);
        qs.read_idx++;
    }
}

hsa_status_t
register_or_refresh_queue(hsa_queue_t* queue, void*)
{
    if(rocprofiler::registration::get_fini_status() > 0) return HSA_STATUS_SUCCESS;

    {
        std::lock_guard<std::mutex> lk(g_ring_mu);
        if(g_queue_rings.count(queue->id)) return HSA_STATUS_SUCCESS;
    }

    auto* ext = hsa::get_amd_ext_table();
    if(!ext || !ext->hsa_amd_queue_get_info_fn || !ext->hsa_amd_profiling_set_profiler_enabled_fn)
        return HSA_STATUS_SUCCESS;

    hsa_agent_t agent{};
    if(ext->hsa_amd_queue_get_info_fn(queue, HSA_AMD_QUEUE_INFO_AGENT, &agent) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

    hsa_device_type_t dt = HSA_DEVICE_TYPE_CPU;
    if(hsa::get_core_table()->hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &dt) !=
           HSA_STATUS_SUCCESS ||
       dt != HSA_DEVICE_TYPE_GPU)
        return HSA_STATUS_SUCCESS;

    if(!g_standalone_mode)
    {
        PCSAgentSession* session = find_agent_session(agent, g_pcs_service);
        if(!session) return HSA_STATUS_SUCCESS;
    }

    if(ext->hsa_amd_profiling_set_profiler_enabled_fn(queue, true) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

    void*               buf  = nullptr;
    uint32_t            sz   = 0;
    volatile uint32_t* wptr = nullptr;
    auto get_fn = hsa::dispatch_ring_buffer_get_dispatch_records_fn_v();
    if(!get_fn || get_fn(queue, &buf, &sz, &wptr) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

    uint32_t rec_sz = infer_record_size(sz);
    if(rec_sz == 0 || !buf || !wptr) return HSA_STATUS_SUCCESS;

    std::lock_guard<std::mutex> lk(g_ring_mu);
    auto&                       ent    = g_queue_rings[queue->id];
    ent.queue                          = queue;
    ent.buf                            = buf;
    ent.ring_bytes                     = sz;
    ent.wptr                           = wptr;
    ent.agent                          = agent;
    ent.record_size                    = rec_sz;
    if(ent.read_idx > *wptr) ent.read_idx = *wptr;
    return HSA_STATUS_SUCCESS;
}

// Tracks dispatch_idx values that have already been emitted across all queues
std::unordered_set<uint32_t> g_emitted_dispatch_idx;

void
drain_all_standalone()
{
    // Find the queue with a valid AQL ring (used for kernel_object lookup)
    queue_ring_state_t* aql_qs = nullptr;
    for(auto& [_, qs] : g_queue_rings)
    {
        if(qs.queue && qs.queue->base_address && qs.queue->size > 0)
        {
            aql_qs = &qs;
            break;
        }
    }

    // Drain each queue's buffer independently, pair START+END by dispatch_idx within same buffer
    for(auto& [qid, qs] : g_queue_rings)
    {
        if(!qs.buf || qs.record_size == 0 || qs.ring_bytes < qs.record_size) continue;
        if(qs.record_size != sizeof(mec_dispatch_record_16)) continue;

        const auto*    base      = static_cast<const uint8_t*>(qs.buf);
        const uint32_t num_slots = qs.ring_bytes / qs.record_size;

        // Collect all valid records from this buffer
        std::vector<timed_record_t> records;
        for(uint32_t i = 0; i < num_slots; i++)
        {
            mec_dispatch_record_16 r16{};
            std::memcpy(&r16, base + i * sizeof(r16), sizeof(r16));
            uint64_t ts = (static_cast<uint64_t>(r16.ts_hi) << 32) | r16.ts_lo;
            if(ts != 0 && (r16.record_type == 1 || r16.record_type == 2))
                records.push_back({ts, r16.record_type, i, r16.dispatch_idx});
        }

        if(records.size() <= qs.last_processed_record_count) continue;

        qs.last_processed_record_count = static_cast<uint32_t>(records.size());

        // Build per-dispatch_idx: collect all START and END timestamps.
        // Multi-XCC can produce multiple START/END records per dispatch_idx in the same buffer.
        // We pair the START with the END from the same XCC by finding a valid pair (end > start)
        // with the smallest duration.
        // Filter out records whose dispatch_idx was already emitted
        std::vector<timed_record_t> fresh;
        for(auto& r : records)
        {
            if(!g_emitted_dispatch_idx.count(r.dispatch_idx))
                fresh.push_back(r);
        }

        std::sort(fresh.begin(), fresh.end(),
                  [](const timed_record_t& a, const timed_record_t& b) { return a.ts < b.ts; });
        auto& records_ref = fresh;

        queue_ring_state_t* lookup_qs = (aql_qs ? aql_qs : &qs);

        // Two pending STARTs: one per XCC (we detect XCC by slot range).
        // Simpler approach: maintain a stack of pending starts, pop on END.
        struct pending_t { uint64_t ts; uint32_t dispatch_idx; };
        std::vector<pending_t> pending_starts;

        for(auto& r : records_ref)
        {
            if(r.record_type == 1)
            {
                pending_starts.push_back({r.ts, r.dispatch_idx});
            }
            else if(r.record_type == 2 && !pending_starts.empty())
            {
                // Find the most recent pending START with ts < this END
                int best_i = -1;
                uint64_t best_gap = UINT64_MAX;
                for(int i = static_cast<int>(pending_starts.size()) - 1; i >= 0; i--)
                {
                    if(pending_starts[i].ts < r.ts)
                    {
                        uint64_t gap = r.ts - pending_starts[i].ts;
                        if(gap < best_gap)
                        {
                            best_gap = gap;
                            best_i   = i;
                        }
                    }
                }
                if(best_i < 0) continue;

                auto start = pending_starts[best_i];
                pending_starts.erase(pending_starts.begin() + best_i);

                if(g_emitted_dispatch_idx.count(start.dispatch_idx)) continue;

                uint64_t kernel_obj = lookup_kernel_object(*lookup_qs, start.dispatch_idx);

                mec_dispatch_record_40 rec{};
                rec.start_ts      = start.ts;
                rec.end_ts        = r.ts;
                rec.kernel_object = kernel_obj;

                process_dispatch_record_standalone(lookup_qs, rec, kernel_obj);
                g_emitted_dispatch_idx.insert(start.dispatch_idx);
                lookup_qs->dispatch_count++;
            }
        }
    }
}

void
drain_all_registered()
{
    std::lock_guard<std::mutex> lk(g_ring_mu);
    if(g_standalone_mode)
    {
        drain_all_standalone();
        return;
    }
    auto* svc = g_pcs_service;
    if(!svc) return;
    for(auto& [_, qs] : g_queue_rings)
        drain_queue_entry(qs, svc);
}

void
discover_queues()
{
    auto it_fn = hsa::dispatch_ring_buffer_queue_iterate_fn_v();
    if(it_fn) it_fn(register_or_refresh_queue, nullptr);
}

void
drainer_loop()
{
    while(!g_drainer_stop.load(std::memory_order_acquire))
    {
        if((g_standalone_mode || g_pcs_service) && hsa::firmware_dispatch_ring_available())
        {
            discover_queues();
            drain_all_registered();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Final drain while tracing service is still alive
    if((g_standalone_mode || g_pcs_service) && hsa::firmware_dispatch_ring_available())
    {
        {
            std::lock_guard<std::mutex> lk(g_ring_mu);
            for(auto& [_, qs] : g_queue_rings)
                qs.last_processed_record_count = 0;
        }
        discover_queues();
        drain_all_registered();
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
    // Give the drainer thread time to catch final records while services are alive.
    // The thread runs every 1ms, so 10ms is plenty for several drain cycles.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    g_drainer_stop.store(true, std::memory_order_release);
    if(g_drainer_thread.joinable()) g_drainer_thread.join();

    std::lock_guard<std::mutex> lk(g_ring_mu);
    g_queue_rings.clear();
    g_emitted_dispatch_idx.clear();
    g_pcs_service     = nullptr;
    g_standalone_mode = false;
}

void
drain_firmware_dispatch_rings_before_pcs_batch()
{
    if(!hsa::firmware_dispatch_ring_available()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    drain_all_registered();
}

void
start_firmware_dispatch_ring_drainer_standalone()
{
    if(!hsa::firmware_dispatch_ring_available()) return;

    {
        std::lock_guard<std::mutex> lk(g_ring_mu);
        g_standalone_mode = true;
    }

    g_drainer_stop.store(false, std::memory_order_release);
    if(g_drainer_thread.joinable()) g_drainer_thread.join();
    g_drainer_thread = std::thread{drainer_loop};
}

}  // namespace pc_sampling
}  // namespace rocprofiler

#endif
