// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

namespace rocprofiler
{
namespace context
{
struct pc_sampling_service;
}
namespace pc_sampling
{
struct PCSAgentSession;

void
start_firmware_dispatch_ring_drainer(context::pc_sampling_service* service);

void
stop_firmware_dispatch_ring_drainer();

/** Option B: drain before parsing a PCS batch; optional delay applied inside when enabled. */
void
drain_firmware_dispatch_rings_before_pcs_batch();

}  // namespace pc_sampling
}  // namespace rocprofiler

#endif
