////////////////////////////////////////////////////////////////////////////////
//
// MEC firmware dispatch record layout (userspace mirror for sizing).
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_MEC_DISPATCH_RECORD_H_
#define HSA_RUNTIME_CORE_INC_MEC_DISPATCH_RECORD_H_

#include <cstdint>

namespace rocr {
namespace AMD {

/// 40-byte variant (preferred) — must match firmware / MQD agreement.
struct mec_dispatch_record {
  uint64_t start_ts;
  uint64_t end_ts;
  uint64_t kernel_object;
  uint32_t doorbell_id;
  uint32_t ring_index;
  uint32_t queue_size;
  uint32_t flags;
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_MEC_DISPATCH_RECORD_H_
