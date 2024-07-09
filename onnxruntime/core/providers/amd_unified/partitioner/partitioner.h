// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Standard headers/libs.
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <memory>

// 1st-party headers/libs.
//include "core/framework/op_kernel_info.h"
//#include "core/framework/execution_provider.h"
#include "core/providers/shared_library/provider_api.h"

#include "../amd_unified_execution_provider_info.h"


namespace onnxruntime {
namespace amduai {

// `onnxruntime::NodeIndex` is an alias of `size_t`.
// This is for deduplicating single-node sub-graphs
// from different downstream EPs.
// TODO: We might extend to support multi-node sub-graphs in future,
// so we are using `vector` as the element type.
struct SingleElemVectorHasher {
  size_t operator()(const std::vector<size_t>& vec) const {
    return vec[0];
  }
};

/*
Phase 1) Partitioner using coverage or priority based decisions.
Phase 2) Partitions decided based on estimated performance.
Phase 3) Cut based mappings, each device providing multiple options of subgraphs.
Phase 4) Device switching cost minimized by merging partitions into fused partition.

1) Split the input model when its size exceeds a limit.
2) Iterate over the list of multiple sub-models and during each iteration:
   2.1) send the current sub-model to ALL (??) available downstream EPs.
   2.2) routines of merging the results from all EPs.
3) Merge all results (from step 2) of all sub-models.
*/

// Heads-up:
// 1) https://github.com/microsoft/onnxruntime/blob/3170a48e60979ce1fb0d391cab7b0572bab90fff/onnxruntime/core/framework/graph_partitioner.cc#L464-L466
// 2) https://github.com/microsoft/onnxruntime/blob/efad5bbc5aed1717200d3e8f6ddd253394af4b99/include/onnxruntime/core/framework/execution_provider.h#L107-L119
class Partitioner {
 public:
  // Rules for either splitting an ONNX model itself
  // or partitioning a model across multiple EPs.
  // TODO: More to be defined.
  static constexpr uint8_t PartitionRule_PRIORITY = 1;
  static constexpr uint8_t PartitionRule_MODEL_SIZE= 2;
  static constexpr uint8_t PartitionRule_SHAPE = 4;
  static constexpr uint8_t PartitionRule_PERF_ESTIMATE = 8;
  static constexpr uint8_t PartitionRule_DEFAULT = PartitionRule_PRIORITY;

  explicit Partitioner(
      const std::unordered_map<const char*, onnxruntime::IExecutionProvider*>& ep_ptrs,
      uint8_t rules);

  explicit Partitioner(
      const std::unordered_map<const char*, onnxruntime::IExecutionProvider*>& ep_ptrs,
      uint8_t rules,
      const onnxruntime::GraphViewer* p_gv,
      const onnxruntime::IExecutionProvider::IKernelLookup* p_kl);

  ~Partitioner() = default;

  void SetGraphViewerPtr(const onnxruntime::GraphViewer* p_gv);
  void SetKernelLookupPtr(
      const onnxruntime::IExecutionProvider::IKernelLookup* p_kl);

  std::vector<std::unique_ptr<onnxruntime::ComputeCapability>> Partition();

  std::vector<std::unique_ptr<onnxruntime::ComputeCapability>> Partition_VitisAI();
  std::vector<std::unique_ptr<onnxruntime::ComputeCapability>> Partition_MIGraphX();
  std::vector<std::unique_ptr<onnxruntime::ComputeCapability>> Partition_ZenDNN();

  uint8_t GetRules() const;

 private:
  uint8_t rules_;
  // FIXME: Raw pointers vs smart pointers?
  const onnxruntime::GraphViewer* p_gv_{nullptr};
  const onnxruntime::IExecutionProvider::IKernelLookup* p_kl_{nullptr};
  onnxruntime::IExecutionProvider* p_vai_ep_{nullptr};
  onnxruntime::IExecutionProvider* p_migraphx_ep_{nullptr};
  onnxruntime::IExecutionProvider* p_zendnn_ep_{nullptr};
};

}  // namespace amduai
}  // namespace onnxruntime
