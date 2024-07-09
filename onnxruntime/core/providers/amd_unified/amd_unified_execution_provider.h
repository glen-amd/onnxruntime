// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Standard libs/headers.
#include <sstream>
#include <algorithm>
#include <cstdint>

// 1st-party libs/headers.
#include "core/providers/shared_library/provider_api.h"
#include "core/session/onnxruntime_c_api.h"
//#include "core/framework/execution_provider.h"

#include "./amd_unified_execution_provider_info.h"
#include "./amd_unified_execution_provider_utils.h"
#include "./partitioner/partitioner.h"


namespace onnxruntime {

//class InferenceSession;

// Logical representation of AMD devices CPU/GPU/NPU etc.
// Unifiying AMD EPs such as VitisAI EP, MIGraphX EP, ZenDNN EP, etc.
class AMDUnifiedExecutionProvider : public IExecutionProvider {
 public:
  explicit AMDUnifiedExecutionProvider(const AMDUnifiedExecutionProviderInfo&);
  virtual ~AMDUnifiedExecutionProvider();

  ProviderOptions GetProviderOptions() const override {
    return AMDUnifiedExecutionProviderInfo::ToProviderOptions(ep_info_);
  }

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer&, const IKernelLookup&) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>&,
      std::vector<NodeComputeInfo>&) override;

  // TODO: More methods to be added to override
  // the methods declared in `IExecutionProvider`.
  //std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  void InitializePartitionerPtr() const;

  //std::shared_ptr<InferenceSession> GetCurrentSession() const;
  //void SetCurrentSession(std::shared_ptr<InferenceSession> sess);

  // XXX: Trade-off - raw pointers vs smart pointers?
  void SetVitisAIEPPtr(std::unique_ptr<IExecutionProvider>&& p_vai_ep);
  void SetMIGraphXEPPtr(std::unique_ptr<IExecutionProvider>&& p_migraphx_ep);
  void SetZenDNNEPPtr(std::unique_ptr<IExecutionProvider>&& p_zendnn_ep);
  bool BackendReady(DownstreamEPRanks) const;

 private:
  //void CreateKernelRegistry();

  std::vector<std::unique_ptr<ComputeCapability>> CombineDownstreamCapabilites(
      const onnxruntime::GraphViewer&, const IKernelLookup&) const;

  common::Status CombineDownstreamCompilation(
      const std::vector<FusedNodeAndGraph>&, std::vector<NodeComputeInfo>&);

  common::Status DownstreamCompile(DownstreamEPRanks,
      const std::vector<FusedNodeAndGraph>&, std::vector<NodeComputeInfo>&);

  AMDUnifiedExecutionProviderInfo ep_info_;
  //std::vector<OrtCustomOpDomain*> custom_op_domains_;
  //std::shared_ptr<KernelRegistry> kernel_registry_;

  // If downstream EPs (such as VitisAI EP and MIGraphX EP) are
  // not decommissioned, leveraging `InferenceSession` to access
  // the live EP instances may be better.
  // We opened a GitHub issue for this approach in ONNX Runtime official repo.
  //std::shared_ptr<InferenceSession> curr_sess_;

  mutable std::unique_ptr<amduai::Partitioner> p_partitioner_{nullptr};

  // A limited set of covered downstream EPs/libs.
  std::unique_ptr<IExecutionProvider> p_vai_ep_{nullptr};
  std::unique_ptr<IExecutionProvider> p_migraphx_ep_{nullptr};
  std::unique_ptr<IExecutionProvider> p_zendnn_ep_{nullptr};
};

}  // namespace onnxruntime
