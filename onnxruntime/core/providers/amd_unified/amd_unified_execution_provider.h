// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// 1st-party libs/headers.
#include "core/framework/execution_provider.h"
#include "core/framework/customregistry.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/graph/constants.h"
// XXX: Once the unified EP is ready, we probably need to
// decommission current separate EPs such as Vitis AI EP,
// MIGraphX EP, and ZenDNN EP from ONNXRuntime, but they
// will still work as libraries (most likely shared libraries)
// to serve the unified EP.
#include "core/providers/vitisai/vitisai_execution_provider.h"
#include "./amd_unified_execution_provider_info.h"
#include "./amd_unified_execution_provider_utils.h"

// Standard libs.
#include <sstream>
#include <algorithm>


namespace onnxruntime {

class InferenceSession;

// Logical representation of AMD devices CPU/GPU/IPU/FPGA etc.
class AMDUnifiedExecutionProvider : public IExecutionProvider {
 public:
  explicit AMDUnifiedExecutionProvider(const AMDUnifiedExecutionProviderInfo&);
  ~AMDUnifiedExecutionProvider() = default;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer&, const IKernelLookup&) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>&,
      std::vector<NodeComputeInfo>&) override;

  // TODO: More methods (overriding the methods in `IExecutionProvider`)
  // need to be added on demand.
  //std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  std::shared_ptr<InferenceSession> GetCurrentSession() const;
  void SetCurrentSession(std::shared_ptr<InferenceSession> sess);

 private:
  //void CreateKernelRegistry();

  std::vector<std::unique_ptr<ComputeCapability>> CombineDownstreamCapabilites(
      const onnxruntime::GraphViewer&, const IKernelLookup&) const;

  common::Status CombineDownstreamCompilation(
      const std::vector<FusedNodeAndGraph>&, std::vector<NodeComputeInfo>&);

  AMDUnifiedExecutionProviderInfo ep_info_;
  //std::vector<OrtCustomOpDomain*> custom_op_domains_;
  //std::shared_ptr<KernelRegistry> kernel_registry_;

  std::shared_ptr<InferenceSession> curr_sess_;
};

}  // namespace onnxruntime
