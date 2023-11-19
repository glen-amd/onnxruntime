// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <ctime>

#include "core/framework/execution_provider.h"
#include "core/framework/customregistry.h"
#include "core/session/onnxruntime_c_api.h"

// we cannot include vaip/vaip.hpp here because header file referred by
// onnxruntime_pybind_state_common.cc
namespace vaip_core {
template <typename T>
class DllSafe;
class ExecutionProvider;
}  // namespace vaip_core
namespace onnxruntime {

// Information needed to construct execution providers.
struct VitisAIExecutionProviderInfo {
  VitisAIExecutionProviderInfo(const ProviderOptions& provider_options);

  const char* get_json_config_str(size_t i = 0) const {
    return json_configs_[i].c_str();
  }

 private:
  ProviderOptions provider_options_;
  const std::vector<std::string> json_configs_;
};

// Logical device representation.
class VitisAIExecutionProvider : public IExecutionProvider {
 public:
  explicit VitisAIExecutionProvider(const VitisAIExecutionProviderInfo& info);
  ~VitisAIExecutionProvider() = default;

  enum class CompilerRank : size_t {
    XCOMPILER = 0,
    TVM = 1,
  };

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const IKernelLookup& /*kernel_lookup*/) const override;

  int GetDeviceId() const { return 0; }

  common::Status Compile(
      const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
      std::vector<NodeComputeInfo>& node_compute_funcs) override;
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

 private:
  void CreateKernelRegistry();
  std::vector<std::unique_ptr<ComputeCapability>> GetCapabilityStandalone(
      size_t, const onnxruntime::GraphViewer&) const;
  //Model* CloneModel(const Model&);
  //Graph GenerateInterimGraph(const onnxruntime::GraphViewer&,
  //    const std::vector<std::unique_ptr<ComputeCapability>>&);
  void CombineCapabilities(std::vector<std::unique_ptr<ComputeCapability>>&,
      std::vector<std::unique_ptr<ComputeCapability>>&);
  common::Status CompileStandalone(size_t,
      const std::vector<FusedNodeAndGraph>&, std::vector<NodeComputeInfo>&);
  using my_ep_t = vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>>;
  using my_ep_uptr_t = std::shared_ptr<my_ep_t>;
  // we have to hide the implementation by forward declaration.
  mutable std::vector<mutable my_ep_uptr_t> execution_providers_group_;
  VitisAIExecutionProviderInfo info_;
  std::vector<OrtCustomOpDomain*> custom_op_domains_;
  std::shared_ptr<KernelRegistry> registry_;
  std::set<std::string> vitisai_optypes_;
};

}  // namespace onnxruntime
