// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "amd_unified_execution_provider.h"

// 1st-party libs/headers.
#include "core/graph/graph_utils.h"
#include "core/common/common.h"
#include "core/session/custom_ops.h"
#include "core/framework/execution_providers.h"


using namespace ONNX_NAMESPACE;

namespace onnxruntime {

constexpr const char* AMD_UNIFIED = "AMD_UNIFIED";

struct MyCustomOpKernal : OpKernel {
  MyCustomOpKernal(const OpKernelInfo& info, const OrtCustomOp& op)
    : OpKernel(info), op_(op) {
      op_kernel_ = op_.CreateKernel(&op_, OrtGetApiBase()->GetApi(op_.version),
          reinterpret_cast<const OrtKernelInfo*>(&info));
  }

  ~MyCustomOpKernal() override {
    op_.KernelDestroy(op_kernel_);
  }

  Status Compute(OpKernelContext* ctx) const override {
    op_.KernelCompute(op_kernel_, reinterpret_cast<OrtKernelContext*>(ctx));
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MyCustomOpKernal);

  const OrtCustomOp& op_;
  void* op_kernel_;
};

AMDUnifiedExecutionProvider::AMDUnifiedExecutionProvider(
    const AMDUnifiedExecutionProviderInfo& ep_info)
  : IExecutionProvider{onnxruntime::kAMDUnifiedExecutionProvider},
  ep_info_(ep_info) {
    // FIXME:
    // In summary, `initialize_amd_unified_ep()` basically does two things:
    // 1) Specifiy the implementation of function pointers in `struct OrtApi`;
    // 2) Creation and registration of EP-custom op domains and ops/kernels.
    //custom_op_domains_ = initialize_amd_unified_ep();
    //kernel_registry_ = std::make_shared<KernelRegistry>();
    //CreateKernelRegistry();
}

AMDUnifiedExecutionProvider::~AMDUnifiedExecutionProvider() {
  if (vitisai_ep_ptr_) {
    vitisai_ep_ptr_.reset();
    vitisai_ep_ptr_ = nullptr;
  }
}

// TODO
#if 0
void AMDUnifiedExecutionProvider::CreateKernelRegistry() {
  for (const auto& domain : custom_op_domains_) {
    for (const auto* op : domain->custom_ops_) {
      KernelDefBuilder def_builder;
      def_builder.SetName(op->GetName(op));
      def_builder.SetDomain(domain->domain_);
      def_builder.SinceVersion(1);
      if (op->version > 12) {
        auto input_count = op->GetInputTypeCount(op);
        for (auto i = 0u; i < input_count; i++) {
          def_builder.InputMemoryType(op->GetInputMemoryType(op, i), i);
        }
      }
      def_builder.Provider(onnxruntime::kAMDUnifiedExecutionProvider);
      KernelCreateFn kernel_create_fn =
        [op](FuncManager&, const OpKernelInfo& info,
            std::unique_ptr<OpKernel>& out) -> Status {
          out = std::make_unique<MyCustomOpKernel>(info, *op);
          return Status::OK();
        };
      std::ignore = registry_->Register(def_builder, kernel_create_fn);
      amd_unified_optypes_.insert(op->GetName(op));
    }
  }
}

std::shared_ptr<KernelRegistry>
AMDUnifiedExecutionProvider::GetKernelRegistry() const {
  return kernel_registry_;
}
#endif

#if 0
std::shared_ptr<InferenceSession>
AMDUnifiedExecutionProvider::GetCurrentSession() const {
  // FIXME: We should log this.
  if (curr_sess_.unique()) {
    curr_sess_.reset();
    curr_sess_ = nullptr;
  }
  return curr_sess_;
}

void AMDUnifiedExecutionProvider::SetCurrentSession(
    std::shared_ptr<InferenceSession>& sess) {
  // FIXME: We should log this.
  if (!curr_sess_) {
    curr_sess_ = sess;
  }
}
#endif

void AMDUnifiedExecutionProvider::SetVitisAIEPPtr(
  std::unique_ptr<VitisAIExecutionProvider> vitisai_ep_ptr) {
    if (!vitisai_ep_ptr_) {
      vitisai_ep_ptr_ = std::move(vitisai_ep_ptr);
    }
}

// XXX: This getter method is not needed.
const VitisAIExecutionProvider&
AMDUnifiedExecutionProvider::GetVitisAIEP() const {
  return *vitisai_ep_ptr_;
}

// TODO: When the unified EP is fully ready, we need to combine
// the `ComputeCapability`s from all covered downstream decommissioned EPs.
// We need to figure out the algorithm of combining downstream
// `ComputeCapability`s, such as the order/priority, overlap, etc.
std::vector<std::unique_ptr<ComputeCapability>>
AMDUnifiedExecutionProvider::CombineDownstreamCapabilites(
    const onnxruntime::GraphViewer& graph,
    const IKernelLookup& kernel_lookup) const {
  //const ExecutionProviders& eps =
  //  curr_sess_.GetSessionState().GetExecutionProviders();
  return vitisai_ep_ptr_->GetCapability(graph, kernel_lookup);
}

// TODO: When the unified EP is fully ready, we need to combine
// the `NodeComputeInfo`s from all covered downstream decommissioned EPs.
// We need to figure out the algorithm of combining downstream
// `NodeComputeInfo`s, such as the order/priority, overlap, etc
common::Status AMDUnifiedExecutionProvider::CombineDownstreamCompilation(
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  //const ExecutionProviders& eps =
  //  curr_sess_.GetSessionState().GetExecutionProviders();
  return vitisai_ep_ptr_->Compile(fused_nodes_and_graphs, node_compute_funcs);
}

std::vector<std::unique_ptr<ComputeCapability>>
AMDUnifiedExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const IKernelLookup& kernel_lookup) const {
  return CombineDownstreamCapabilites(graph, kernel_lookup);
}

common::Status VitisAIExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  return CombineDownstreamCompilation(fused_nodes_and_graphs,
                                      node_compute_funcs);
}

}  // namespace onnxruntime
