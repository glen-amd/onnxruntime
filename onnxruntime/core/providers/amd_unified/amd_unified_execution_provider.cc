// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// 1st-party libs/headers.
//#include "core/graph/graph_utils.h"
#include "core/common/common.h"
//#include "core/session/custom_ops.h"
//#include "core/framework/execution_providers.h"

#include "./amd_unified_execution_provider.h"


using namespace ONNX_NAMESPACE;

namespace onnxruntime {

constexpr const char* AMD_UNIFIED = "AMD_UNIFIED";

AMDUnifiedExecutionProvider::AMDUnifiedExecutionProvider(
    const AMDUnifiedExecutionProviderInfo& ep_info)
  : IExecutionProvider{onnxruntime::kAMDUnifiedExecutionProvider},
    ep_info_(ep_info) {
  InitProviderOrtApi();
  // TODO
  //kernel_registry_ = std::make_shared<KernelRegistry>();
  //CreateKernelRegistry();

  auto& vai_ep_options =
    ep_info_.downstream_ep_options[kVitisAIExecutionProvider];
  // For Vitis AI EP, at least, the option "config_file" is required.
  if (!vai_ep_options.empty()) {
    auto* p_provider = GetVitisAIProviderPtr();
    if (p_provider) {
      std::unique_ptr<IExecutionProvider> p_vai_ep =
        std::move(p_provider->CreateExecutionProviderFactory(&vai_ep_options)->CreateProvider());
      SetVitisAIEPPtr(std::move(p_vai_ep));
    }
  } else {
    LOGS_DEFAULT(WARNING) << "Invalid ProviderOptions for AMD Vitis AI";
  }
  auto& migraphx_ep_options =
    ep_info_.downstream_ep_options[kMIGraphXExecutionProvider];
  if (!migraphx_ep_options.empty()) {
    auto* p_provider = GetMIGraphXProviderPtr();
    if (p_provider) {
      OrtMIGraphXProviderOptions ort_provider_options;
      p_provider->UpdateProviderOptions(&ort_provider_options, migraphx_ep_options);
      std::unique_ptr<IExecutionProvider> p_migraphx_ep =
        std::move(p_provider->CreateExecutionProviderFactory(&ort_provider_options)->CreateProvider());
      SetMIGraphXEPPtr(std::move(p_migraphx_ep));
    }
  } else {
    LOGS_DEFAULT(WARNING) << "Invalid ProviderOptions for AMD MIGraphX";
  }
  auto& zendnn_ep_options =
    ep_info_.downstream_ep_options[kZendnnExecutionProvider];
  if (!zendnn_ep_options.empty()) {
    auto* p_provider = GetZenDNNProviderPtr();
    if (p_provider) {
      OrtZendnnProviderOptions ort_provider_options;
      p_provider->UpdateProviderOptions(&ort_provider_options, zendnn_ep_options);
      std::unique_ptr<IExecutionProvider> p_zendnn_ep =
        std::move(p_provider->CreateExecutionProviderFactory(&ort_provider_options)->CreateProvider());
      SetZenDNNEPPtr(std::move(p_zendnn_ep));
    }
  } else {
    LOGS_DEFAULT(WARNING) << "Invalid ProviderOptions for AMD ZenDNN";
  }
  if (!p_vai_ep_ && !p_migraphx_ep_ && !p_zendnn_ep_) {
    ORT_THROW("No available backends for AMD Unified Execution Provider");
  }
}

AMDUnifiedExecutionProvider::~AMDUnifiedExecutionProvider() {
  //LOGS_DEFAULT(INFO) << "Destructing an AMDUnifiedExecutionProvider...";
  if (p_vai_ep_) {
    //LOGS_DEFAULT(INFO) << "Releasing a Vitis AI EP pointer...";
    p_vai_ep_.reset();
    p_vai_ep_ = nullptr;
  }
  if (p_migraphx_ep_) {
    //LOGS_DEFAULT(INFO) << "Releasing an MIGraphX EP pointer...";
    p_migraphx_ep_.reset();
    p_migraphx_ep_= nullptr;
  }
  if (p_zendnn_ep_) {
    //LOGS_DEFAULT(INFO) << "Releasing a ZenDNN EP pointer...";
    p_zendnn_ep_.reset();
    p_zendnn_ep_ = nullptr;
  }
  if (p_partitioner_) {
    //LOGS_DEFAULT(INFO) << "Releasing a UAI Partitioner pointer...";
    p_partitioner_.reset();
    p_partitioner_ = nullptr;
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

void AMDUnifiedExecutionProvider::InitializePartitionerPtr() const {
  if (!p_partitioner_) {
    //LOGS_DEFAULT(INFO) << "Initializing a UAI Partitioner pointer...";
    std::unordered_map<const char*, IExecutionProvider*> ep_ptrs;
    if (p_vai_ep_) {
      ep_ptrs[kVitisAIExecutionProvider] = p_vai_ep_.get();
    }
    if (p_migraphx_ep_) {
      ep_ptrs[kMIGraphXExecutionProvider] = p_migraphx_ep_.get();
    }
    if (p_zendnn_ep_) {
      ep_ptrs[kZendnnExecutionProvider] = p_zendnn_ep_.get();
    }
    p_partitioner_ = std::make_unique<amduai::Partitioner>(
        ep_ptrs, amduai::Partitioner::PartitionRule_DEFAULT);
  }
}

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

bool AMDUnifiedExecutionProvider::BackendReady(
    DownstreamEPRanks ep_rank) const {
  switch (ep_rank) {
    case DownstreamEPRanks::VITISAI:
      return p_vai_ep_ ? true : false;
    case DownstreamEPRanks::MIGRAPHX:
      return p_migraphx_ep_ ? true : false;
    case DownstreamEPRanks::ZENDNN:
      return p_zendnn_ep_ ? true : false;
    default:
      return false;
  }
}

// XXX: We expect callers to guarantee the correct type.
void AMDUnifiedExecutionProvider::SetVitisAIEPPtr(
    std::unique_ptr<IExecutionProvider>&& p_vai_ep) {
  if (!p_vai_ep_ && p_vai_ep) {
    p_vai_ep_ = std::move(p_vai_ep);
  }
}

// XXX: We expect callers to guarantee the correct type.
void AMDUnifiedExecutionProvider::SetMIGraphXEPPtr(
    std::unique_ptr<IExecutionProvider>&& p_migraphx_ep) {
  if (!p_migraphx_ep_ && p_migraphx_ep) {
    p_migraphx_ep_ = std::move(p_migraphx_ep);
  }
}

// XXX: We expect callers to guarantee the correct type.
void AMDUnifiedExecutionProvider::SetZenDNNEPPtr(
    std::unique_ptr<IExecutionProvider>&& p_zendnn_ep) {
  if (!p_zendnn_ep_ && p_zendnn_ep) {
    p_zendnn_ep_ = std::move(p_zendnn_ep);
  }
}

std::vector<std::unique_ptr<ComputeCapability>>
AMDUnifiedExecutionProvider::CombineDownstreamCapabilites(
    const onnxruntime::GraphViewer& graph_viewer,
    const IKernelLookup& kernel_lookup) const {
  //const ExecutionProviders& eps =
  //  curr_sess_.GetSessionState().GetExecutionProviders();

  return p_partitioner_->Partition();
}

common::Status AMDUnifiedExecutionProvider::CombineDownstreamCompilation(
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  //const ExecutionProviders& eps =
  //  curr_sess_.GetSessionState().GetExecutionProviders();

  if (p_vai_ep_) {
    auto status = DownstreamCompile(DownstreamEPRanks::VITISAI,
        fused_nodes_and_graphs, node_compute_funcs);
    if (!status.IsOK()) {
      // TODO: Logging.
      return status;
    }
  }
  if (p_migraphx_ep_) {
    auto status = DownstreamCompile(DownstreamEPRanks::MIGRAPHX,
        fused_nodes_and_graphs, node_compute_funcs);
    if (!status.IsOK()) {
      // TODO: Logging.
      return status;
    }
  }
  if (p_zendnn_ep_) {
    auto status = DownstreamCompile(DownstreamEPRanks::ZENDNN,
        fused_nodes_and_graphs, node_compute_funcs);
    if (!status.IsOK()) {
      // TODO: Logging.
      return status;
    }
  }
  return Status::OK();
}

common::Status AMDUnifiedExecutionProvider::DownstreamCompile(
    DownstreamEPRanks ep_rank,
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  //LOGS_DEFAULT(INFO) << "Compiling using EP " << std::to_string((uint8_t)ep_rank);
  std::vector<FusedNodeAndGraph> nodes_graphs;
  std::vector<NodeComputeInfo> compute_infos;
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const auto& attrs =
      const_cast<const Node&>(fused_node_graph.fused_node.get()).GetAttributes();
    auto attr_it = attrs.find("ep_rank");
    if (!(attr_it != attrs.end())) {
      //LOGS_DEFAULT(INFO) << "\"ep_rank\" attribute missing";
      continue;
    }
    if ((uint8_t)attr_it->second().i() != (uint8_t)ep_rank) {
      //LOGS_DEFAULT(INFO) << "Mismatched \"ep_rank\" attribute: expected "
      //  << std::to_string((uint8_t)ep_rank)
      //  << ", gotten " << std::to_string(attr_it->second().i());
      continue;
    }
    nodes_graphs.push_back(fused_node_graph);
  }
  switch (ep_rank) {
    case DownstreamEPRanks::VITISAI:
      if (p_vai_ep_) {
        auto status = p_vai_ep_->Compile(nodes_graphs, compute_infos);
        if (!status.IsOK()) {
          return status;
        }
      }
      break;
    case DownstreamEPRanks::MIGRAPHX:
      if (p_migraphx_ep_) {
        auto status = p_migraphx_ep_->Compile(nodes_graphs, compute_infos);
        if (!status.IsOK()) {
          return status;
        }
      }
      break;
    case DownstreamEPRanks::ZENDNN:
      if (p_zendnn_ep_) {
        auto status = p_zendnn_ep_->Compile(nodes_graphs, compute_infos);
        if (!status.IsOK()) {
          return status;
        }
      }
    case DownstreamEPRanks::NONE:
    default:
      break;
  }
  node_compute_funcs.insert(node_compute_funcs.end(),
      compute_infos.begin(), compute_infos.end());
  return Status::OK();
}

std::vector<std::unique_ptr<ComputeCapability>>
AMDUnifiedExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const IKernelLookup& kernel_lookup) const {
  //LOGS_DEFAULT(INFO) << "AMD Unified EP GetCapability...";

  if (!p_partitioner_) {
    InitializePartitionerPtr();
  }
  p_partitioner_->SetGraphViewerPtr(&graph_viewer);
  p_partitioner_->SetKernelLookupPtr(&kernel_lookup);

  return CombineDownstreamCapabilites(graph_viewer, kernel_lookup);
}

common::Status AMDUnifiedExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  //LOGS_DEFAULT(INFO) << "AMD Unified EP Compile...";
  return CombineDownstreamCompilation(fused_nodes_and_graphs,
                                      node_compute_funcs);
}

}  // namespace onnxruntime
