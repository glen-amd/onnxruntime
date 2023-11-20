// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/graph_utils.h"
#include "vitisai_execution_provider.h"

#include <cassert>
#include <codecvt>
#include <fstream>
#include <istream>

#include "core/common/common.h"
#include "core/common/logging/logging.h"

#include "vaip/capability.h"
#include "vaip/global_api.h"
#include "core/session/custom_ops.h"
#include "core/session/inference_session.h"

#include "onnxruntime_vitisai_ep/onnxruntime_vitisai_ep.h"
#include "./imp/vai_assert.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

constexpr const char* VITISAI = "VITISAI";

static vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>> compile_onnx_model(
    const onnxruntime::GraphViewer& graph_viewer,
    const logging::Logger& logger, const char* json_config) {
#ifndef _WIN32
  auto model_path = graph_viewer.ModelPath().ToPathString();
#else
  using convert_t = std::codecvt_utf8<wchar_t>;
  std::wstring_convert<convert_t, wchar_t> strconverter;
  auto model_path = strconverter.to_bytes(graph_viewer.ModelPath().ToPathString());
#endif
  return onnxruntime_vitisai_ep::compile_onnx_model_3(model_path, graph_viewer.GetGraph(), json_config);
}

struct MyCustomOpKernel : OpKernel {
  MyCustomOpKernel(const OpKernelInfo& info, const OrtCustomOp& op) : OpKernel(info), op_(op) {
    op_kernel_ = op_.CreateKernel(&op_, OrtGetApiBase()->GetApi(op_.version),
                                  reinterpret_cast<const OrtKernelInfo*>(&info));
  }

  ~MyCustomOpKernel() override { op_.KernelDestroy(op_kernel_); }

  Status Compute(OpKernelContext* ctx) const override {
    op_.KernelCompute(op_kernel_, reinterpret_cast<OrtKernelContext*>(ctx));
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MyCustomOpKernel);

  const OrtCustomOp& op_;
  void* op_kernel_;
};

VitisAIExecutionProvider::VitisAIExecutionProvider(
    const VitisAIExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kVitisAIExecutionProvider}, info_(info) {
  custom_op_domains_ = initialize_vitisai_ep();
  registry_ = std::make_shared<KernelRegistry>();
  CreateKernelRegistry();
  execution_providers_group_.reserve(2);
}

void VitisAIExecutionProvider::CreateKernelRegistry() {
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
      def_builder.Provider(onnxruntime::kVitisAIExecutionProvider);
      KernelCreateFn kernel_create_fn = [op](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
        out = std::make_unique<MyCustomOpKernel>(info, *op);
        return Status::OK();
      };
      std::ignore = registry_->Register(def_builder, kernel_create_fn);
      vitisai_optypes_.insert(op->GetName(op));
    }
  }
}

std::shared_ptr<KernelRegistry> VitisAIExecutionProvider::GetKernelRegistry() const { return registry_; }

std::vector<std::unique_ptr<ComputeCapability>>
VitisAIExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
    const IKernelLookup& /*kernel_lookup*/) const {
  if (graph.IsSubgraph()) {
    // VITIS AI EP not support sungraph. Assigned to CPU.
    return {};
  }
  // XCompiler
  auto capability_ptrs1 = GetCapabilityStandalone(0, graph);
  // TVM
  auto capability_ptrs2 = GetCapabilityStandalone(1, graph);
  CombineCapabilities(capability_ptrs1, capability_ptrs2);
  return capability_ptrs1;
}

// compiler_rank:
// 0 - XCompiler
// 1 - TVM
std::vector<std::unique_ptr<ComputeCapability>>
VitisAIExecutionProvider::GetCapabilityStandalone(size_t compiler_rank,
    const onnxruntime::GraphViewer& graph) const {
  LOGS_DEFAULT(WARNING) << "Getting compute capabilities for the compiler "
    << compiler_rank;
  if (graph.IsSubgraph()) {
    LOGS_DEFAULT(WARNING) << "The graph for the compiler "
      << compiler_rank << " is a sub-graph";
    // Vitis AI EP does not support sub-graphs.
    return {};
  }
  if (execution_providers_group_[compiler_rank]) {
    // In the context of Vitis AI EP, a model can only be compiled once at most.
    return {};
  }
  auto opt_str = info_.get_json_config_str(compiler_rank);
  execution_providers_group_[compiler_rank] = std::make_unique<my_ep_t>(
      compile_onnx_model(graph, *GetLogger(), opt_str));
  auto result = vaip::GetComputeCapabilityOps(
      graph, execution_providers_group_[compiler_rank].get(), vitisai_optypes_);
  size_t index = 0u;
  for (auto& ep : **execution_providers_group_[compiler_rank]) {
    result.emplace_back(vaip::XirSubgraphToComputeCapability1(
          graph, ep.get(), index));
    index += 1;
  }
  LOGS_DEFAULT(WARNING) << "Getting compute capabilities for the compiler "
    << compiler_rank << " done";
  return result;
}

#if 0
// XXX: It would be better if we could clone a graph only,
// instead of cloning an owning model.
Model* VitisAIExecutionProvider::CloneModel(const Model& model) {
  auto& logger = logging::LoggingManager::DefaultLogger();
  auto model_proto = const_cast<onnxruntime::Model&>(model).ToProto();
  auto model_path = model.ModelPath().ToPathString();
  auto cloned_model = std::make_unique<Model>(
      std::move(model_proto), model_path, nullptr, logger);
  auto status = cloned_model->MainGraph().Resolve();
  vai_assert(status.IsOK(), status.ErrorMessage());
  return cloned_model.release();
}

Graph VitisAIExecutionProvider::GenerateInterimGraph(
    const onnxruntime::GraphViewer& graph,
    const std::vector<std::unique_ptr<ComputeCapability>>& capability_ptrs) {
  auto interim_model = CloneModel(graph.GetGraph().GetModel());
  auto interim_graph = interim_model.MainGraph();
  std::unordered_set<onnxruntime::NodeIndex> node_indices;
  for (auto& ptr : capability_ptrs) {
    const auto& nodes = ptr->sub_graph->nodes;
    node_indices.insert(nodes.begin(), nodes.end());
  }
  // Step 1: Remove out edges of this specified node.
  // Step 2: Remove this specified node.
  for (auto& node_index : node_indices) {
    auto node = interim_graph->GetNode(node_index);
    for (auto& output_edge : node->GetRelationships().output_edges) {
      interim_graph->RemoveEdge(output_edge.src_node, output_edge.dst_node,
          output_edge.src_arg_index, output_edge.dst_arg_index);
    }
    interim_graph->RemoveNode(node_index);
  }
  // TODO: Step 3: Add new input edges connected with all out nodes.
  return interim_graph;
}
#endif

struct SingleElemVectorHasher {
  size_t operator()(const std::vector<size_t>& vec) const {
    return vec[0];
  }
};

void VitisAIExecutionProvider::CombineCapabilities(
    std::vector<std::unique_ptr<ComputeCapability>>& capability_ptrs1,
    std::vector<std::unique_ptr<ComputeCapability>>& capability_ptrs2) const {
  std::unordered_set<std::vector<size_t>, SingleElemVectorHasher> vec_set;
  for (const auto& p : capability_ptrs1) {
    // 1. Overlap is allowed. Ref.:
    // https://github.com/microsoft/onnxruntime/blob/9364c05170d78c4516886dc91ec86afdce06ad6d/include/onnxruntime/core/framework/execution_provider.h#L110-L111
    // 2. In order to avoid complication (e.g., changes to `MetaDef`) and
    // broken sub-graphs, we don't touch sub-grpahs containing 1+ nodes,
    // which are generated by the function `XirSubgraphToComputeCapability1`.
    if (p->sub_graph->nodes.size() == 1) {
      vec_set.insert(p->sub_graph->nodes);
    }
  }
  //for (const auto& p : capability_ptrs2) {
  for (auto& p : capability_ptrs2) {
    if (p->sub_graph->nodes.size() == 1) {
      if (vec_set.count(p->sub_graph->nodes) == 0) {
        LOGS_DEFAULT(WARNING) << "Combining the node " << p->sub_graph->nodes[0];
        auto p_capability =
          std::make_unique<ComputeCapability>(std::move(p->sub_graph));
        capability_ptrs1.push_back(std::move(p_capability));
      }
    } else {
      LOGS_DEFAULT(WARNING) << "Combining a sub-graph consisting of "
        << p->sub_graph->nodes.size() << " nodes.";
      //capability_ptrs1.push_back(std::move(p));
      capability_ptrs1.push_back(
          std::forward<std::unique_ptr<ComputeCapability>>(std::move(p)));
      //capability_ptrs1.emplace_back(p);
    }
  }
}

void VitisAIExecutionProvider::CompileStandalone(size_t compiler_rank,
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  // FIXME: Is it OK to iterate over the same `FusedNodeAndGraph`s twice?
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    NodeComputeInfo compute_info;
    const onnx::AttributeProto* attr = graph_utils::GetNodeAttribute(
        fused_node_graph.fused_node, "index");
    if (attr == nullptr) {
      continue;
    }
    size_t index = (size_t)attr->i();
    compute_info.create_state_func =
      [this, compiler_rank, index](ComputeContext* context, FunctionState* state) {
        auto* p =
          (**this->execution_providers_group_[compiler_rank])[index]->compile().release();
        *state = p;
        return 0;
      };
    compute_info.release_state_func = [](FunctionState state) {
      if (state) {
        delete reinterpret_cast<vaip_core::CustomOp*>(state);
      }
    };
    compute_info.compute_func =
      [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
        reinterpret_cast<vaip_core::CustomOp*>(state)->Compute(api, context);
        return Status::OK();
      };
    node_compute_funcs.push_back(compute_info);
  }
}

common::Status VitisAIExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (size_t i = 0, l = execution_providers_group_.size(); i < l; i++) {
    CompileStandalone(i, fused_nodes_and_graphs, node_compute_funcs);
  }
  return Status::OK();
}

}  // namespace onnxruntime
