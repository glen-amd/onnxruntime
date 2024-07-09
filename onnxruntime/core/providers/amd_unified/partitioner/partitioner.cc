// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// 3rd-party headers/libs.
//#include "onnx/defs/attr_proto_util.h"

// 1st-party headers/libs.
//#include "core/graph/node_attr_utils.h"

#include "./utils.h"
#include "./partitioner.h"


namespace onnxruntime {
namespace amduai {

// FIXME: Raw pointers vs smart pointers?
Partitioner::Partitioner(
    const std::unordered_map<const char*, onnxruntime::IExecutionProvider*>& ep_ptrs,
    uint8_t rules) : rules_(rules), p_gv_(nullptr), p_kl_(nullptr) {
  auto it = ep_ptrs.find(onnxruntime::kVitisAIExecutionProvider);
  if (it != ep_ptrs.end() && it->second) {
    p_vai_ep_ = it->second;
  }
  it = ep_ptrs.find(onnxruntime::kMIGraphXExecutionProvider);
  if (it != ep_ptrs.end() && it->second) {
    p_migraphx_ep_ = it->second;
  }
  it = ep_ptrs.find(onnxruntime::kZendnnExecutionProvider);
  if (it != ep_ptrs.end() && it->second) {
    p_zendnn_ep_= it->second;
  }
}

// FIXME: Raw pointers vs smart pointers?
Partitioner::Partitioner(
    const std::unordered_map<const char*, onnxruntime::IExecutionProvider*>& ep_ptrs,
    uint8_t rules,
    const onnxruntime::GraphViewer* p_gv,
    const onnxruntime::IExecutionProvider::IKernelLookup* p_kl)
  : rules_(rules), p_gv_(p_gv), p_kl_(p_kl) {
  auto it = ep_ptrs.find(onnxruntime::kVitisAIExecutionProvider);
  if (it != ep_ptrs.end() && it->second) {
    p_vai_ep_ = it->second;
  }
  it = ep_ptrs.find(onnxruntime::kMIGraphXExecutionProvider);
  if (it != ep_ptrs.end() && it->second) {
    p_migraphx_ep_ = it->second;
  }
  it = ep_ptrs.find(onnxruntime::kZendnnExecutionProvider);
  if (it != ep_ptrs.end() && it->second) {
    p_zendnn_ep_= it->second;
  }
}

// FIXME: Check the pointer validity and act accordingly.
void Partitioner::SetGraphViewerPtr(const onnxruntime::GraphViewer* p_gv) {
  p_gv_ = p_gv;
}

// FIXME: Check the pointer validity and act accordingly.
void Partitioner::SetKernelLookupPtr(
    const onnxruntime::IExecutionProvider::IKernelLookup* p_kl) {
  p_kl_ = p_kl;
}

std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
Partitioner::Partition_VitisAI() {
  if (!p_vai_ep_) {
    // TODO: Logging.
    return {};
  }

  std::vector<std::unique_ptr<onnxruntime::ComputeCapability>> capability_ptrs;

  const onnxruntime::Model& model = p_gv_->GetGraph().GetModel();
  auto model_size = GetModelSize(model);
  if (model_size > MAXIMUM_PROTOBUF) {
    rules_ |= Partitioner::PartitionRule_MODEL_SIZE;
    auto split_model_ptrs = SplitModel(model, model_size);
    for (auto& p_sub_model : split_model_ptrs) {
      auto p_graph_viewer = p_sub_model->MainGraph().CreateGraphViewer();
      // FIXME: Since `p_kl_` is originally used for the original whole model,
      // do we need to deal with `p_kl_` specially
      // when using it for split models?
      auto ptrs = p_vai_ep_->GetCapability(*p_graph_viewer, *p_kl_);
      for (auto& p : ptrs) {
        capability_ptrs.push_back(std::move(p));
      }
    }
  } else {
    // TODO: Potential memory optimization.
    // Keep it in mind - in "Structured Programming With GoTo Statements"
    // by Donald Knuth:
    // "Programmers waste enormous amounts of time thinking about,
    // or worrying about, the speed of noncritical parts of their programs,
    // and these attempts at efficiency actually have a strong negative impact
    // when debugging and maintenance are considered. We should forget about
    // small efficiencies, say about 97% of the time:
    // premature optimization is the root of all evil.
    // Yet we should not pass up our opportunities in that critical 3%."
    capability_ptrs = p_vai_ep_->GetCapability(*p_gv_, *p_kl_);
  }

  auto p_attr_proto = ONNX_NAMESPACE::AttributeProto::Create();
  p_attr_proto->set_name("ep_rank");
  p_attr_proto->set_i((int64_t)onnxruntime::DownstreamEPRanks::VITISAI);
  for (size_t i = 0, n = capability_ptrs.size(); i < n; ++i) {
    auto& p_subgraph = capability_ptrs[i]->SubGraph();
    auto p_meta_def = const_cast<ONNX_NAMESPACE::IndexedSubGraph_MetaDef*>(
        p_subgraph->GetMetaDef());
    auto& attrs = p_meta_def->attributes();
    attrs.insert_or_assign("ep_rank", *p_attr_proto);
  }

  return capability_ptrs;
}

std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
Partitioner::Partition_MIGraphX() {
  if (!p_migraphx_ep_) {
    return {};
  }
  auto capability_ptrs = p_migraphx_ep_->GetCapability(*p_gv_, *p_kl_);
  auto p_attr_proto = ONNX_NAMESPACE::AttributeProto::Create();
  p_attr_proto->set_name("ep_rank");
  p_attr_proto->set_i((int64_t)onnxruntime::DownstreamEPRanks::MIGRAPHX);
  for (size_t i = 0, n = capability_ptrs.size(); i < n; ++i) {
    auto& p_subgraph = capability_ptrs[i]->SubGraph();
    auto p_meta_def = const_cast<ONNX_NAMESPACE::IndexedSubGraph_MetaDef*>(
        p_subgraph->GetMetaDef());
    auto& attrs = p_meta_def->attributes();
    attrs.insert_or_assign("ep_rank", *p_attr_proto);
  }
  return capability_ptrs;
}

std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
Partitioner::Partition_ZenDNN() {
  if (!p_zendnn_ep_) {
    return {};
  }
  auto capability_ptrs = p_zendnn_ep_->GetCapability(*p_gv_, *p_kl_);
  auto p_attr_proto = ONNX_NAMESPACE::AttributeProto::Create();
  p_attr_proto->set_name("ep_rank");
  p_attr_proto->set_i((int64_t)onnxruntime::DownstreamEPRanks::ZENDNN);
  for (size_t i = 0, n = capability_ptrs.size(); i < n; ++i) {
    auto& p_subgraph = capability_ptrs[i]->SubGraph();
    auto p_meta_def = const_cast<ONNX_NAMESPACE::IndexedSubGraph_MetaDef*>(
        p_subgraph->GetMetaDef());
    auto& attrs = p_meta_def->attributes();
    attrs.insert_or_assign("ep_rank", *p_attr_proto);
  }
  return capability_ptrs;
}

std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
Partitioner::Partition() {
  std::unordered_set<std::vector<size_t>, SingleElemVectorHasher> node_indexes_set;
  std::vector<std::unique_ptr<onnxruntime::ComputeCapability>> capability_ptrs;
  if (p_vai_ep_) {
    auto ptrs = Partition_VitisAI();
    if ((rules_ & Partitioner::PartitionRule_PRIORITY) != 0) {
      for (size_t i = 0, n = ptrs.size(); i < n; ++i) {
        auto& p_subgraph = ptrs[i]->SubGraph();
        if (p_subgraph->Nodes().size() == 1) {
          node_indexes_set.insert(p_subgraph->Nodes());
        }
        capability_ptrs.push_back(std::move(ptrs[i]));
      }
    } else {
      for (auto& p : ptrs) {
        capability_ptrs.push_back(std::move(p));
      }
    }
  }
  if (p_migraphx_ep_) {
    auto ptrs = Partition_MIGraphX();
    if ((rules_ & Partitioner::PartitionRule_PRIORITY) != 0) {
      for (size_t i = 0, n = ptrs.size(); i < n; ++i) {
        auto& p_subgraph = ptrs[i]->SubGraph();
        if (p_subgraph->Nodes().size() == 1) {
          node_indexes_set.insert(p_subgraph->Nodes());
        }
        capability_ptrs.push_back(std::move(ptrs[i]));
      }
    } else {
      for (auto& p : ptrs) {
        capability_ptrs.push_back(std::move(p));
      }
    }
  }
  if (p_zendnn_ep_) {
    auto ptrs = Partition_ZenDNN();
    if ((rules_ & Partitioner::PartitionRule_PRIORITY) != 0) {
      for (size_t i = 0, n = ptrs.size(); i < n; ++i) {
        auto& p_subgraph = ptrs[i]->SubGraph();
        if (p_subgraph->Nodes().size() == 1) {
          node_indexes_set.insert(p_subgraph->Nodes());
        }
        capability_ptrs.push_back(std::move(ptrs[i]));
      }
    } else {
      for (auto& p : ptrs) {
        capability_ptrs.push_back(std::move(p));
      }
    }
  }
  return capability_ptrs;
}

uint8_t Partitioner::GetRules() const {
  return rules_;
}

}  // namespace amduai
}  // namespace onnxruntime
