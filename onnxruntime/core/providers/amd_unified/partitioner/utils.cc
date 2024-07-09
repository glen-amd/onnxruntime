// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "./utils.h"


namespace onnxruntime {
namespace amduai {

size_t GetModelSize(const onnxruntime::Model& model) {
  const onnxruntime::Path& model_path =
    const_cast<onnxruntime::Model&>(model).MainGraph().ModelPath();
  if (!model_path.IsEmpty()) {
    return std::filesystem::file_size(
        PathToUTF8String(model_path.ToPathString()));
  } else {
    // TODO: Logging.
  }
  auto p_model_proto = const_cast<onnxruntime::Model&>(model).ToProto();
  std::string model_str;
  p_model_proto->SerializeToString(model_str);
  return model_str.length();
}

// FIXME: Potential memory leak.
// XXX: Raw pointers vs smart pointers.
onnxruntime::Model* CloneModel(const onnxruntime::Model& model) {
  auto& logger = onnxruntime::logging::LoggingManager::DefaultLogger();
  auto p_model_proto = const_cast<onnxruntime::Model&>(model).ToProto();
  // XXX: `ModelProto(const ModelProto&) = delete;` in provider_wrappedtypes.h.
  auto&& temp_model_proto = *p_model_proto;
  const onnxruntime::PathString model_path_str =
    const_cast<onnxruntime::Model&>(model).MainGraph().ModelPath().ToPathString();
  // XXX: As of early March 2024, the signature of `Model::Create()` defined
  // in provider_wrappedtypes.h is still changing (with breaking changes).
  auto p_cloned_model = onnxruntime::Model::Create(
      std::forward<ONNX_NAMESPACE::ModelProto>(temp_model_proto),
      model_path_str, logger);
  // Necessary?
  auto status = p_cloned_model->MainGraph().Resolve();
  if (!status.IsOK()) {
    // TODO: Logging.
    return nullptr;
  }
  return p_cloned_model.release();
}

// XXX: Raw pointers vs smart pointers.
void DeleteModel(onnxruntime::Model* p_model) {
  if (p_model != nullptr) {
    delete p_model;
  }
}

size_t GetRoughLayerSize(onnxruntime::Node* p_node) {
  auto p_node_proto = ONNX_NAMESPACE::NodeProto::Create();
  // The input argument `p_node` belongs to
  // a cloned model which is already made mutable.
  p_node->ToProto(*p_node_proto, true);
  size_t layer_size = 0;
  for (int i = 0, n = p_node_proto->attribute_size(); i < n; ++i) {
    const auto& attr = p_node_proto->attribute(i);
    // FIXME: This condition may be not enough or even incorrect.
    if (attr.type() ==
        ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_TENSOR) {
      size_t num_units = 1;
      for (int j = 0, m = attr.t().dims().size(); j < m; ++j) {
        num_units *= attr.t().dims().Get(j);
      }
      switch (static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
            attr.t().data_type())) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          layer_size += num_units * sizeof(float);
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ:
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2:
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ:
          layer_size += num_units * 8;
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
        case ONNX_NAMESPACE::TensorProto_DataType_INT16:
        case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
          layer_size += num_units * 16;
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_INT32:
          layer_size += num_units * 32;
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:
          layer_size += num_units * 64;
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:
          layer_size += num_units * 128;
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_UINT4:
        case ONNX_NAMESPACE::TensorProto_DataType_INT4:
          layer_size += num_units * 4;
          break;
        //case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
        //case ONNX_NAMESPACE::TensorProto_DataType_STRING:
        //case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
        default:
          // TODO: Logging.
          break;
      }
    }
  }
  return layer_size;
}

#if 0
ONNX_NAMESPACE::ValueInfoProto FindValueInfoInGraph(
    const onnxruntime::Graph& graph, uint8_t fields, const std::string& name) {
  ONNX_NAMESPACE::GraphProto graph_proto = graph.ToProto();
  if ((fields & ValueInfoField_INPUT) != 0) {
    for (const auto& info: graph_proto.input()) {
      if (info.name() == name) {
        return info;
      }
    }
  }
  if ((fields & ValueInfoField_OUTPUT) != 0) {
    for (const auto& info: graph_proto.output()) {
      if (info.name() == name) {
        return info;
      }
    }
  }
  if ((fields & ValueInfoField_OTHER) != 0) {
    for (const auto& info: graph_proto.value_info()) {
      if (info.name() == name) {
        return info;
      }
    }
  }
  // TODO: Logging.
  return {};
}
#endif

// The parameters of indexes here are indexes of the vector
// where `NodeIndex`es are stored in topological order,
// NOT the `NodeIndex`es themselves.
#if 0
onnxruntime::Model CreateSubModel(const onnxruntime::Model& model,
    size_t start_index, size_t end_index) {
  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  ONNX_NAMESPACE::ModelProto sub_model_proto;

  // TODO: More to be set.
  // Ref.:
  // 1) model.h in the "onnxruntime" code base.
  // 2) onnx-ml.proto in the "onnx" code base.
  // 2.1) https://github.com/onnx/onnx/blob/a600e2fe60fcec561a448aabacc43033e5025220/onnx/onnx-ml.proto#L458-L460
  sub_model_proto.set_ir_version(model_proto.ir_version());
  sub_model_proto.set_domain(model_proto.domain());
  sub_model_proto.set_producer_name(model_proto.producer_name());
  sub_model_proto.set_producer_version(model_proto.producer_version());
  for (size_t i = start_index; i <= end_index; i++) {
    *sub_model_proto.mutable_graph()->add_node() = model_proto.graph().node(i);
  }
  for (const auto& input : model_proto.graph().node(start_index).input()) {
    *sub_model_proto.mutable_graph()->add_input() = FindValueInfoInGraph(
        model.MainGraph(), ValueInfoField_INPUT | ValueInfoField_OTHER, input);
  }
  for (const auto& output : model_proto.graph().node(end_index).output()) {
    *sub_model_proto.mutable_graph()->add_output() = FindValueInfoInGraph(
        model.MainGraph(), ValueInfoField_OUTPUT | ValueInfoField_OTHER, output);
  }
  for (const auto& initializer : model_proto.graph().initializer()) {
    bool used = false;
    for (const auto& node : sub_model_proto.graph().node()) {
      for (const auto& input : node.input()) {
        if (input == initializer.name()) {
          used = true;
          break;
        }
      }
      if (used) {
        break;
      }
    }
    if (used) {
      *sub_model_proto.mutable_graph()->add_initializer() = initializer;
    }
  }

  auto& logger = onnxruntime::logging::LoggingManager::DefaultLogger();
  auto sub_model = onnxruntime::Model(
      std::move(sub_model_proto), {}, nullptr, logger);
  auto status = sub_model.MainGraph().Resolve();
  if (!status.IsOK()) {
    // TODO: Logging.
    return {};
  }
  return sub_model;
}
#endif
std::unique_ptr<onnxruntime::Model> CreateSubModel(
    const onnxruntime::Model& model, size_t start_index, size_t end_index) {
  auto p_model_proto = const_cast<onnxruntime::Model&>(model).ToProto();
  auto& graph = const_cast<onnxruntime::Model&>(model).MainGraph();
  auto p_sub_model_proto = ONNX_NAMESPACE::ModelProto::Create();
  auto p_sub_graph_proto = p_sub_model_proto->mutable_graph();

  // As of March 2024, the IR version, domain, producer info, etc. of a model
  // can not be read/written with the shared-lib APIs defined in provider_api.h
  // as well as all related files.

  // Construct the sub graph proto by adding nodes one by one.
  for (size_t i = start_index; i <= end_index; i++) {
    auto p_node = graph.GetNode(i);
    auto p_sub_node_proto = p_sub_graph_proto->add_node();
    p_node->ToProto(*p_sub_node_proto, true);
  }

  auto& logger = onnxruntime::logging::LoggingManager::DefaultLogger();
  // XXX: `ModelProto(const ModelProto&) = delete;` in provider_wrappedtypes.h.
  auto&& temp_sub_model_proto = *p_sub_model_proto;
  // XXX: As of early March 2024, the signature of `Model::Create()` defined
  // in provider_wrappedtypes.h is still changing (with breaking changes).
  auto p_sub_model = onnxruntime::Model::Create(
      std::forward<ONNX_NAMESPACE::ModelProto>(temp_sub_model_proto), {}, logger);
  auto& sub_graph = p_sub_model->MainGraph();

  // Set the inputs of the sub graph.
  const auto p_first_node = graph.GetNode(start_index);
  // `ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept { return g_host->Node__InputDefs(this); }`
  auto node_inputs = p_first_node->InputDefs();
  std::vector<const NodeArg*> first_node_input_arg_ptrs;
  for (const auto& p_arg : node_inputs) {
    first_node_input_arg_ptrs.push_back(p_arg);
  }
  // `void SetInputs(gsl::span<const NodeArg* const> inputs) { g_host->Graph__SetInputs(this, inputs); }`
  sub_graph.SetInputs(first_node_input_arg_ptrs);

  // Set the outputs of the sub graph.
  const auto p_last_node = graph.GetNode(end_index);
  // `ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept { return g_host->Node__OutputDefs(this); }`
  auto node_outputs = p_last_node->OutputDefs();
  std::vector<const NodeArg*> last_node_output_arg_ptrs;
  for (const auto& p_arg : node_outputs) {
    last_node_output_arg_ptrs.push_back(p_arg);
  }
  // `void SetOutputs(gsl::span<const NodeArg* const> outputs) { return g_host->Graph__SetOutputs(this, outputs); }`
  sub_graph.SetOutputs(last_node_output_arg_ptrs);

  // TODO:
  // Set the initializers of the sub graph.
  //auto p_tensor_protos = graph.ToGraphProto()->mutable_initializer();

  return p_sub_model;
}

#if 0
std::vector<onnxruntime::Model> SplitModel(
    const onnxruntime::Model& model, size_t model_size) {
  // Original graph.
  auto graph = model.MainGraph();
  // Redundant.
  if (graph.NumberOfNodes() < 1) {
    return {};
  }

  if (model_size == 0) {
    model_size = GetModelSize(model);
  }

  size_t num_models = model_size / MAXIMUM_PROTOBUF;
  num_models += model_size % MAXIMUM_PROTOBUF == 0 ? 0 : 1;

  // Cloned model.
  onnxruntime::Model* p_cloned_model = CloneModel(model);
  // FIXME: What if clone fails?
  if (p_cloned_model == nullptr) {
    // Logging.
    return {};
  }
  // Cloned graph.
  auto cloned_graph = p_cloned_model->MainGraph();

  std::vector<onnxruntime::Model> split_models;
  size_t start_index = 0;
  size_t split_size = 0;
  auto node_indices =
    onnxruntime::GraphViewer(graph).GetNodesInTopologicalOrder();
  size_t l = node_indices.size();
  for (size_t i = 0; i < l; i++) {
    onnxruntime::NodeIndex ni = node_indices[i];
    onnxruntime::Node* p_node = cloned_graph.GetNode(ni);
    split_size += GetRoughLayerSize(p_node);
    if (split_size == MAXIMUM_PROTOBUF) {
      auto sub_model = CreateSubModel(model, start_index, i);
      split_models.push_back(std::move(sub_model));
      start_index = i + 1;
      split_size = 0;
    } else if (split_size > MAXIMUM_PROTOBUF) {
      auto sub_model = CreateSubModel(model, start_index, i - 1);
      split_models.push_back(std::move(sub_model));
      start_index = i;
      split_size = 0;
    }
  }
  if (split_models.size() < num_models) {
    auto sub_model = CreateSubModel(model, start_index, l - 1);
    split_models.push_back(std::move(sub_model));
  }

  DeleteModel(p_cloned_model);

  return split_models;
}
#endif

std::vector<std::unique_ptr<onnxruntime::Model>> SplitModel(
    const onnxruntime::Model& model, size_t model_size) {
  // Original graph.
  auto& graph = const_cast<onnxruntime::Model&>(model).MainGraph();
  // Redundant.
  if (graph.Nodes().empty()) {
    return {};
  }

  if (model_size == 0) {
    model_size = GetModelSize(model);
  }

  size_t num_models = model_size / MAXIMUM_PROTOBUF;
  num_models += model_size % MAXIMUM_PROTOBUF == 0 ? 0 : 1;

  // Cloned model that is mutable.
  onnxruntime::Model* p_cloned_model = CloneModel(model);
  // FIXME: What if clone fails?
  if (p_cloned_model == nullptr) {
    // Logging.
    return {};
  }
  // Cloned graph.
  auto& cloned_graph = p_cloned_model->MainGraph();

  // FIXME: memory leak.
  std::vector<std::unique_ptr<onnxruntime::Model>> split_model_ptrs;
  size_t start_index = 0;
  size_t split_size = 0;
  auto p_graph_viewer = graph.CreateGraphViewer();
  auto& node_indices = p_graph_viewer->GetNodesInTopologicalOrder();
  size_t l = node_indices.size();
  for (size_t i = 0; i < l; i++) {
    onnxruntime::NodeIndex ni = node_indices[i];
    onnxruntime::Node* p_node = cloned_graph.GetNode(ni);
    split_size += GetRoughLayerSize(p_node);
    if (split_size == MAXIMUM_PROTOBUF) {
      auto p_sub_model = CreateSubModel(model, start_index, i);
      split_model_ptrs.push_back(std::move(p_sub_model));
      start_index = i + 1;
      split_size = 0;
    } else if (split_size > MAXIMUM_PROTOBUF) {
      auto p_sub_model = CreateSubModel(model, start_index, i - 1);
      split_model_ptrs.push_back(std::move(p_sub_model));
      start_index = i;
      split_size = 0;
    }
  }
  if (split_model_ptrs.size() < num_models) {
    auto p_sub_model = CreateSubModel(model, start_index, l - 1);
    split_model_ptrs.push_back(std::move(p_sub_model));
  }

  DeleteModel(p_cloned_model);

  return split_model_ptrs;
}

}  // namespace amduai
}  // namespace onnxruntime
