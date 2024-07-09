// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Standard headers/libs.
#include <filesystem>
#include <string>
#include <vector>
#include <memory>
#include <utility>

// 3rd-party headers/libs.
//#include "onnx/defs/attr_proto_util.h"

// 1st-party headers/libs.
#include "core/common/status.h"
//#include "core/graph/onnx_protobuf.h"
#include "core/providers/shared_library/provider_api.h"


namespace onnxruntime {
namespace amduai {

// `input`, `output`, and `value_info` fields of `message GraphProto`
// in onnx-ml.proto in the "onnx" Git repo.
constexpr const uint8_t ValueInfoField_INPUT = 1;
constexpr const uint8_t ValueInfoField_OUTPUT = 2;
constexpr const uint8_t ValueInfoField_OTHER = 4;

// Size limit of a Protobuf file - 2GB.
constexpr const size_t MAXIMUM_PROTOBUF = 2147483648LU;

size_t GetModelSize(const onnxruntime::Model& model);

// FIXME: Potential memory leak.
onnxruntime::Model* CloneModel(const onnxruntime::Model& model);
void DeleteModel(onnxruntime::Model* p_model);

size_t GetRoughLayerSize(onnxruntime::Node* p_node);

// It's not needed any longer because of the difference
// between static-lib APIs and shard-lib APIs.
//ONNX_NAMESPACE::ValueInfoProto FindValueInfoInGraph(
//    const onnxruntime::Graph& graph, uint8_t fields, const std::string& name);

// The index-related parameters here are indexes of the vector
// where `NodeIndex`es are stored in topological order,
// NOT the `NodeIndex`es themselves.
//onnxruntime::Model CreateSubModel(const onnxruntime::Model& model,
//    size_t start_index, size_t end_index);
std::unique_ptr<onnxruntime::Model> CreateSubModel(
    const onnxruntime::Model& model, size_t start_index, size_t end_index);

// FIXME: Potential memory leak.
std::vector<std::unique_ptr<onnxruntime::Model>> SplitModel(
    const onnxruntime::Model& model, size_t model_size = 0);

}  // namespace amduai
}  // namespace onnxruntime
