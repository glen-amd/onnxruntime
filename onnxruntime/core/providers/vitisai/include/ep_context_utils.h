#pragma once

// Standard headers/libs.
#include <vector>
#include <string>
#include <memory>

// 1st-party headers/libs.
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {

static constexpr const char* kEPContextOp = "EPContext";
static constexpr const char* kMainContextAttr = "main_context";
static constexpr const char* kEPCacheContextAttr = "ep_cache_context";
static constexpr const char* kEmbedModeAttr = "embed_mode";
static constexpr const char* kPartitionNameAttr = "partition_name";
static constexpr const char* kSourceAttr = "source";
static constexpr const char* kEPSDKVersionAttr = "ep_sdk_version";
static constexpr const char* kONNXModelFileNameAttr = "onnx_model_filename";
static constexpr const char* kNotesAttr = "notes";
static constexpr const char* kEPContextOpDomain = "com.microsoft";

std::unique_ptr<ONNX_NAMESPACE::FunctionProto>
ConvertIndexedSubGraphToFunctionProto(const IndexedSubGraph&, const Graph&);

std::unique_ptr<IndexedSubGraph> ConvertFunctionProtoToIndexedSubGraph(
    const std::unique_ptr<ONNX_NAMESPACE::FunctionProto>&);

std::string SerializeCapabilities(
    const std::vector<std::unique_ptr<ComputeCapability>>&, const Graph&);

void DeserializeCapabilities(
    const std::string&, std::vector<std::unique_ptr<ComputeCapability>>&);

std::string SerializeOrigialGraph(const GraphViewer&);

std::unique_ptr<Model> CreateEPContexModel(const GraphViewer&, const std::string&,
                                           const std::string&, const int64_t, const logging::Logger*);

void DumpEPContextModel(std::unique_ptr<Model>&, const std::string&);

bool ValidateEPContextNode(const Graph&);

std::string RetrieveEPContextCache(const Graph&, bool binary_mode = true);

std::unique_ptr<GraphViewer> RetrieveOriginalGraph(const Graph&);

bool GraphHasEPContextNode(const GraphViewer&);

bool FusedGraphHasEPContextNode(
    const std::vector<IExecutionProvider::FusedNodeAndGraph>&);

const Path& GetTopLevelModelPath(const GraphViewer&);

bool GetEPContextModelFileLocation(
    const std::string&, const PathString&, bool, PathString&);

// The file for EP context binary is in the same folder as the EP context model file.
PathString GetEPContextCacheFileLocation(const PathString&, const PathString&);

std::string Slurp(const fs::path&);

std::string GetBackendCompileCache(const fs::path&);

void RestoreBackendCompileCache(const fs::path&, const std::string&);

// Different from `onnxruntime::GraphViewer::GetNodesInTopologicalOrder()`.
std::vector<NodeIndex> GetNodeIndicesInTopologicalOrder(const GraphViewer&);

std::vector<const NodeArg*> FilterOutputNodeArgs(const Node&);

// TODO: VitisAI specifiec MD5 algorithm selection is pending.
std::string GetModelSignature(const GraphViewer&);

}  // namespace onnxruntime
