// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Standard headers/libs.
#include <string>
#include <unordered_map>
#include <fstream>

// 1st-party headers/libs.
#include "core/providers/shared_library/provider_api.h"

#include "./amd_unified_execution_provider_info.h"


namespace onnxruntime {

// `onnxruntime::migraphx::provider_option_names` is defined
// in an implementation file rather than in a header file:
// core/providers/migraphx/migraphx_execution_provider_info.cc.

// Indices of the vector:
// index 0 -
// index 1 - DownstreamEPRanks::VITISAI
// index 2 - DownstreamEPRanks::MIGRAPHX
// index 3 - DownstreamEPRanks::ZENDNN
std::vector<std::unordered_set<std::string>> amd_provider_option_names{
  {"backends"},
  {"config_file", "cacheDir", "cacheKey"},
  {"device_id", "trt_fp16_enable", "migx_int8_enable", "migx_int8_calibration_table_name", "migx_int8_use_native_calibration_table"},
  {"use_arena", "threadpool_args"}
};

// FIXME:
// In the case where different downstream EPs share the same option names,
// there would be issues.
//
// The implementation here implies priorities.
AMDUnifiedExecutionProviderInfo::AMDUnifiedExecutionProviderInfo(
    const ProviderOptions& provider_options) {
  // TODO: `OrtAMDUnifiedProviderOptions::backends`
  //
  downstream_ep_options[kVitisAIExecutionProvider] = {};
  for (auto& option_name : amd_provider_option_names[(size_t)DownstreamEPRanks::VITISAI]) {
    auto it = provider_options.find(option_name);
    if (it != provider_options.end()) {
      downstream_ep_options[kVitisAIExecutionProvider][option_name] = it->second;
    }
  }
  downstream_ep_options[kMIGraphXExecutionProvider] = {};
  for (auto& option_name : amd_provider_option_names[(size_t)DownstreamEPRanks::MIGRAPHX]) {
    auto it = provider_options.find(option_name);
    if (it != provider_options.end()) {
      downstream_ep_options[kMIGraphXExecutionProvider][option_name] = it->second;
    }
  }
  downstream_ep_options[kZendnnExecutionProvider] = {};
  for (auto& option_name : amd_provider_option_names[(size_t)DownstreamEPRanks::ZENDNN]) {
    auto it = provider_options.find(option_name);
    if (it != provider_options.end()) {
      downstream_ep_options[kZendnnExecutionProvider][option_name] = it->second;
    }
  }
}

ProviderOptions AMDUnifiedExecutionProviderInfo::ToProviderOptions(
    const AMDUnifiedExecutionProviderInfo& ep_info) {
  ProviderOptions options;
  for (const auto& it : ep_info.downstream_ep_options) {
    // FIXME: Insertion may fail due to duplicate keys.
    for (const auto& it2 : it.second) {
      options.insert(it2);
    }
  }
  return options;
}

}  // namespace onnxruntime
