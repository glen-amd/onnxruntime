// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "./amd_unified_execution_provider_info.h"
#include "./amd_unified_execution_provider_utils.h"

// 3rd-party libs/headers.
#include "nlohmann/json.hpp"

// Standard libs/headers.
#include <string>
#include <unordered_map>
#include <fstream>


using json = nlohmann::json;

namespace onnxruntime {

static std::string ConfigToJsonStr(
    const std::unordered_map<std::string, std::string>& config) {
  const auto& filename = config.at("config_file");
  std::ifstream f(filename);
  json data = json::parse(f);
  for (const auto& entry : config) {
    data[entry.first] = entry.second;
  }
  return data.dump();
}

AMDUnifiedExecutionProviderInfo::AMDUnifiedExecutionProviderInfo(
    const ProviderOptions& provider_options)
  : provider_options(provider_options),
    json_config_{ConfigToJsonStr(provider_options)} {}

AMDUnifiedExecutionProviderInfo::AMDUnifiedExecutionProviderInfo(
    const std::string& device_types_str) {
    device_types = ParseDevicesStrRepr(device_types_str);
}

}  // namespace onnxruntime
