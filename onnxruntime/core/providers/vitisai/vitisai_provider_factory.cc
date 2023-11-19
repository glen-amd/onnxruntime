// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "vitisai_provider_factory_creator.h"

#include "vaip/global_api.h"
#include "./vitisai_execution_provider.h"
#include "core/framework/execution_provider.h"
#include "core/common/logging/logging.h"

#include "core/session/abi_session_options_impl.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <unordered_map>
#include <string>

using namespace onnxruntime;
using json = nlohmann::json;
namespace onnxruntime {

// Different from onnxruntime::utils::SplitString defined in
// onnxruntime/core/common/string_utils.h
static std::vector<std::string> SplitStr(const std::string& str,
    char delim = ',', size_t start_pos = 0) {
  std::vector<std::string> res;
  std::stringstream ss(start_pos == 0 ? str : str.substr(start_pos));
  std::string item;

  while (std::getline(ss, item, delim)) {
    res.push_back(item);
  }

  return res;
}

static std::string ConfigToJsonStr(const std::unordered_map<std::string, std::string>& config) {
  const auto& filename = config.at("config_file");
  std::ifstream f(filename);
  json data = json::parse(f);
  for (const auto& entry : config) {
    data[entry.first] = entry.second;
  }
  return data.dump();
}

// Provider options example:
// {"config_file": "/etc/vaip_config.json","/etc/vaip_config_gemm_asr.json"}
VitisAIExecutionProviderInfo::VitisAIExecutionProviderInfo(
    const ProviderOptions& provider_options)
  : provider_options_(provider_options) {
  // Possible keys at the same level as the key "config_file":
  // "cacheDir", "cacheKey", "encryptionKey", "onnxPath" (??), "version" (??).
  // References in VAIP:
  // doc/vitis_ai_ep_tutorial.md
  // python_doc/source/session_option/cache_option.rst
  // python_doc/source/session_option/encryption.rst
  // Note: we assume the key "config_file" always exists.
  const auto& filenames = SplitStr(provider_options.at("config_file"));
  json_configs_.reserve(filenames.size());
  auto cache_dir_it = provider_options.find("cacheDir");
  const auto& cache_dirs = cache_dir_it != provider_options.end() ?
    SplitStr(cache_dir_it->second) : std::vector<std::string>();
  auto cache_key_it = provider_options.find("cacheKey");
  const auto& cache_keys = cache_key_it != provider_options.end() ?
    SplitStr(cache_key_it->second) : std::vector<std::string>();
  LOGS_DEFAULT(WARNING) << "Parsing JSON config: " << '\n';
  for (size_t i = 0, l = filenames.size(); i < l; i++) {
    std::ifstream f(filenames[i]);
    json temp = json::parse(f);
    temp["config_file"] = filenames[i];
    if (cache_dirs.size() > i) {
      temp["cacheDir"] = cache_dirs[i];
    }
    if (cache_keys.size() > i) {
      temp["cacheKey"] = cache_keys[i];
    }
    for (const auto& entry : provider_options) {
      if (entry.first == "config_file" || entry.first == "cacheDir"
          || entry.first == "cacheKey") {
        continue;
      }
      temp[entry.first] = entry.second;
    }
    json_configs_[i] = temp.dump();
    LOGS_DEFAULT(WARNING) << json_configs_[i] << '\n';
  }
}

struct VitisAIProviderFactory : IExecutionProviderFactory {
  VitisAIProviderFactory(const VitisAIExecutionProviderInfo& info) : info_(info) {}
  ~VitisAIProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  VitisAIExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> VitisAIProviderFactory::CreateProvider() {
  return std::make_unique<VitisAIExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory>
CreateExecutionProviderFactory_VITISAI(const VitisAIExecutionProviderInfo& info) {
  initialize_vitisai_ep();
  return std::make_shared<VitisAIProviderFactory>(info);
}

std::shared_ptr<IExecutionProviderFactory> VitisAIProviderFactoryCreator::Create(const ProviderOptions& provider_options) {
  initialize_vitisai_ep();
  auto info = VitisAIExecutionProviderInfo{provider_options};
  return std::make_shared<VitisAIProviderFactory>(info);
}

}  // namespace onnxruntime
