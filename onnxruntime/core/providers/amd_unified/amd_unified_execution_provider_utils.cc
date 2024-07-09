// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Standard headers/libs.
#include <filesystem>

// 1st-party headers/libs.
#include "core/common/common.h"
#include "core/providers/shared/common.h"
#include "core/common/path_string.h"

#include "./amd_unified_execution_provider_utils.h"


namespace onnxruntime {

std::vector<std::string> SplitStr(const std::string& str, char delim,
    size_t start_pos) {
  std::vector<std::string> res;
  std::stringstream ss(start_pos == 0 ? str : str.substr(start_pos));
  std::string item;

  while (std::getline(ss, item, delim)) {
    res.push_back(item);
  }

  return res;
}

std::vector<std::string> ParseDevicesStrRepr(const std::string& devices_str) {
  size_t colon_prefix_pos = devices_str.find(':');
  auto devices = SplitStr(devices_str, ',',
      colon_prefix_pos == std::string::npos ? 0 : colon_prefix_pos + 1);

  const std::string device_options[] = {"CPU", "GPU", "NPU"};
  for (const auto& d : device_options) {
    if (std::find(devices.begin(), devices.end(), d) == devices.end()) {
      ORT_THROW("Invalid device string: " + devices_str);
    }
  }

  return devices;
}

// The filename extension for a shared library is different per platform
#ifdef _WIN32
#define LIBRARY_PREFIX
#define LIBRARY_EXTENSION ORT_TSTR(".dll")
#elif defined(__APPLE__)
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".dylib"
#else
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".so"
#endif

static Provider* GetSubProviderPtr(const char* provider_type) {
  Provider* p_provider{nullptr};
  void* handle{nullptr};
  try {
    LOGS_DEFAULT(WARNING) << "UAI CWD: " << std::filesystem::current_path();
    auto& env = Provider_GetHost()->Env__Default();
    auto lib_location = env.GetRuntimePath() + PathString(ORT_TSTR("./"));
    LOGS_DEFAULT(WARNING) << "UAI Runtime path: " << lib_location.c_str();
    if (provider_type == kVitisAIExecutionProvider) {
      lib_location += PathString(
          LIBRARY_PREFIX ORT_TSTR("onnxruntime_providers_vitisai") LIBRARY_EXTENSION);
    } else if (provider_type == kMIGraphXExecutionProvider) {
      lib_location += PathString(
          LIBRARY_PREFIX ORT_TSTR("onnxruntime_providers_migraphx") LIBRARY_EXTENSION);
    } else if (provider_type == kZendnnExecutionProvider) {
      lib_location += PathString(
          LIBRARY_PREFIX ORT_TSTR("onnxruntime_providers_zendnn") LIBRARY_EXTENSION);
    } else {
      ORT_THROW("Invalid sub provider type: ", provider_type);
    }
    if (!std::filesystem::exists(lib_location)) {
      LOGS_DEFAULT(WARNING) << lib_location.c_str() << " not existing";
      return nullptr;
    }
    LOGS_DEFAULT(WARNING) << "Trying loading " << lib_location.c_str();
    ORT_THROW_IF_ERROR(env.LoadDynamicLibrary(lib_location, false, &handle));
    Provider* (*p_GetProvider)();
    LOGS_DEFAULT(WARNING) << "Trying getting symbols from " << lib_location.c_str();
    ORT_THROW_IF_ERROR(
        env.GetSymbolFromLibrary(handle, "GetProvider", (void**)&p_GetProvider));
    p_provider = p_GetProvider();
    p_provider->Initialize();
    LOGS_DEFAULT(WARNING) << lib_location.c_str() << " initialization done";
    return p_provider;
  } catch (const std::exception&) {
    if (handle) {
      if (p_provider) {
        p_provider->Shutdown();
      }
      // TODO: `Env::UnloadDynamicLibrary`.
    }
    throw;
  }
}

Provider* GetVitisAIProviderPtr() {
  static Provider* p_provider{nullptr};
  if (!p_provider) {
    p_provider = GetSubProviderPtr(kVitisAIExecutionProvider);
  }
  return p_provider;
}

Provider* GetMIGraphXProviderPtr() {
  static Provider* p_provider{nullptr};
  if (!p_provider) {
    p_provider = GetSubProviderPtr(kMIGraphXExecutionProvider);
  }
  return p_provider;
}

Provider* GetZenDNNProviderPtr() {
  static Provider* p_provider{nullptr};
  if (!p_provider) {
    p_provider = GetSubProviderPtr(kZendnnExecutionProvider);
  }
  return p_provider;
}

}  // namespace onnxruntime
