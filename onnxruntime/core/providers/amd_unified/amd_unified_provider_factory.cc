// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Standard headers/libs.
#include <fstream>
#include <unordered_map>
#include <iostream>

// 1st-party headers/libs.
#include "core/providers/shared_library/provider_api.h"
#include "core/platform/env.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/common/common.h"

#include "./amd_unified_provider_factory_creator.h"
#include "./amd_unified_execution_provider_utils.h"
#include "./amd_unified_provider_factory.h"
#include "./amd_unified_execution_provider_info.h"
#include "./amd_unified_execution_provider.h"


namespace onnxruntime {

#if 0
void InitializeRegistry();
void DeleteRegistry();
#endif

// The EP info of AMD Unified EP should cover
// the info/options of all downstream EPs.
std::unique_ptr<IExecutionProvider>
AMDUnifiedProviderFactory::CreateProvider() {
  auto p_amd_unified_ep =
    std::make_unique<AMDUnifiedExecutionProvider>(ep_info_);
#if 0
  auto& vai_ep_options =
    ep_info_.downstream_ep_options[kVitisAIExecutionProvider];
  // For Vitis AI EP, at least, the option "config_file" is required.
  if (!vai_ep_options.empty()) {
    auto* p_provider = GetVitisAIProviderPtr();
    if (p_provider) {
      std::unique_ptr<IExecutionProvider> p_vai_ep =
        std::move(p_provider->CreateExecutionProviderFactory(&vai_ep_options)->CreateProvider());
      p_amd_unified_ep->SetVitisAIEPPtr(std::move(p_vai_ep));
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
      p_provider->UpdateProviderOptions(
          reinterpret_cast<void*>(&ort_provider_options), migraphx_ep_options);
      std::unique_ptr<IExecutionProvider> p_migraphx_ep =
        std::move(p_provider->CreateExecutionProviderFactory(&ort_provider_options)->CreateProvider());
      p_amd_unified_ep->SetMIGraphXEPPtr(std::move(p_migraphx_ep));
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
      p_provider->UpdateProviderOptions(
          reinterpret_cast<void*>(&ort_provider_options), zendnn_ep_options);
      std::unique_ptr<IExecutionProvider> p_zendnn_ep =
        std::move(p_provider->CreateExecutionProviderFactory(&ort_provider_options)->CreateProvider());
      p_amd_unified_ep->SetZenDNNEPPtr(std::move(p_zendnn_ep));
    }
  } else {
    LOGS_DEFAULT(WARNING) << "Invalid ProviderOptions for AMD ZenDNN";
  }
  if (!p_amd_unified_ep->BackendReady(DownstreamEPRanks::VITISAI) &&
      !p_amd_unified_ep->BackendReady(DownstreamEPRanks::MIGRAPHX) &&
      !p_amd_unified_ep->BackendReady(DownstreamEPRanks::ZENDNN)) {
    ORT_THROW("No available backends for AMD Unified Execution Provider");
  }
#endif
  p_amd_unified_ep->InitializePartitionerPtr();
  return p_amd_unified_ep;
}

#if 0
std::shared_ptr<IExecutionProviderFactory>
AMDUnifiedProviderFactoryCreator::Create(
    const ProviderOptions& provider_options) {
  return std::make_shared<AMDUnifiedProviderFactory>(
      AMDUnifiedExecutionProviderInfo{provider_options});
}
#endif

struct AMD_Unified_Provider : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(
      int device_id) override {
    AMDUnifiedExecutionProviderInfo ep_info;
    ep_info.downstream_ep_options[kMIGraphXExecutionProvider] = {
      {"device_id", std::to_string(device_id)}
    };
    return std::make_shared<AMDUnifiedProviderFactory>(ep_info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(
      const void* p_ort_ep_options_struct) override {
    auto ptr = reinterpret_cast<const OrtAMDUnifiedProviderOptions*>(p_ort_ep_options_struct);
    AMDUnifiedExecutionProviderInfo ep_info;
    ep_info.downstream_ep_options[kVitisAIExecutionProvider]["config_file"] =
      ptr->config_file;
    ep_info.downstream_ep_options[kVitisAIExecutionProvider]["cacheDir"] =
      ptr->cache_dir;
    ep_info.downstream_ep_options[kVitisAIExecutionProvider]["cacheDir"] =
      ptr->cache_key;
    ep_info.downstream_ep_options[kMIGraphXExecutionProvider]["device_id"] =
      std::to_string(ptr->device_id);
    ep_info.downstream_ep_options[kMIGraphXExecutionProvider]["trt_fp16_enable"] =
      std::to_string(ptr->migraphx_fp16_enable);
    ep_info.downstream_ep_options[kMIGraphXExecutionProvider]["migx_int8_enable"] =
      std::to_string(ptr->migraphx_int8_enable);
    ep_info.downstream_ep_options[kMIGraphXExecutionProvider]["migx_int8_calibration_table_name"] =
      ptr->migraphx_int8_calibration_table_name;
    ep_info.downstream_ep_options[kMIGraphXExecutionProvider]["migx_int8_use_native_calibration_table"] =
      std::to_string(ptr->migraphx_use_native_calibration_table);
    ep_info.downstream_ep_options[kZendnnExecutionProvider]["use_arena"] = std::to_string(ptr->use_arena);
    //ep_info.downstream_ep_options[kZendnnExecutionProvider]["threadpool_args"] = ptr->threadpool_args;
    return std::make_shared<AMDUnifiedProviderFactory>(ep_info);
  }

  // FIXME: In the base abstract struct `Provider`, there isn't any method with the signature like
  // `std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const ProviderOptions&)`,
  // the implementation below would cause compilation error in "provider_bridge_ort.cc".
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(
      const ProviderOptions& provider_options) override {
    AMDUnifiedExecutionProviderInfo ep_info(provider_options);
    return std::make_shared<AMDUnifiedProviderFactory>(ep_info);
  }

  // FIXME: In order to initialize the resources of downstream EPs,
  // must we implement this method?
  void Initialize() override {
    //InitializeRegistry();
  }

  // FIXME: In order to release the resources of downstream EPs,
  // must we implement this method?
  void Shutdown() override {
    //DeleteRegistry();
  }
} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}

}
