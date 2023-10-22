// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "./amd_unified_provider_factory.h"
#include "./amd_unified_provider_factory_creator.h"
#include "./amd_unified_execution_provider.h"

// 1st-party libs/headers.
#include "core/providers/shared_library/provider_api.h"
#include "core/framework/execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"

// Standard libs/headers.
#include <fstream>
#include <unordered_map>
#include <string>


using namespace onnxruntime;

namespace onnxruntime {

#if 0
void InitializeRegistry();
void DeleteRegistry();
#endif

std::unique_ptr<IExecutionProvider> AMDUnifiedProviderFactory::CreateProvider() {
  return std::make_unique<AMDUnifiedExecutionProvider>(ep_info_);
}

std::shared_ptr<IExecutionProviderFactory>
CreateExecutionProviderFactory_AMDUnified(
    const AMDUnifiedExecutionProviderInfo& ep_info) {
  //initialize_amd_unified_ep();
  return std::make_shared<AMDUnifiedProviderFactory>(ep_info);
}

std::shared_ptr<IExecutionProviderFactory>
AMDUnifiedProviderFactoryCreator::Create(
    const ProviderOptions& provider_options) {
  //initialize_amd_unified_ep();
  return std::make_shared<AMDUnifiedProviderFactory>(
      AMDUnifiedExecutionProviderInfo{provider_options});
}

#if 0
struct AMDUnified_Provider : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(
      int device_id) override {}

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(
      const void* provider_options) override {}

  void Initialize() override {
    InitializeRegistry();
  }

  void Shutdown() override {
    DeleteRegistry();
  }
};
#endif

}  // namespace onnxruntime
