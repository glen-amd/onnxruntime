// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "./amd_unified_execution_provider_info.h"

// 1st-party libs/headers.
#include "core/framework/provider_options.h"
#include "onnxruntime_c_api.h"

namespace onnxruntime {

struct IExecutionProviderFactory;
struct AMDUnifiedExecutionProviderInfo;

struct AMDUnifiedProviderFactory : IExecutionProviderFactory {
  AMDUnifiedProviderFactory(const AMDUnifiedExecutionProviderInfo& ep_info)
    : ep_info_(ep_info) {}
  ~AMDUnifiedProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  AMDUnifiedExecutionProviderInfo ep_info_;
};

}  // namespace onnxruntime
