// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Standard libs/headers.
#include <memory>

// 1st-party libs/headers.
#include "core/providers/providers.h"
#include "core/framework/provider_options.h"

struct OrtAMDUnifiedProviderOptions;

namespace onnxruntime {

// The concrete implementation of the three methods are defined in
// onnxruntime/core/session/provider_bridge_ort.cc.
struct AMDUnifiedProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(
      const ProviderOptions& provider_options);
  static std::shared_ptr<IExecutionProviderFactory> Create(
      const OrtAMDUnifiedProviderOptions* p_options);
  static std::shared_ptr<IExecutionProviderFactory> Create(int device_id);
  // TODO: As AMD unified EP tries unifying Vitis AI EP, MIGraphX EP,
  // and ZenDNN EP, more `Create()` methods with different
  // input parameters might be needed.
};

}  // namespace onnxruntime
