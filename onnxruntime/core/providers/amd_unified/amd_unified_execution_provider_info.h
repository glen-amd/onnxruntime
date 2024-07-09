// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Standard libs/headers.
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>

// 1st-party libs/headers.
#include "core/framework/provider_options.h"
//#include "core/graph/constants.h"
//#include "core/providers/shared_library/provider_api.h"


namespace onnxruntime {

enum class DownstreamEPRanks : uint8_t {
  NONE = 0,
  VITISAI = 1,
  MIGRAPHX = 2,
  ZENDNN = 3,
};

// A combination of `ProviderOptions`s of different downstream EPs.
// As this needs to cover the `ProviderOptions` or EP info of all
// downstream EPs, flexibility and extensibility are important.
// So, the initial design is following the principle of simplicity.
struct AMDUnifiedExecutionProviderInfo {
  std::unordered_map<const char*, ProviderOptions> downstream_ep_options;
  //std::vector<std::string> device_types;

  AMDUnifiedExecutionProviderInfo() = default;
  explicit AMDUnifiedExecutionProviderInfo(const ProviderOptions&);

  //static AMDUnifiedExecutionProviderInfo FromProviderOptions(
  //    const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(
      const AMDUnifiedExecutionProviderInfo& ep_info);
};

}  // namespace onnxruntime
