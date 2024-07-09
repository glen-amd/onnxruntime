// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Standard headers/libs.
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

// 1st-party headers/libs.
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {

// Different from onnxruntime::utils::SplitString defined in
// onnxruntime/core/common/string_utils.h
std::vector<std::string> SplitStr(const std::string&, char delim = ',',
    size_t start_pos = 0);

std::vector<std::string> ParseDevicesStrRepr(const std::string&);

// XXX: raw pointers vs smart pointers.
Provider* GetVitisAIProviderPtr();
Provider* GetMIGraphXProviderPtr();
Provider* GetZenDNNProviderPtr();
}  // namespace onnxruntime
