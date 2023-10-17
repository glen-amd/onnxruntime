// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Standard libs.
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>


namespace onnxruntime {

// Different from onnxruntime::utils::SplitString defined in
// onnxruntime/core/common/string_utils.h
std::vector<std::string> SplitStr(const std::string&, char delim = ',',
    size_t start_pos = 0);

std::vector<std::string> ParseDevicesStrRepr(const std::string&);

}  // namespace onnxruntime
