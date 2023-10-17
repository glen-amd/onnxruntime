// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "./amd_unified_execution_provider_utils.h"


namespace onnxruntime {

std::vector<std::string> SplitStr(const std::string& str, char delim = ',',
    size_t start_pos = 0) {
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

  const std::string device_options[] = {"CPU", "GPU", "FGGA"};
  for (auto& d : device_options) {
    if (std::find(devices.begin(), devices.end(), d) == devices.end()) {
      ORT_THROW("Invalid device string: " + devices_str);
    }
  }

  return devices;
}

}  // namespace onnxruntime
