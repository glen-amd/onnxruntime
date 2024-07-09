// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __cplusplus
// Standard headers/libs.
#include <vector>
#include <string>
#include <utility>

// 1st-party headers/libs.
#include "core/framework/provider_options.h"
#include "onnxruntime_c_api.h"
// XXX: headers/libs for shared-lib EPs vs static-lib EPs.
//#include "core/session/onnxruntime_c_api.h"
//#include "core/framework/execution_provider.h"
//#include "core/session/abi_session_options_impl.h"
#include "core/providers/providers.h"

#include "./amd_unified_execution_provider_info.h"


namespace onnxruntime {

//class IAllocator;
//class IDataTransfer;
//struct IExecutionProviderFactory;
//enum class ArenaExtendStrategy : int32_t;

struct ProviderInfo_AMD_Unified {
  virtual std::vector<std::string> GetAvailableDevices() const = 0;
 protected:
  // Can only be destroyed through a subclass instance.
  ~ProviderInfo_AMD_Unified() = default;
};

struct AMDUnifiedProviderFactory : IExecutionProviderFactory {
  AMDUnifiedProviderFactory(const AMDUnifiedExecutionProviderInfo& ep_info)
    : ep_info_(ep_info) {}
  virtual ~AMDUnifiedProviderFactory() = default;

  virtual std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  AMDUnifiedExecutionProviderInfo ep_info_;
};

extern "C" {
#endif

/**
 * \param device_id MIGraphX/HIP/ROCm device Id
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_AMD_Unified,
               _In_ OrtSessionOptions* options, int device_id);

#ifdef __cplusplus
}

}  // namespace onnxruntime
#endif
