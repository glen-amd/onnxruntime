// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// 1st-party libs/headers.
#include "onnxruntime_c_api.h"
#include "core/providers/providers.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param use_arena zero: false. non-zero: true.
 *
 * A dedicated SessionOptionsAppendExecutionProvider_<provider name> function
 * for `AMDUnifiedExecutionProvider`.
 * Ref.: `SessionOptionsAppendExecutionProvider` in
 * "include/onnxruntime/core/session/onnxruntime_c_api.h".
 */
ORT_EXPORT
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_AMD_Unified, _In_ OrtSessionOptions* options, int use_arena)
ORT_ALL_ARGS_NONNULL;

#ifdef __cplusplus
}
#endif

struct AMDUnifiedProviderFactory : IExecutionProviderFactory {
  AMDUnifiedProviderFactory(const AMDUnifiedExecutionProviderInfo& ep_info)
    : ep_info_(ep_info) {}
  ~VitisAIProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  VitisAIExecutionProviderInfo ep_info_;
};
