// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "./global_apis.h"

using namespace onnxruntime;

#if 0
std::vector<OrtCustomOpDomain*> initialize_amd_unified_ep() {
  // FIXME: common::Status or Ort::Status?
  Status status = Status::OK();
  try {
    OrtEnv::LoggingManagerConstructionInfo lm_info{nullptr, nullptr,
      ORT_LOGGING_LEVEL_WARNING, "onnxruntime-amd-unified-ep"};
    std::ignore = OrtEnv::GetInstance(lm_info, status);
  } catch (onnxruntime::OnnxRuntimeException& /*e*/) {
  }
  auto domains = std::vector<OrtCustomOpDomain*>();
  // XXX
  domains.reserve(100);
  onnxruntime_amd_unified_ep::initialize_onnxruntime_amd_unified_ep(
    create_org_api_hook(), domains);
  auto& domainToVersionRangeInstance =
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
  if (domainToVersionRangeInstance.Map().find("com.xilinx") ==
      domainToVersionRangeInstance.Map().end()) {
    vaip::register_xir_ops(domains);
  }

  return domains;
}
#endif
