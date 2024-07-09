// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Standard libs/headers.
#include <string>
#include <memory>

// 1st-party libs/headers.
#include "core/providers/shared_library/provider_api.h"
//#include "core/common/logging/logging.h"
//#include "core/common/logging/macros.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/common/logging/sinks/clog_sink.h"


namespace onnxruntime {

// We expected that we can use the natively implemented
// `LOGS_DEFAULT` macro, but it's not working as expected
// perhaps due to the severity level (kINFO vs kWARNING).
// So, we developed this logging tool with the philosophy
// of leveraging the ONNXRT infra as much as possible.
class UnifiedEPLogger final {
 public:
  static void Init(const std::string& logger_id = "AMD_Unified_EP_logger");
  //static void Log(const std::string& message,
  //    OrtLoggingLevel level = OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO);
  static void Log(const std::string& message,
      logging::Severity level = logging::Severity::kINFO);

 private:
  static std::unique_ptr<logging::LoggingManager> p_logging_manager_;
  //static std::unique_ptr<Ort::Logger> p_logger_;
  static std::unique_ptr<logging::Logger> p_logger_;
  static std::string logger_id_;
};

std::unique_ptr<logging::LoggingManager>
UnifiedEPLogger::p_logging_manager_ = nullptr;
//std::unique_ptr<Ort::Logger> UnifiedEPLogger::p_logger_ = nullptr;
std::unique_ptr<logging::Logger> UnifiedEPLogger::p_logger_ = nullptr;
std::string UnifiedEPLogger::logger_id_ = "";

void UnifiedEPLogger::Init(const std::string& logger_id) {
  // FIXME: `logger_id_` should always be synchronized
  // with `p_logging_manager_` and `p_logger_`.
  if (logger_id_.empty()) {
    logger_id_ = logger_id.empty() ? "AMD_Unified_EP_logger" : logger_id;
  }
  if (!p_logging_manager_) {
    p_logging_manager_ = std::make_unique<logging::LoggingManager>(
        std::make_unique<logging::CLogSink>(), logging::Severity::kINFO,
        false, logging::LoggingManager::InstanceType::Temporal);
  }
  if (!p_logger_) {
    //p_logger_ = std::make_unique<Ort::Logger>(reinterpret_cast<const OrtLogger*>(
    //    p_logging_manager_->CreateLogger(logger_id_).get()));
    p_logger_ = std::move(p_logging_manager_->CreateLogger(logger_id_));
  }
}

//void UnifiedEPLogger::Log(const std::string& message, OrtLoggingLevel level) {
void UnifiedEPLogger::Log(const std::string& message, logging::Severity level) {
  // XXX: A little bit of overhead.
  Init();
  //ORT_CXX_LOG((*p_logger_), level, (message.c_str()));
  switch (level) {
  case logging::Severity::kINFO:
    LOGS((*p_logger_), INFO) << message;
    break;
  case logging::Severity::kWARNING:
    LOGS((*p_logger_), WARNING) << message;
    break;
  case logging::Severity::kVERBOSE:
    LOGS((*p_logger_), VERBOSE) << message;
    break;
  case logging::Severity::kERROR:
    LOGS((*p_logger_), ERROR) << message;
    break;
  case logging::Severity::kFATAL:
    LOGS((*p_logger_), FATAL) << message;
    break;
  default:
    LOGS((*p_logger_), WARNING) << message;
    break;
  }
}

}  // namespace onnxruntime
