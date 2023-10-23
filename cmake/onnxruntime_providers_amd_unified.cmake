# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if ("${GIT_COMMIT_ID}" STREQUAL "")
execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND git rev-parse HEAD
  OUTPUT_VARIABLE GIT_COMMIT_ID
  OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
#configure_file(${ONNXRUNTIME_ROOT}/core/providers/amd_unified/imp/version_info.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/AMD_Unified/version_info.h)
file(GLOB onnxruntime_providers_amd_unified_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/amd_unified/*.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/amd_unified/*.h"
)
#list(REMOVE_ITEM onnxruntime_providers_vitisai_cc_srcs "${ONNXRUNTIME_ROOT}/core/providers/vitisai/onnxruntime_vitisai_ep_stub.cc")
source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_amd_unified_cc_srcs})
onnxruntime_add_static_library(onnxruntime_providers_vitisai ${onnxruntime_providers_amd_unified_cc_srcs})
onnxruntime_add_include_to_target(onnxruntime_providers_amd_unified onnxruntime_common onnxruntime_framework onnx onnx_proto)
#onnxruntime_add_shared_library(onnxruntime_vitisai_ep ${ONNXRUNTIME_ROOT}/core/providers/vitisai/onnxruntime_vitisai_ep_stub.cc)
onnxruntime_add_include_to_target(onnxruntime_amd_unified_ep onnxruntime_common)
target_include_directories(onnxruntime_amd_unified_ep PRIVATE "${ONNXRUNTIME_ROOT}" "${ONNXRUNTIME_ROOT}/core/providers/amd_unified/include")
target_link_libraries(onnxruntime_providers_amd_unified PUBLIC onnxruntime_amd_unified_ep PRIVATE onnx protobuf::libprotobuf nlohmann_json::nlohmann_json )
#target_compile_definitions(onnxruntime_amd_unified_ep
#                         PRIVATE "-DONNXRUNTIME_VITISAI_EP_STUB=1" "-DONNXRUNTIME_VITISAI_EP_EXPORT_DLL=1")
if(NOT MSVC)
  target_compile_options(onnxruntime_providers_amd_unified PUBLIC $<$<CONFIG:DEBUG>:-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0>)
endif(NOT MSVC)

target_include_directories(onnxruntime_providers_amd_unified PRIVATE "${ONNXRUNTIME_ROOT}/core/providers/vitisai/include" ${XRT_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR}/AMD_Unified)
if(MSVC)
  target_compile_options(onnxruntime_providers_amd_unified PRIVATE "/Zc:__cplusplus")
  # for dll interface warning.
  target_compile_options(onnxruntime_providers_amd_unified PRIVATE "/wd4251")
  # for unused formal parameter
  target_compile_options(onnxruntime_providers_amd_unified PRIVATE "/wd4100")
else(MSVC)
  target_compile_options(onnxruntime_providers_amd_unified PRIVATE -Wno-unused-parameter)
endif(MSVC)

set_target_properties(onnxruntime_providers_amd_unified PROPERTIES FOLDER "ONNXRuntime")
set_target_properties(onnxruntime_providers_amd_unified PROPERTIES LINKER_LANGUAGE CXX)

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers_amd_unified
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
