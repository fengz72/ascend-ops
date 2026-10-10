# GE custom op (custom_op_v2) for the prefix-attention macro kernel.
#
# Kernel binaries are compiled here from the checked-in codegen'd AscendC
# sources (../codegen) with bisheng -- NO cannbotdsl needed at build time.
# cannbotdsl is only used by scripts/dump_kernel_src.py to (re)generate the
# sources when the DSL kernel changes.
#
# fp16/bf16 device kernel symbols collide, so each dtype is a separate .so;
# libcust_opapi.so dlopens them (see src/dsl_launcher.cpp).
#
# Targets / outputs (installed under opp/):
#   kernel_fp16.so / kernel_bf16.so   macro kernel (bisheng ASC, from .asc)
#   aicpu_metadata.so                 AICPU planner (bisheng aicpu, from codegen/aicpu)
#   libcust_opapi.so                  GE custom op + launcher (CXX)
#   prefix_attention_op_proto         REG_OP-only proto lib for the ES generator
#   es_prefix_attn                    generated ES python wrapper (GE Session test)
#
# Configure:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release [-DNPU_ARCH=dav-3510]
#   cmake --build build -j && cmake --install build
cmake_minimum_required(VERSION 3.16)
project(prefix_attention_ge LANGUAGES CXX)

if(NOT ASCEND_HOME_PATH)
  set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "ASCEND_HOME_PATH is empty. Source CANN set_env.sh first.")
endif()
set(NPU_ARCH "dav-3510" CACHE STRING "bisheng --npu-arch (board-specific; Ascend950PR=dav-3510)")
set(BISHENG "${ASCEND_HOME_PATH}/bin/bisheng")
if(NOT EXISTS "${BISHENG}")
  message(FATAL_ERROR "bisheng not found at ${BISHENG}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0 google=ascend_private)

set(GE_INCLUDES
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/graph"
  "${ASCEND_HOME_PATH}/include/register"
  "${ASCEND_HOME_PATH}/include/external"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
set(CODEGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../codegen")
set(AICPU_DIR "${CODEGEN_DIR}/aicpu")
set(KERNEL_BUILD "${CMAKE_BINARY_DIR}/kernel_artifacts")
file(MAKE_DIRECTORY "${KERNEL_BUILD}")
set(BENV "${CMAKE_COMMAND}" -E env "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")

# --- macro kernel: .asc -> kernel_<dtype>.so (bisheng ASC language) ---
foreach(dtype fp16 bf16)
  set(ASC_SRC "${CODEGEN_DIR}/launch_prefix_macro_${dtype}.asc")
  set(K_OBJ "${KERNEL_BUILD}/kernel_${dtype}.o")
  set(K_SO "${KERNEL_BUILD}/kernel_${dtype}.so")
  add_custom_command(
    OUTPUT "${K_SO}"
    COMMAND ${BENV} "${BISHENG}" -fPIC --npu-arch=${NPU_ARCH} -c -O3 -x asc
            "${ASC_SRC}" -o "${K_OBJ}"
    COMMAND ${BENV} "${BISHENG}" -shared -fPIC --npu-arch=${NPU_ARCH}
            -o "${K_SO}" "${K_OBJ}" -lm
    DEPENDS "${ASC_SRC}"
    COMMENT "bisheng ASC: macro kernel ${dtype}"
    VERBATIM)
  list(APPEND KERNEL_SOS "${K_SO}")
endforeach()

# --- AICPU metadata planner: .aicpu + shim.asc -> aicpu_metadata.so ---
set(HCC "${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include")
set(AICPU_KERNEL_SRC "${AICPU_DIR}/_prefix_attn_metadata_kernel_kernel.aicpu")
set(AICPU_SHIM_SRC "${AICPU_DIR}/_prefix_attn_metadata_kernel_shim.asc")
set(AICPU_KOBJ "${KERNEL_BUILD}/aicpu_kernel.o")
set(AICPU_SOBJ "${KERNEL_BUILD}/aicpu_shim.o")
set(AICPU_SO "${KERNEL_BUILD}/aicpu_metadata.so")
add_custom_command(
  OUTPUT "${AICPU_SO}"
  COMMAND ${BENV} "${BISHENG}" -O2 -c -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0
          "-I${ASCEND_HOME_PATH}/asc/include/aicpu_api"
          "--cce-aicpu-L${ASCEND_HOME_PATH}/lib64/device/lib64"
          "--cce-aicpu-laicpu_api"
          "--cce-aicpu-sysroot=${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/sysroot"
          -isystem "${HCC}" -isystem "${HCC}/c++/7.3.0"
          -isystem "${HCC}/c++/7.3.0/aarch64-target-linux-gnu"
          -isystem "${HCC}/c++/7.3.0/backward"
          --cce-aicpu-launch-with-interface
          -x aicpu "${AICPU_KERNEL_SRC}" -o "${AICPU_KOBJ}"
  COMMAND ${BENV} "${BISHENG}" -fPIC --npu-arch=${NPU_ARCH} -c -O3
          --cce-aicpu-launch-with-interface -x asc
          "${AICPU_SHIM_SRC}" -o "${AICPU_SOBJ}"
  COMMAND ${BENV} "${BISHENG}" -shared -fPIC --npu-arch=${NPU_ARCH}
          -o "${AICPU_SO}" "${AICPU_KOBJ}" "${AICPU_SOBJ}" -lm
  DEPENDS "${AICPU_KERNEL_SRC}" "${AICPU_SHIM_SRC}"
          "${AICPU_DIR}/_prefix_attn_metadata_kernel_args.h"
          "${AICPU_DIR}/_prefix_attn_metadata_kernel_body.inc"
  COMMENT "bisheng aicpu: metadata planner"
  VERBATIM)
list(APPEND KERNEL_SOS "${AICPU_SO}")

add_custom_target(kernel_artifacts ALL DEPENDS ${KERNEL_SOS})

# --- runtime deliverable: GE custom op + DSL artifact launcher (CXX) ---
add_library(cust_opapi SHARED
  src/prefix_attention_proto.cc
  src/prefix_attention_op.cpp
  src/dsl_launcher.cpp
)
set_target_properties(cust_opapi PROPERTIES PREFIX "lib" OUTPUT_NAME "cust_opapi")
add_dependencies(cust_opapi kernel_artifacts)
target_include_directories(cust_opapi PRIVATE ${GE_INCLUDES})
target_link_directories(cust_opapi PRIVATE
  "${ASCEND_HOME_PATH}/lib64"
  "${ASCEND_HOME_PATH}/aarch64-linux/devlib/linux/aarch64"
)
target_link_libraries(cust_opapi PRIVATE ascendcl acl_rt register lowering dl)

# --- REG_OP-only proto library for the ES wrapper generator ---
add_library(prefix_attention_op_proto SHARED src/prefix_attention_proto.cc)
target_compile_definitions(prefix_attention_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(prefix_attention_op_proto PRIVATE -fvisibility=hidden)
target_include_directories(prefix_attention_op_proto PRIVATE
  "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/include/external"
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(prefix_attention_op_proto PRIVATE
  "${ASCEND_HOME_PATH}/lib64/libopp_registry.so"
  "${ASCEND_HOME_PATH}/lib64/libgraph.so"
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so")

# --- OPP package install (ASCEND_CUSTOM_OPP_PATH points at opp/) ---
set(OPP_LIB "${CMAKE_CURRENT_SOURCE_DIR}/opp/op_graph/lib/linux/aarch64")
install(TARGETS cust_opapi DESTINATION "${OPP_LIB}")
install(FILES ${KERNEL_SOS} DESTINATION "${OPP_LIB}/artifacts")
install(FILES "${CODEGEN_DIR}/SOURCE_INFO.json"
        DESTINATION "${OPP_LIB}/artifacts" RENAME MANIFEST.json)
install(FILES include/prefix_attention_op.h
        DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/opp/op_graph/include")

# --- ES python wrapper generation (online GE Session tests) ---
list(APPEND CMAKE_MODULE_PATH "${ASCEND_HOME_PATH}/include/ge/cmake")
find_package(GenerateEsPackage REQUIRED)
add_es_library(
  ES_LINKABLE_AND_ALL_TARGET es_prefix_attn
  OPP_PROTO_TARGET prefix_attention_op_proto
  OUTPUT_PATH ${CMAKE_BINARY_DIR}/es_output)
