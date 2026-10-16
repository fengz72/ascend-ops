# GE custom op (custom_op_v2) for the prefix-attention macro kernel.
#
# Builds the checked-in codegen'd sources (codegen/) with bisheng -- no
# cannbotdsl needed at build time (only dump_kernel_src.py uses it).
# Kernel artifacts are linked into libcust_opapi.so (DT_NEEDED + RUNPATH
# $ORIGIN/artifacts) -- no dlopen at runtime.
#
# Outputs, installed into $ASCEND_HOME_PATH/opp/op_graph (GE finds them via
# ASCEND_CUSTOM_OPP_PATH=$ASCEND_HOME_PATH/opp):
#   kernel_fp16.so    macro kernel, fp16-only (see README)
#   aicpu_metadata.so AICPU planner
#   libcust_opapi.so  GE custom op + launchers (CXX)
#
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
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
set(CODEGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/codegen")
set(KERNEL_BUILD "${CMAKE_BINARY_DIR}/kernel_artifacts")
file(MAKE_DIRECTORY "${KERNEL_BUILD}")
set(BENV "${CMAKE_COMMAND}" -E env "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")

# --- macro kernel (fp16 only): .asc -> kernel_fp16.so via bisheng ---
# The .asc files are codegen'd (dump_kernel_src.py); fp16-only, see README.
set(ASC_SRC "${CODEGEN_DIR}/launch_prefix_macro_fp16.asc")
set(K_OBJ "${KERNEL_BUILD}/kernel_fp16.o")
set(K_SO "${KERNEL_BUILD}/kernel_fp16.so")
add_custom_command(
  OUTPUT "${K_SO}"
  COMMAND ${BENV} "${BISHENG}" -fPIC --npu-arch=${NPU_ARCH} -c -O3 -x asc
          "${ASC_SRC}" -o "${K_OBJ}"
  # SONAME so DT_NEEDED records the file name, not the build path.
  # -Bsymbolic-functions: without it the two artifacts' embedded bisheng
  # runtime glue cross-binds and their per-.so registries cross-wire
  # (AICPU launch failure 0x91).
  COMMAND ${BENV} "${BISHENG}" -shared -fPIC --npu-arch=${NPU_ARCH}
          -Wl,-soname,kernel_fp16.so -Wl,-Bsymbolic-functions
          -o "${K_SO}" "${K_OBJ}" -lm
  DEPENDS "${ASC_SRC}"
  COMMENT "bisheng ASC: macro kernel fp16"
  VERBATIM)
set(KERNEL_SOS "${K_SO}")

# --- AICPU metadata planner: .aicpu + shim.asc -> aicpu_metadata.so ---
# Sunk per-node task, re-plans from live bounds on every graph run.  The
# kernel's scratch is a per-thread TLS arena -- safe under concurrency.
set(HCC "${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include")
set(AICPU_DIR "${CODEGEN_DIR}/aicpu")
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
          -Wl,-soname,aicpu_metadata.so -Wl,-Bsymbolic-functions
          -o "${AICPU_SO}" "${AICPU_KOBJ}" "${AICPU_SOBJ}" -lm
  DEPENDS "${AICPU_KERNEL_SRC}" "${AICPU_SHIM_SRC}"
          "${AICPU_DIR}/_prefix_attn_metadata_kernel_args.h"
          "${AICPU_DIR}/_prefix_attn_metadata_kernel_body.inc"
  COMMENT "bisheng aicpu: metadata planner"
  VERBATIM)
list(APPEND KERNEL_SOS "${AICPU_SO}")

add_custom_target(kernel_artifacts ALL DEPENDS ${KERNEL_SOS})

# --- runtime deliverable: GE custom op + launchers (CXX) ---
# Links the kernel artifacts (DT_NEEDED); RUNPATH resolves them relative to
# this .so: build tree $ORIGIN/kernel_artifacts, install $ORIGIN/artifacts.
add_library(kernel_fp16_artifact SHARED IMPORTED)
set_target_properties(kernel_fp16_artifact PROPERTIES
  IMPORTED_LOCATION "${K_SO}" IMPORTED_SONAME "kernel_fp16.so")
add_library(aicpu_metadata_artifact SHARED IMPORTED)
set_target_properties(aicpu_metadata_artifact PROPERTIES
  IMPORTED_LOCATION "${AICPU_SO}" IMPORTED_SONAME "aicpu_metadata.so")

add_library(cust_opapi SHARED
  src/kernel_launcher.cc
  src/prefix_attention_op.cc
  src/prefix_attn_eager.cc
)
set_target_properties(cust_opapi PROPERTIES PREFIX "lib" OUTPUT_NAME "cust_opapi")
set_target_properties(cust_opapi PROPERTIES
  BUILD_RPATH "$ORIGIN/kernel_artifacts"
  INSTALL_RPATH "$ORIGIN/artifacts")
add_dependencies(cust_opapi kernel_artifacts)
target_include_directories(cust_opapi PRIVATE ${GE_INCLUDES})
target_link_directories(cust_opapi PRIVATE
  "${ASCEND_HOME_PATH}/lib64"
  "${ASCEND_HOME_PATH}/aarch64-linux/devlib/linux/aarch64"
)
target_link_libraries(cust_opapi PRIVATE ascendcl acl_rt register lowering dl
  kernel_fp16_artifact aicpu_metadata_artifact)

# --- OPP install directly into the CANN directory ---
# A CANN upgrade wipes this directory -- re-run install.sh afterwards.
set(OPP_ROOT "${ASCEND_HOME_PATH}/opp")
set(OPP_LIB "${OPP_ROOT}/op_graph/lib/linux/aarch64")
install(TARGETS cust_opapi DESTINATION "${OPP_LIB}")
install(FILES ${KERNEL_SOS} DESTINATION "${OPP_LIB}/artifacts")

