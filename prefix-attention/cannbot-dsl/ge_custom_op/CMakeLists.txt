# GE custom op (custom_op_v2) for the cannbotdsl prefix-in-Q attention.
#
# Targets:
#   cust_opapi                 runtime deliverable -> opp/op_graph/lib/<os>/<arch>/libcust_opapi.so
#   prefix_attention_op_proto    REG_OP-only proto lib, consumed by add_es_library
#   es_prefix_attn             generated ES python wrapper (tests/run_ge_session.py)
#
# Configure:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j && cmake --install build
cmake_minimum_required(VERSION 3.16)
project(prefix_attention_ge LANGUAGES CXX)

if(NOT ASCEND_HOME_PATH)
  set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "ASCEND_HOME_PATH is empty. Source CANN set_env.sh first.")
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

# --- runtime deliverable: proto + eager execute + DSL artifact launcher ---
add_library(cust_opapi SHARED
  src/prefix_attention_proto.cc
  src/prefix_attention_op.cpp
  src/dsl_launcher.cpp
)
set_target_properties(cust_opapi PROPERTIES PREFIX "lib" OUTPUT_NAME "cust_opapi")
target_include_directories(cust_opapi PRIVATE ${GE_INCLUDES})
target_link_directories(cust_opapi PRIVATE
  "${ASCEND_HOME_PATH}/lib64"
  "${ASCEND_HOME_PATH}/aarch64-linux/devlib/linux/aarch64"
)
target_link_libraries(cust_opapi PRIVATE
  ascendcl
  acl_rt
  register
  lowering
  dl
)

# --- REG_OP-only proto library for the ES wrapper generator ---
add_library(prefix_attention_op_proto SHARED
  src/prefix_attention_proto.cc
)
target_compile_definitions(prefix_attention_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(prefix_attention_op_proto PRIVATE -fvisibility=hidden)
target_include_directories(prefix_attention_op_proto PRIVATE
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/external"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(prefix_attention_op_proto PRIVATE
  "${ASCEND_HOME_PATH}/lib64/libopp_registry.so"
  "${ASCEND_HOME_PATH}/lib64/libgraph.so"
  "${ASCEND_HOME_PATH}/lib64/libgraph_base.so"
)

# --- OPP package install (ASCEND_CUSTOM_OPP_PATH points at opp/) ---
set(OPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/opp")
install(TARGETS cust_opapi
        DESTINATION "${OPP_ROOT}/op_graph/lib/linux/aarch64")
install(FILES include/prefix_attention_op.h
        DESTINATION "${OPP_ROOT}/op_graph/include")

# --- ES python wrapper generation (online GE Session tests) ---
list(APPEND CMAKE_MODULE_PATH "${ASCEND_HOME_PATH}/include/ge/cmake")
find_package(GenerateEsPackage REQUIRED)
add_es_library(
  ES_LINKABLE_AND_ALL_TARGET es_prefix_attn
  OPP_PROTO_TARGET prefix_attention_op_proto
  OUTPUT_PATH ${CMAKE_BINARY_DIR}/es_output
)
