add_op_to_compiled_list()
# Needed by the kernel build: add_ops_src_copy copies the dependency's op_kernel
# tree next to this op so ../../causal_conv1d/op_kernel includes still resolve.
# cmake/custom_build.cmake deliberately removes causal_conv1d from the
# add_subdirectory list so its op_host (and thus its CausalConv1d proto/tiling)
# is never built into this vendor package.
set(causal_conv1d_qkv_depends causal_conv1d PARENT_SCOPE)

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        causal_conv1d_qkv_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME CausalConv1dQkv
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE causal_conv1d_qkv ACLNNTYPE aclnn)

    # The shared causal_conv1d tiling definitions (CausalConv1dTilingFunc /
    # TilingParseForCausalConv1d) are compiled through a project-owned shim TU
    # instead of the vendored .cpp directly: the vendored file ends with
    # IMPL_OP_OPTILING(CausalConv1d), which must not be registered here.
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../causal_conv1d_qkv_tiling_shim.cpp)

    # add_modules_sources only globs *_infershape*.cpp, so the upstream
    # *_proto.cpp (IMPL_OP_INFERSHAPE) would otherwise never be compiled.
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/causal_conv1d_qkv_proto.cpp)
endif()
