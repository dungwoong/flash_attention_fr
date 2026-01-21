#pragma once
#include "cute/tensor.hpp"
#include "cute/layout.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/kernel_launch.h"

#include "sm90_pipeline_no_cluster.hpp"
#include "mainloop.h"
#include "kernel.h"
#include "epilogue.h"
#include "tile_scheduler.h"


// I think this is the check cuda and stuff we want
#include "util.hpp"


// kHeadDim is d_model / num_heads for qk, just the embedding dim
// TODO if you put clusters in, only allow ClusterM. cluster can only multicast in the m dimension, and it will multicast k and v
// This will be full self-attention
using namespace cute;
template <int kHeadDim, int kHeadDimV, typename Element, typename ElementOut, int BlockM, int BlockN>
void run_flash_fwd(cudaStream_t stream) {
    // static constexpr bool IntraWGOverlap = false;
    // static constexpr bool MMA_PV_is_RS = true;

    using ClusterShape = cute::Shape<Int<1>, _1, _1>; // TODO add cluster later

    // I'm gonna say that QKt is (MNkHeadDim) and PV is (MkHeadDimVN)
    using TileShape_MNK = cute::Shape<Int<BlockM>, Int<BlockN>, Int<kHeadDim>>;
    using TileShape_MNK_PV = cute::Shape<Int<BlockM>, Int<kHeadDimV>, Int<BlockN>>;

    using AttnKernel = myflash::Kernel<
        kHeadDim, kHeadDimV, 
        Element, float, ElementOut, 
        BlockM, BlockN, 
        TileShape_MNK, TileShape_MNK_PV,
        2, // stages
        ClusterShape
        >;

    // TODO construct mainloop, epilogue, scheduler args

    // TODO add CHECK_CUDA
    // create kernel params by feeding in mainloop args, epilogue args, etc.

    dim3 grid_dims = AttnKernel::get_grid_shape(/*need stuff*/);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;

    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({});

    // CUDA kernels have a default 48kB dynamic SMEM per block
    if constexpr (size(ClusterShape{}) > 1) {

    } else {
        auto kernel = cutlass::device_kernel<AttnKernel>; // only required to set smem
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch with pdl*/);
        CHECK_CUDA_KERNEL_LAUNCH();
    }

}