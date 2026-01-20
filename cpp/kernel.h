#pragma once
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"


namespace myflash{

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class Kernel {
public:
    static constexpr int SharedStorageSize = 0;
    using TileScheduler = TileScheduler_;
    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;

    // These must be implemented for cutlass::device_kernel to work
    static constexpr int MaxThreadsPerBlock = 32;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    using ArchTag = cutlass::arch::Sm80; // change later

    static dim3
    get_grid_shape() {
        return TileScheduler::get_grid_shape();
    }

    static dim3
    get_block_shape() {
        return TileScheduler::get_block_shape();
    }

    struct Params {};

    struct Arguments {};

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {};
    }

    // NEED TO MODIFY
    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {
        
    }
};
}