#pragma once

#include "cute/layout.hpp"

namespace myflash {
template<class MainloopPipelineK_, class MainloopPipelineVt_, class PipelineState_>
class CollectiveMainloop {
public:
    using MainloopPipelineK = MainloopPipelineK_;
    using MainloopPipelineVt = MainloopPipelineVt_;
    using PipelineState = PipelineState_;
    // static constexpr int DimM = cute::get<0>(TileShape_MNK{});
    

    // using AtomLayoutQK = Layout<Shape<Int<DimM / 64>, _1, _1>>;
    // using AtomLayoutPV = AtomLayoutQK; // -_-

    struct Params {
        int num_loops;
    };

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    load(
        int const& num_loops, 
        uint32_t const& TmaTransactionBytesQ,
        MainloopPipelineK pipeline_k, 
        MainloopPipelineVt pipeline_v,
        PipelineState& smem_pipe_write,
        SharedStorage& shared_storage
    ) {

        // producer acquire will wait empty arrive and expect tx full
        auto load_K = [&] (int const n_block, auto const& smem_pipe_write) {
            pipeline_k.producer_acquire(smem_pipe_write);
        };

        auto load_V = [&] (int const n_block, auto const& smem_pipe_write) {
            pipeline_v.producer_acquire(smem_pipe_write);
        };

        if (cute::elect_one_sync()) {
            shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
        }

        for (int i = 0; i < num_loops; ++i) {
            load_K(i, smem_pipe_write);
            load_V(i, smem_pipe_write);
            ++smem_pipe_write;
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    load_tail(
        MainloopPipelineK pipeline_k, MainloopPipelineVt pipeline_v, 
        PipelineState& smem_pipe_write, SharedStorage& shared_storage
    ) {
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {
            pipeline_k.producer_tail(smem_pipe_write);
            pipeline_v.producer_tail(smem_pipe_write);
        }

    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    mma(
        int const& num_loops,
        MainloopPipelineK pipeline_k,
        MainloopPipelineVt pipeline_v,
        PipelineState& smem_pipe_read,
        SharedStorage& shared_storage
    ) {
        // wait full 
        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        shared_storage.pipelines.barrier_Q.wait(0); // phase should flip when you move onto next work

        for (int i = 0; i < num_loops; ++i) {
            consumer_wait(pipeline_k, smem_pipe_read); // wait full
            pipeline_k.consumer_release(smem_pipe_read); // arrive empty
            consumer_wait(pipeline_v, smem_pipe_read);
            pipeline_v.consumer_release(smem_pipe_read);
            ++smem_pipe_read;
        }

    }
};
}