#pragma once
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "tile_scheduler.h"


namespace myflash{

using namespace cute;
template <int kHeadDim, int kHeadDimV, 
    typename Element_, typename ElementAccum_, typename ElementOut_, 
    int BlockM, int BlockN,
    class TileShape_MNK_, class TileShape_MNK_PV_, int Stages, class ClusterShape_>
class Kernel {
public:
    static constexpr int kStages = Stages;
    using ClusterShape = ClusterShape_;

    // Element Types
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ElementOut = ElementOut_;

    // Tile Shapes
    using TileShape_MNK = TileShape_MNK_;
    using TileShape_MNK_PV = TileShape_MNK_PV_;
    static constexpr int DimM = cute::get<0>(TileShape_MNK{});
    static constexpr int DimV = kHeadDimV;
    using ShapeM = decltype(cute::get<0>(TileShape_MNK{}));
    using ShapeN = decltype(cute::get<1>(TileShape_MNK{}));
    using ShapeK = decltype(cute::get<2>(TileShape_MNK{}));

    /*
    MMAs
    */
    static constexpr bool Mma_PV_is_RS = true;
    static constexpr bool V_colmajor = false; // V will be row-major --> N-major

    using AtomLayoutQK = Layout<Shape<Int<DimM / 64>, _1, _1>>;
    using AtomLayoutPV = AtomLayoutQK; // FA does a horiz. layout if PV tile is too big

    using TiledMmaQK = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutQK{}
    ));

    static constexpr cute::GMMA::Major MmaMajorV = !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;
    using TiledMmaPV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !Mma_PV_is_RS,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>())
        >{},
        AtomLayoutPV{}
    ));

    static constexpr int NumMmaThreadsQK = size(TiledMmaQK{});
    static constexpr int NumMmaThreads = size(TiledMmaPV{});
    
    static constexpr int NumMmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumLoadWarpGroups = 1;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp; // one producer warp
    static constexpr int NumThreadsPerCTA = (NumLoadWarpGroups + NumMmaWarpGroups) * cutlass::NumThreadsPerWarpGroup;    

    /*
    SMEM Layouts
    */
    static constexpr cute::GMMA::Major TmaMajorV = GMMA::Major::MN;

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, ShapeM, ShapeK>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, make_shape(ShapeM{}, ShapeK{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, ShapeN, ShapeK>());
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{}, 
        make_shape(
            ShapeN{},
            ShapeK{}, 
            Int<kStages>{})));
    
    // You expect stuff
    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<TmaMajorV, Element,
            Int<DimV>, 
            ShapeN>()); // n, k but can be n-major

    using SmemLayoutVt = decltype(tile_to_shape(
        SmemLayoutAtomVt{},
        make_shape(Int<DimV>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
        std::conditional_t<TmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{} // mode order, so it will be n, k in the end(?)
    ));


    /*
    Copies
    - Q: every warp has one tile
    - KV: all warps iterate over all of K and V(one head), so we can multicast them all the same way
    */
    // These shapes get filled at runtime
    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideV = std::conditional_t<!V_colmajor, StrideQK, cute::Stride<_1, int64_t, int64_t, int64_t>>;
    using Q_TMA_Copy = cute::SM90_TMA_LOAD;
    using KV_TMA_Copy = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{}))); // either TMA_LOAD or TMA_LOAD_MULTICAST
    
    using TMA_Q = decltype(make_tma_copy_A_sm90(
        Q_TMA_Copy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}), // this is just a dummy to get the right type
        SmemLayoutQ{},
        TileShape_MNK{},
        ClusterShape{}
    )); // I don't think they need numbers for QKV shape and stride

    using TMA_K = decltype(make_tma_copy_B_sm90(
        KV_TMA_Copy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
        take<0, 2>(SmemLayoutK{}), // slice(0:2) so no stages
        TileShape_MNK{},
        ClusterShape{}
    ));

    using TMA_V = decltype(make_tma_copy(
        KV_TMA_Copy{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, select<1, 0, 2, 3>(StrideV{})), // (d, seqlen, head, batch) to match the below smem layouts
        take<0, 2>(SmemLayoutVt{}),
        select<1, 2>(TileShape_MNK_PV{}),
        size<0>(ClusterShape{})
    ));

    /*
    SharedStorage for Tensors
    */
    static constexpr size_t SmemAlignmentQ = 128;
    static constexpr size_t SmemAlignmentK = 128;
    static constexpr size_t SmemAlignmentVtNoTranspose = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});

    struct MainloopTensorStorage : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose), _0> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVtNoTranspose> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
    };

    struct EpilogueTensorStorage {}; // TODO

    /*
    Pipelines
    */
    // this works because both KV use the same multicasting
    using PipelineTmaAsync = std::conditional_t<CUTE_STATIC_V(size(ClusterShape{})) == 1,
        typename cutlass::PipelineTmaAsyncNoCluster<kStages>, 
        typename cutlass::PipelineTmaAsync<kStages>>;
    using MainloopPipelineK = PipelineTmaAsync;
    using MainloopPipelineVt = PipelineTmaAsync;
    using PipelineState = cutlass::PipelineState<kStages>;

    // Params help you construct each pipeline
    using PipelineParamsK = typename MainloopPipelineK::Params;
    using PipelineParamsVt = typename MainloopPipelineVt::Params;

    /*
    Shared Memory
    */
    //TODO we have to figure out mainloop_smem_padding later
    static constexpr int mainloop_smem_padding = 0;
    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128, _1> {
            union {
                struct {
                    cute::array<uint32_t, mainloop_smem_padding / sizeof(uint32_t)> padding_;
                    struct MainloopTensorStorage mainloop;
                };
                struct EpilogueTensorStorage epilogue; // TODO we want smem_o to line up with smem_v
            };
        } tensors;
        struct PipelineStorage : cute::aligned_struct<16, _1> {
            // TODO
            alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_Q; // barrier that supports transaction count
            alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_O;
            alignas(16) typename MainloopPipelineK::SharedStorage pipeline_k;
            alignas(16) typename MainloopPipelineVt::SharedStorage pipeline_vt;
            // TODO add tile scheduler::sharedstorage
        } pipelines;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);


    /*
    Register Requirement
    */
    static constexpr uint32_t LoadRegisterRequirement = NumMmaWarpGroups == 1 ? 56 : (NumMmaWarpGroups == 2 ? 40 : 32);
    static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 1 ? 256 : (NumMmaWarpGroups == 2 ? 232 : 160);
    // NOTE if you want to print from producer warp
    // static constexpr uint32_t LoadRegisterRequirement = 40;
    // static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 232 : 152;

    /*
    These must be implemented for cutlass::device_kernel to work
    */
    static constexpr int MaxThreadsPerBlock = (NumMmaWarpGroups + NumLoadWarpGroups) * cutlass::NumThreadsPerWarpGroup;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    using ArchTag = cutlass::arch::Sm80; // change later
    static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

    // TODO construct Mainloop, Epilogue, Tile Scheduler here
    // ---------------------------------------------------------------
    using TileScheduler = myflash::SingleTileScheduler<0>;
    using Mainloop = myflash::CollectiveMainloop<MainloopPipelineK, MainloopPipelineVt, PipelineState>;

    static dim3
    get_grid_shape() {
        return TileScheduler::get_grid_shape();
    }

    static dim3
    get_block_shape() {
        return dim3(NumThreadsPerCTA, 1, 1);
    }

    struct Params {};

    struct Arguments {};

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {};
    }

    static CUTLASS_DEVICE
    PipelineParamsK get_pipeline_params_k(const int& warp_group_idx, const int& warp_group_thread_idx, const uint32_t& transaction_bytes) {
        PipelineParamsK pipeline_params_k;
        pipeline_params_k.role = warp_group_idx == 0
        ? MainloopPipelineK::ThreadCategory::Producer
        : MainloopPipelineK::ThreadCategory::Consumer;
        pipeline_params_k.transaction_bytes = transaction_bytes;
        pipeline_params_k.is_leader = warp_group_thread_idx == 0;
        pipeline_params_k.num_consumers = NumMmaThreads; // In sm90_pipeline.hpp they do a ceil div, so 1 thread per wg arrives
        // I think they always assume there's just 1 consumer
        return pipeline_params_k;
    }

    template <class C>
    static CUTLASS_DEVICE
    void println(C s) {
        if (threadIdx.x == 0) {
            cute::print(s);
        }
    }

    // NEED TO MODIFY
    /*
    Remember
    - Load warpgroup comes first, MMA after with a thread offset
    */
    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {
        println("Entered Kernel\n");
        static constexpr int MmaThreadOffset = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
        int warp_group_idx = cutlass::canonical_warp_group_idx();
        /*
        Pipeline Params
        */
        PipelineParamsK pipeline_params_k = get_pipeline_params_k(warp_group_idx, warp_group_thread_idx, 0);
        static_assert(is_same_v<PipelineParamsK, PipelineParamsVt>);
        PipelineParamsVt pipeline_params_vt = pipeline_params_k;
        pipeline_params_vt.transaction_bytes = 0;

        /*
        Initialize helper classes
        */
        Mainloop mainloop;

        // This is given in cuteDSL
        if constexpr (size(ClusterShape{}) > 1) {
            cute::cluster_arrive_relaxed();
            cute::cluster_wait();
        } else {
            __syncthreads();
        }


        int const lane_predicate = cute::elect_one_sync();
        int const warp_idx = cutlass::canonical_warp_idx_sync();

        if (warp_idx == 0 && lane_predicate) {
            // prefetch TMA
        }

        // You must init certain pipelines manually
        if (warp_idx == 0 && lane_predicate) {
            shared_storage.pipelines.barrier_Q.init(1);
        }

        /*
        Pipelines
        */
        MainloopPipelineK pipeline_k = MainloopPipelineK(shared_storage.pipelines.pipeline_k, pipeline_params_k, ClusterShape{});
        MainloopPipelineVt pipeline_vt = MainloopPipelineVt(shared_storage.pipelines.pipeline_vt, pipeline_params_vt, ClusterShape{});

        const int n_loops = 4; // stand-in for now

        println("Reached branching\n");
        if (warp_group_idx == 0) {
            // reg dealloc
            println("Entered Producer\n");
            PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipelineK>();
            int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
            static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
            if constexpr (SingleProducerWarp) {
                if (warp_idx_in_warpgroup != 0) {return;}
            }
            
            // cutlass::arch::wait_on_dependent_grids(); // this is for PDL
            {
                // we'd run the scheduler but yeah
                mainloop.load(n_loops, 0, pipeline_k, pipeline_vt, smem_pipe_write, shared_storage); // TODO
                mainloop.load_tail(pipeline_k, pipeline_vt, smem_pipe_write, shared_storage);
            }
        } else { // consumer
            // reg alloc
            // initialize mma or whatever
            {
                PipelineState smem_pipe_read; // no need to make start state(producer gets flipped initial producer phase)
                mainloop.mma(n_loops, pipeline_k, pipeline_vt, smem_pipe_read, shared_storage);
            }
            // epilogue
        }


    }
};
} // namespace flash