#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "named_barrier.hpp"

using namespace cute;

template <int Stages, class TileShape_MNK_, int DimV_, class Element_, class ElementAccum_, class ClusterShape_>
struct CollectiveMainloopFwdSm90 {
    static constexpr int kStages = Stages;
    using ClusterShape = ClusterShape_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;

    static_assert(cute::size(ClusterShape{}) == 1, "Clusters Not Supported Yert");

    /*
    MNK splits for Attention
    Q(M, K) @ K.T(K, N) --> P(M, N) @ V(N, DimV) --> (M, DimV) (DimV was formerly kHeadDimV)

    just think mnk and then mvn
    
    They use weird shapes, but I'll just ignore that and simplify
    */
    using TileShape_MNK = TileShape_MNK_;
    static constexpr int DimM = cute::get<0>(TileShape_MNK{});
    static constexpr int DimN = cute::get<1>(TileShape_MNK{});
    static constexpr int DimK = cute::get<2>(TileShape_MNK{});
    static constexpr int DimV = DimV_;
    using ShapeM = decltype(cute::get<0>(TileShape_MNK{}));
    using ShapeN = decltype(cute::get<1>(TileShape_MNK{}));
    using ShapeK = decltype(cute::get<2>(TileShape_MNK{}));
    using TileShape_MNK_PV = Shape<decltype(get<0>(TileShape_MNK{})), Int<DimV>, decltype(get<1>(TileShape_MNK{}))>;
    // using TileShape_MNK_QV = Shape<decltype(get<0>(TileShape_MNK{})), decltype(get<1>(TileShape_MNK{})), Int<DimV>>;

    // We have to populate tiledMMAQK and PV with atom layouts

    /*
    SMEM layouts

    Swizzle atom -- goes back to some like ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;
        but these are bits so that's like 64x16-bits
    SMEMLayout just tiles to shape
    K and V have stages
    */
    // hardcode v's major-ness. V will be row-major --> N-major
    static constexpr bool V_colmajor = false;
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
    // TODO not sure why they declare the shapes/strides like this, maybe just cuz we don't know dims till runtime
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
    MMA
    - Assume Q not in registers since reg bandwidth is a bottleneck
    */
    static constexpr bool Mma_PV_is_RS = true; // TODO add to template if needed
    static constexpr cute::GMMA::Major MmaMajorV = !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;

    using AtomLayoutQK = Layout<Shape<Int<DimM / 64>, _1, _1>>;
    using TiledMmaQK = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutQK{}
    ));

    // we can handle largeheaddimv later if necessary
    using AtomLayoutPV = AtomLayoutQK;
    using TiledMmaPV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !Mma_PV_is_RS,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>())
        >{},
        AtomLayoutPV{}
    ));

    static constexpr int NumMmaThreadsQK = size(TiledMmaQK{}); // size gives us nthreads I guess
    static constexpr int NumMmaThreads = size(TiledMmaPV{});
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp; // if we aren't using TMA for anything, we want a whole warpgroup. Only TMA --> 1 warp
    static_assert(NumMmaThreadsQK % cutlass::NumThreadsPerWarpGroup == 0);
    static_assert(NumMmaThreads % cutlass::NumThreadsPerWarpGroup == 0);
    static constexpr int NumMmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
    static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

    /*
    SharedStorage
    */
    static constexpr size_t SmemAlignmentQ = 128;
    static constexpr size_t SmemAlignmentK = 128;
    static constexpr size_t SmemAlignmentVtNoTranspose = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});

    // NOTE not sure what the _0 does
    struct TensorStorage : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose), _0> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVtNoTranspose> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
    }

    /*
    Arguments and params
    Whatever you would pass into this mainloop basically
    */
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element* const ptr_K;
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
    };

    struct Params {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q; // NOTE this is where you can instantiate shape and stride
        StrideQK const stride_Q;
        Element* const ptr_K;
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
        TMA_Q tma_load_Q;
    }

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_Q tma_load_Q = make_tma_copy_A_sm90(
            Q_TMA_Copy{},
            mQ,
            SmemLayoutQ{},
            TileShape_MNK{},
            ClusterShape{}
        ); // no mcast for Q, it'll take size<1> cluster shape which has to be 1 I guess
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        // TODO
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    load(
        Params const& params,
        SharedStorage &shared_storage,
        cute::tuple<int32_t, int32_t, int32_t> block_coord // (seqlen_block, head, batch) TODO Idk about split idx
    ) {
        int const m_block = get<0>(block_coord);
        int const bidh = get<1>(block_coord);
        int const bidb = get<2>(block_coord);

        // setup tensors
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        // Tensor sK_pi = as_position_independent_swizzle_tensor(sK);
        
        // Load Q with TMA
        // line 818
        // NOTE this only works because producer is 1 warp. In their code they choose warp0 in the WG

        // TODO why do we have to sync to know that QSMEM is ready?
        // cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty)); // bar.sync id, nthreads
        if (cute::elect_one_sync()) {
            shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(/*TODO*/);
            // params.tma_load_Q
        }

        // load k and v in a for loop, also with TMA
        domain_offset

        

    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    load_tail() {
        // TODO
    }

    void mma() {
        // QKGEMM
        // m_last = m
        // m = rowmax(P)
        // P -= m
        // P = exp(P)
        // m_last = m_last - m
        // e = exp(m_last)
        // l * e
        // l += rowsum(P)
        // cast P
        // output * e
        // PVGEMM
    }



    // __device__
    // static void prefetch_tma_descriptors() {}

    // __device__
    // void load() {}

    // __device__
    // void load_tail() {}

    // __device__ void mma() {}
};

// nvcc -lineinfo --expt-relaxed-constexpr -I${PWD}/cutlass/include mainloop.cu
int main() {
    using cml = CollectiveMainloopFwdSm90<2, Shape<Int<128>, _32, _64>, 64, cutlass::bfloat16_t, float, Shape<_1, _1>>;
    // cute::print(cml::ShapeQKV{16, 16, 1024, 64});
    cute::print(cml::TiledMmaQK{});
    cute::print(cml::TiledMmaPV{});
}