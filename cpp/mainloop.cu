#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"

using namespace cute;

template <int Stages, class TileShape_MNK_, int DimV_, class Element_, class ElementAccum_, class ClusterShape_>
struct CollectiveMainloopFwdSm90 {
    static constexpr int kStages = Stages;
    using ClusterShape = ClusterShape_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;

    static constexpr bool V_colmajor = false;
    static constexpr cute::GMMA::Major TmaMajorV = GMMA::Major::MN; // V will be row-major --> N-major

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
    using TileShape_MNK_QV = Shape<decltype(get<0>(TileShape_MNK{})), decltype(get<1>(TileShape_MNK{})), Int<DimV>>;

    // We have to populate tiledMMAQK and PV with atom layouts

    /*
    SMEM layouts

    Swizzle atom -- goes back to some like ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;
        but these are bits so that's like 64x16-bits
    SMEMLayout just tiles to shape
    K and V have stages
    */
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
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
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
    using AtomLayoutQK = Layout<Shape<Int<DimM / 64>, _1, _1>>;
    using TiledMmaQK = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutQK{}
    ));


    // __device__
    // static void prefetch_tma_descriptors() {}

    // __device__
    // void load() {}

    // __device__
    // void load_tail() {}

    // __device__ void mma() {}
};

int main() {
    using cml = CollectiveMainloopFwdSm90<2, Shape<Int<128>, _32, _64>, 64, cutlass::bfloat16_t, float, Shape<_1, _1>>;
    // cute::print(cml::ShapeQKV{16, 16, 1024, 64});
    cute::print(cml::TMA_Q{});
    cute::print(cml::TMA_K{});
    cute::print(cml::TMA_V{});
}