#pragma once

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnFwdSm90 {
    struct SharedStorage {};
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // struct arguments?
    // struct params?
    // to_underlying_arguments converts args to params
    // get grid shape, get block shape

    // operator() is like the shell that you'll see when you run a gemm
}