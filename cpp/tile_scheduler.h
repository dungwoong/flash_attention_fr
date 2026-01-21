#pragma once

#include "cute/layout.hpp"

namespace myflash {

template <int BlockM>
class SingleTileScheduler {
public:
    static dim3 get_grid_shape() {
        return dim3(1, 1, 1);
    }
};
}