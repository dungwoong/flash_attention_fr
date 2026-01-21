#include "cutlass/cutlass.h"
#include "launch_template.h"
#include <iostream>

// nvcc -lineinfo -expt-relaxed-constexpr -arch=sm_90 -I${PWD}/cutlass/include test.cu
int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    run_flash_fwd<64, 64, cutlass::bfloat16_t, cutlass::bfloat16_t, 128, 128>(stream);
    std::cout << "Finished" << std::endl;
}