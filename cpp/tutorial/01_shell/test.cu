#include "cutlass/cutlass.h"
#include "launch_template.h"
#include <iostream>


// nvcc -lineinfo --expt-relaxed-constexpr -I${PWD}/cutlass/include ./tutorial/01_shell/test.cu
int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    run_flash_fwd<0, 0, cutlass::bfloat16_t, cutlass::bfloat16_t, 0, 0>(stream);
    std::cout << "Finished" << std::endl;
}