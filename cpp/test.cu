#include "cutlass/cutlass.h"
#include "launch_template.h"
#include <iostream>

int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    run_flash_fwd<0, 0, cutlass::bfloat16_t, cutlass::bfloat16_t, 0, 0>(stream);
    std::cout << "Finished" << std::endl;
}