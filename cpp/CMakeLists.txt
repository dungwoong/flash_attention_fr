cmake_minimum_required(VERSION 3.22.1)

project(FA3 LANGUAGES CXX CUDA)

# compile with -DCC=90

# CUDA ###########################################
# fetch library
find_package(CUDA REQUIRED)

# make sure it's available
include(CheckLanguage)
check_language(CUDA)

# This makes PTX error
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G") # debug
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -lineinfo \
    -Xptxas=--verbose \
    -Xptxas=--warn-on-spills \
    --use_fast_math
")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
#     -DKITTENS_4090 \
#     -lineinfo \
#     -std=c++20 \
#     -DNDEBUG \
#     -Xcompiler=-Wno-psabi \
#     -Xcompiler=-fno-strict-aliasing \
#     --expt-extended-lambda \
#     --expt-relaxed-constexpr \
#     -forward-unknown-to-host-compiler \
#     --use_fast_math \
#     -Xnvlink=--verbose \
#     -Xptxas=--verbose \
#     -Xptxas=--warn-on-spills \
# ")

# Set some variables
set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY ${CC})
set(CMAKE_CUDA_ARCHITECTURES ${CC})

# CUDA_INCLUDE_DIRS is set by find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/cutlass/include)

# add_executable(fa4090TK fa2_4090_ref.cu)
# set_target_properties(fa4090TK PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
# target_link_libraries(fa4090TK ${CUDA_LIBRARIES} cuda)

# I can just add FA from tri dao to compare, I don't reallly need cudnn hopefully