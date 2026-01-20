- I think I can just do the equivalent of F.sdpa(Q, K, V) which is non-causal attention

# Implementation notes
## Mainloop
- mnk is kBlockM, kBlockN and kHeadDim, and then kHeadDimV is for V
- the producer is just 1 warp, they use a warpgroup if the producer is doing more stuff, and then you'd have to add logic to select warp0
- for pipelines, q they just use a barrier since there's no pipeline. KV they use pipeline
# Strategy notes
## Mainloop
- register bandwidth is actually a bottleneck so you don't want Q in registers
- for that same reason, they have an option to put PV in registers vs not in registers
- but they have ANOTHER option to use registers for only warpgroup 1 so they can get on with it, while warpgroup2 does SMEM(?)

- qv gemm is a thing to save for the backwards pass or something? I'm not sure but I won't look into it for now since I probably don't need
- So you can have a different number of threads for QK gemm vs PV gemm, need to figure out why later
    - but if you have e.g. (128, 64) --> (128, 32) --> (128, 64) the MMA sizes will be the same. I think this applies when we have a large something dim. We can figure it out later.

## SharedStorage
- So the mainloop keeps referencing shared_storage.pipelines.barrier_Q or whatever when getting barriers
- in `flash_fwd_kernel_sm90(94)` you get `struct SharedStorage` which contains TensorStorage and PipelineStorage inside. Note that it's like `struct TensorStorage {} tensors;` so you access it at `tensors`. The pipelinestorage stores stuff like `barrier_Q`
- you then allocate SMEM and `reinterpret_cast` to `SharedStorage`

## position independent tensor
- as_position_independent_swizzle_tensor converts it from byte-addressed to type-addressed when calculating indices so instead of (float*)(x+4) you can just do x+1 or whatever [src](https://github.com/NVIDIA/cutlass/issues/2259)
    - but it requires the smem to be aligned to e.g. 512 bytes

## bar.sync and bar.arrive
- sync is like arrive and wait
- arrive is just arrive
- both instructions take an id and the number of expected threads
- arrive can continue after arriving, sync needs to sync yknow
- they initially have all threads arrive at `QueryEmpty` for mma WGs to tell producers that `smem_q` is ready but why do they need to do that, isn't it ready to begin with?
- I will just skip that.

## Loading with cp.async vs TMA
- if you load with cp.async you don't have `expect_tx`
- instead, you do cp async wait group and then arrive at the barrier. Threadcount makes it so all threads have to arrive(load is done) before the barrier is ready
- with TMA you `arrive_and_expect_tx` at the start and then you do your copy

## Transpose_V
- they transpose V in smem if they're FP8 and V is row-major. I guess for FP8 you just need that?

## IntraWGOverlap
- when you're in the mainloop, you load k for the current block. If IntraWGOverlap, we actually load the previous v block
- otherwise, we load the current v block, and then you have this one tail v block to load
- With no intra warpgroup overlap it will load Kn, Vn, (for) (Kn-1 Vn-1) ... (K0 V0)
- With IWO they will load Kn, (for) (Kn-1 Vn) ... (K0 V1) endfor V0. Check paper for details
