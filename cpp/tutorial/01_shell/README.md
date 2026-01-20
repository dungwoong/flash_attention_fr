# Shell
- Entrypoint is `launch_template.h`, we want to use `cutlass::kernel_launch` and `cutlass::device_kernel`
- We CHOOSE to create a kernel that is parameterized by `CollectiveMainloop, CollectiveEpilogue, Scheduler`
    - How we do this doesn't matter, it's up for use to design
    - For now I just put in template args that I'd expect each one should use
- The kernel class just has to have MaxThreadsPerBlock, MinBlocksPerMultiprocessor and ArchTag

- Afterwards, the way we partition stuff out to different classes is up to us.