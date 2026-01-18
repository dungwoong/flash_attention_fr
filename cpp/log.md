- I think I can just do the equivalent of F.sdpa(Q, K, V) which is non-causal attention

# Implementation notes
## Mainloop
- mnk is kBlockM, kBlockN and kHeadDim, and then kHeadDimV is for V
# Strategy notes
## Mainloop
- register bandwidth is actually a bottleneck so you don't want Q in registers
- for that same reason, they have an option to put PV in registers vs not in registers
- but they have ANOTHER option to use registers for only warpgroup 1 so they can get on with it, while warpgroup2 does SMEM(?) bro...