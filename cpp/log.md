## Shell
- Shell is quite easy, we can launch an empty kernel quickly
- Next step is a bit complicated, we have to actually setup the mainloop and then grab stuff like nthreads, etc. from it to add to the template

## Barriers
Let's just try to do no persistence, get the mainloop load store barrier sync'd. We can do 0 bytes expected for now
- no intra wg overlap for now either

#### 0120
- I changed the entire shell so the kernel has everything, and then IT creates the mainloop, the tile scheduler, epilogue etc.
- So now, let's focus on the mainloop with no epilogue and simple tile scheduler. Make sure everything necessary is given to the mainloop and get it to execute simple barrier arrives tomorrow.
- The apptainer already has NVCC, we should be good to go

#### 0121
- So basically I think the reason they set things up the way they do is so they can precompute some stuff on the host when the build params, you don't want that param building to be inside the kernel. So what if we just build all the params for the `Kernel` object, then pass relevant stuff to the mainloop etc. We can even build the sub-params in the kernel params yknow.
- Pipelines in cutlass have `is_leader` built in
- I'm not sure how to setup the consumer count for pipelines, I think it's actually 1 per warpgroup but idk
- why do people do `using ClusterShape = ClusterShape_`, it's just so you can access it from outside like `Mainloop::ClusterShape`. I guess I'll just copy them but I think everything will be in kernel anyways haha