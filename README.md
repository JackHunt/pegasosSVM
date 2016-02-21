# pegasosSVM
A CPU and GPU based implementation of the Primal Estimated Subgradient Solver for Support Vector Machines.

### About this software.
This software has been written in an ad-hoc manner for my own research purposes. As such, 
it may not be as complete as some other, well established packages. However, if you wish to use this 
software and require additional features, or have a suggestion for improvement, do feel free to 
contact me and I shall see what I can do w.r.t my schedule.

This software is provided under a BSD license.

### What does this implementation provide?
This implementation provides classes for CPU and GPU Primal optimisation 
of binary Support Vector Machine's. The CPU implementation is parallelised with OpenMP, the 
GPU implementation, by CUDA.

It should be noted that the CPU implementation is provided primarily for demonstration purposes, 
and that highly optimised CPU SVM libraries are available.

### What expansions shall be made in the future?
*Non binary classification
*Regression
*GPU Mini Batches(see below)

#### Mini Batches
Mini batch support is built in to the CPU implementation, however it has proven to be 
somewhat more complicated to do this efficiently in the CUDA implementation(due to irregular 
memory accesses). However, this effect can be achieved by passing in subsets of your data to 
the algorithm). However, the data must be contiguous in memory.