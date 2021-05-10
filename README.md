# parallel-matrix-multiplication
Optimisation of general matrix-matrix multiplication algorithm (GEMM) using CUDA.

Strategies include blocking/tiling, loop unrolling, matrix transpose and fused multiply-add.

Algorithm 3.1 Blocked GEMM: A is an m×n matrix, B is an n×p, and C is an m×p matrix, tile is a global 
constant: 1 ≤ 𝑡𝑖𝑙𝑒 ≤ max (𝑚, 𝑝, 𝑛). Performs Matrix multiply of A and B resulting in C.

__matmul__(*A, B, C*)
 
     for tr = 0 to m step tile do
 
       for tc = 0 to p step tile do
 
         for r = 0 to tile do
 
           for c = 0 to tile do
 
             Cr+tr,c+tc = 0
 
             for tk = 0 to n step tile do
 
               for k = 0 to tile do
 
                 Cr+tr,c+tc = Cr+tr,c+tc + Ar+tr,k+tk × Bk+tk,c+tc
 
               end for
 
            end for
 
          end for
 
        end for
 
      end for
      
<img src="https://user-images.githubusercontent.com/61798212/117692435-ecccb380-b1b4-11eb-9e98-f38cec0785f1.png" height="320" width="500">

The blue area shows a 
portion of the C matrix which is being computed in parallel within 1 CUDA block. The red areas 
show the portions of the A and B matrices that have been loaded into shared memory for dot 
products. Finally, the grey areas show the portions of the A and B matrices which will be 
loaded into shared memory and subsequently processed within the same C block, 
sequentially.


### Measuring Performance
flops = m * P * (n + n - 1)

flops2gflops = 1 / 1000000000

Gflops/sec = flops / elapsed time * flops2gflops
      
 
# Report

This report focuses on the GEMM algorithm, implemented in parallel using CUDA JIT compilation provided by the Numba API for Python. For the sake of simplicity, each implementation of the algorithm is designed to work exclusively with square matrices that are divisible by the block size and tile size.

Performance evaluation of each different CUDA kernel has been executed on a NVIDIA Tesla T4 hardware which reaches a double precision peak performance of 254.4 GFLOPS1.

###Numpy Matmul

The Numpy matmul function, which is a serial implementation of the matrix multiplication algorithm executing on the CPU, will be considered as a very first benchmark for performance evaluations. The C = A x B matrix resulting from this function also provides a valid output to verify the correctness of the results
of further implementations of the GEMM algorithm. This function sets the benchmark at approximatively 40 GFLOPS.

###Naïve CUDA

Moving to CUDA kernels, after running a number of tests involving block size, I recognised that a shape of (8, 8) threads per block lead to the best performance results. Thus, all the following parallel implementations share the same number and shape of (8, 8) threads per block.

In the naïve CUDA implementation, each thread loads a row of matrix A and a column of matrix B from global 
memory and successively computes the dot product to obtain one element of C. With a performance of more than 70 GFLOPS for large matrix sizes, a significant performance improvement can be observed if compared to Numpy’s matmul.

###Shared Memory - Tiling

By introducing the use of shared memory and tiling, threads belonging to the same block have access to a portion of shared memory, whose shape and dimensions can be set up by the programmer. In this case, I allocated two 2D arrays of shared memory of the same dimension as the thread block, specifying a double precision data type (float64):

           sA = cuda.shared.array(shape=(T,T), dtype=float64)
           sB = cuda.shared.array(shape=(T,T), dtype=float64)

Next, tiles of A and B are iteratively loaded into shared memory, looping over all square submatrices of A and B needed to compute an element of C.
           
           sA[tr, tc] = A[r, tc + i * T]
           sB[tr, tc] = B[tr + i * T, c]

The use of shared memory reduces the loads from global memory, making the algorithm faster than the naïve version, especially for matrices with shape less than 2000x2000, where a gap of about 40 GFLOPS is recorded. This version performs steadily at just under 90 GFLOPS for large matrix sizes.

###Further Implementations
As shown in the table below, the sole unrolling of the innermost loop did not have an improving impact. Similarly, applying the fma function, which combines the multiplying and adding operationsinto a single operation, did not bring any performance improvements in the Python environment. The single most effective improvement to the shared memory implementation is swapping the indexes of sA and sB to take advantage of caching lines.

           sA[tc, tr] = A[r, tc + i * T]
           sB[tc, tr] = B[tr + i * T, c]

Such implementation yields the higher performance so far, reaching 120 GFLOPS. One again, applying fused multiply-add and loop unrolling did not have an beneficial effect overall, but for matrices with sizes between 1024x1024 and 1536x1536 a slight improvement is to be seen.

<img src="https://user-images.githubusercontent.com/61798212/117695529-3bc81800-b1b8-11eb-8057-63c877959e80.png" height="320" width="550">
