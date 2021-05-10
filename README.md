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
      
 
