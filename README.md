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
      
![Screenshot_2021-05-10 6G7Z1003_HPC_Task pdf](https://user-images.githubusercontent.com/61798212/117692435-ecccb380-b1b4-11eb-9e98-f38cec0785f1.png)

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
      
 
