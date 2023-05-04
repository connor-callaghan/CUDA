// Multiply matrices using the BLAS library

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <assert.h>
#include <time.h>

// Verify results on the CPU
void verify_result(float *a, float *b, float *c, int n)
{
    float temp;
    float epsilon = 0.001;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            temp = 0;
            for (int k = 0; k < n; k++)
            {
                // Translate to column major order based on cuBLAS memory layout
                temp += a[k * n + i] * b[j * n + k];
            }
            assert(fabs(c[j * n + i] - temp) < epsilon);
        }
    }
}

int main()
{
    // Define Size of Problem
    int n = 1 << 10;
    size_t bytes = n * n * sizeof(float);

    // Declare matrix pointers on host and device
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // Allocate memory for above pointers
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Pseudo random number generator with a reusable seed, Using curand library
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    // Initialize matrix with random numbers on the device
    curandGenerateUniform(prng, d_a, n*n);
    curandGenerateUniform(prng, d_b, n*n);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scaling Factors
    float alpha = 1.0f;
    float beta = 0.0f;

    // Calculation: C = (alpha*A) * B + (beta*C)
    // (m x n) * (n * k) = (m X k)
    // Signature for cublasSgemm: handle, operation (use CUBLAS_OP_T for transposing), operation, m, n, k, address of alpha, A, lda, B, ldb, address of beta, C, ldc
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

    // Copy back to host
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // verify result
    verify_result(h_a, h_b, h_c, n);

    printf("SUCCESS\n");

    return 0;
}