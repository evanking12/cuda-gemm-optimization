#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <cuda/cmath>

#define TILESIZE 64
#define REGTILE_M 8  // Rows per thread (increased from 4)
#define REGTILE_N 4  // Cols per thread (kept at 4)

__global__ void matmul(float* C, const float* A, const float* B, int M, int N, int K) {
    int row = blockIdx.y * TILESIZE + (threadIdx.y * REGTILE_M);
    int col = blockIdx.x * TILESIZE + (threadIdx.x * REGTILE_N);
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float As[TILESIZE][TILESIZE];
    __shared__ float Bs[TILESIZE][TILESIZE];

    float c[REGTILE_M][REGTILE_N] = { 0.0f };

    int numTiles = cuda::ceil_div(K, TILESIZE);

#pragma unroll
    for (int tile = 0; tile < numTiles; tile++) {
        // Load A tile - each thread loads REGTILE_M x REGTILE_N elements
        int rowA = blockIdx.y * TILESIZE + (ty * REGTILE_M);
        int colA = tile * TILESIZE + (tx * REGTILE_N);

        if (rowA < M && colA + 3 < K) {
#pragma unroll
            for (int i = 0; i < REGTILE_M; i++) {
                float4 rowData = *reinterpret_cast<const float4*>(&A[(rowA + i) * K + colA]);
                As[ty * REGTILE_M + i][tx * REGTILE_N + 0] = rowData.x;
                As[ty * REGTILE_M + i][tx * REGTILE_N + 1] = rowData.y;
                As[ty * REGTILE_M + i][tx * REGTILE_N + 2] = rowData.z;
                As[ty * REGTILE_M + i][tx * REGTILE_N + 3] = rowData.w;
            }
        }
        else {
#pragma unroll
            for (int i = 0; i < REGTILE_M; i++)
#pragma unroll
                for (int j = 0; j < REGTILE_N; j++)
                    As[ty * REGTILE_M + i][tx * REGTILE_N + j] = 0.0f;
        }

        // Load B tile - need to load more to cover the 64x64 tile
        // With 8x16 block, we need each thread to load 2 sets of B rows
        int rowB = tile * TILESIZE + (ty * REGTILE_M);
        int colB = blockIdx.x * TILESIZE + (tx * REGTILE_N);

        if (rowB < K && colB + 3 < N) {
#pragma unroll
            for (int i = 0; i < REGTILE_M; i++) {
                float4 rowData = *reinterpret_cast<const float4*>(&B[(rowB + i) * N + colB]);
                Bs[ty * REGTILE_M + i][tx * REGTILE_N + 0] = rowData.x;
                Bs[ty * REGTILE_M + i][tx * REGTILE_N + 1] = rowData.y;
                Bs[ty * REGTILE_M + i][tx * REGTILE_N + 2] = rowData.z;
                Bs[ty * REGTILE_M + i][tx * REGTILE_N + 3] = rowData.w;
            }
        }
        else {
#pragma unroll
            for (int i = 0; i < REGTILE_M; i++)
#pragma unroll
                for (int j = 0; j < REGTILE_N; j++)
                    Bs[ty * REGTILE_M + i][tx * REGTILE_N + j] = 0.0f;
        }

        __syncthreads();

        // K-unroll by 4 for maximum ILP
#pragma unroll
        for (int k = 0; k < TILESIZE; k += 4) {
            float a0[REGTILE_M], a1[REGTILE_M], a2[REGTILE_M], a3[REGTILE_M];
            float b0[REGTILE_N], b1[REGTILE_N], b2[REGTILE_N], b3[REGTILE_N];

            // Load all A and B values for 4 K iterations
#pragma unroll
            for (int i = 0; i < REGTILE_M; i++) {
                a0[i] = As[k + 0][ty * REGTILE_M + i];
                a1[i] = As[k + 1][ty * REGTILE_M + i];
                a2[i] = As[k + 2][ty * REGTILE_M + i];
                a3[i] = As[k + 3][ty * REGTILE_M + i];
            }

#pragma unroll
            for (int j = 0; j < REGTILE_N; j++) {
                b0[j] = Bs[k + 0][tx * REGTILE_N + j];
                b1[j] = Bs[k + 1][tx * REGTILE_N + j];
                b2[j] = Bs[k + 2][tx * REGTILE_N + j];
                b3[j] = Bs[k + 3][tx * REGTILE_N + j];
            }

            // 128 FMAs (32 per K value × 4 K values) - doubled from before!
#pragma unroll
            for (int i = 0; i < REGTILE_M; i++) {
#pragma unroll
                for (int j = 0; j < REGTILE_N; j++) {
                    c[i][j] += a0[i] * b0[j];
                    c[i][j] += a1[i] * b1[j];
                    c[i][j] += a2[i] * b2[j];
                    c[i][j] += a3[i] * b3[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results with float4 vectorization
    if (row < M && col + 3 < N) {
#pragma unroll
        for (int i = 0; i < REGTILE_M; i++) {
            float4 outData = make_float4(c[i][0], c[i][1], c[i][2], c[i][3]);
            *reinterpret_cast<float4*>(&C[(row + i) * N + col]) = outData;
        }
    }
}

int main() {
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];

    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 2.0f;

    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // NEW: Block is now 16 wide x 8 tall = 128 threads
    dim3 block(TILESIZE / REGTILE_N, TILESIZE / REGTILE_M);
    dim3 grid(cuda::ceil_div(N, TILESIZE), cuda::ceil_div(M, TILESIZE));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    matmul << <grid, block >> > (d_C, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matmul << <grid, block >> > (d_C, d_A, d_B, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    float expected = K * 2.0f;
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C[i] - expected) > 1e-3) {
            printf("Mismatch at %d: got %.2f, expected %.2f\n", i, h_C[i], expected);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("✓ Correct\n");
        double flops = 2.0 * M * N * K;
        double gflops = (flops / 1e9) / (milliseconds / 1000.0);
        printf("Time: %.2f ms\n", milliseconds);
        printf("GFLOPS: %.2f\n", gflops);
    }
    else {
        printf("✗ Wrong\n");
    }

    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}