#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_pipeline.h>
#include <stdio.h>
#include <cmath>
#include <cuda/cmath>

#define TILESIZE 64
#define REGTILE_M 8
#define REGTILE_N 4

__global__ void matmul(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int M, int N, int K
) {
    int ty = threadIdx.y;  // 0-7
    int tx = threadIdx.x;  // 0-15

    int row = blockIdx.y * TILESIZE + (ty * REGTILE_M);
    int col = blockIdx.x * TILESIZE + (tx * REGTILE_N);

    __shared__ float As[TILESIZE][TILESIZE];
    __shared__ float Bs[TILESIZE][TILESIZE];

    float c[REGTILE_M][REGTILE_N] = { 0.0f };

    int numTiles = cuda::ceil_div(K, TILESIZE);


    {
        int rowA = blockIdx.y * TILESIZE + (ty * REGTILE_M);
        int colA = (tx * REGTILE_N);

        if (rowA < M && colA + 3 < K) {
#pragma unroll
            for (int i = 0; i < REGTILE_M; i++) {
                float4 rowData = __ldg(reinterpret_cast<const float4*>(&A[(rowA + i) * K + colA]));
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

        int rowB = (ty * REGTILE_M);
        int colB = blockIdx.x * TILESIZE + (tx * REGTILE_N);

        if (rowB < K && colB + 3 < N) {
#pragma unroll
            for (int i = 0; i < REGTILE_M; i++) {
                float4 rowData = __ldg(reinterpret_cast<const float4*>(&B[(rowB + i) * N + colB]));
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
    }

    __syncthreads();


    for (int tile = 0; tile < numTiles; tile++) {
        constexpr int k_split = TILESIZE / 2;

#pragma unroll
        for (int k = 0; k < k_split; k += 16) {
            
            float a0[REGTILE_M], a1[REGTILE_M], a2[REGTILE_M], a3[REGTILE_M];
            float a4[REGTILE_M], a5[REGTILE_M], a6[REGTILE_M], a7[REGTILE_M];
            float a8[REGTILE_M], a9[REGTILE_M], a10[REGTILE_M], a11[REGTILE_M];
            float a12[REGTILE_M], a13[REGTILE_M], a14[REGTILE_M], a15[REGTILE_M];

           
            float b0[REGTILE_N], b1[REGTILE_N], b2[REGTILE_N], b3[REGTILE_N];
            float b4[REGTILE_N], b5[REGTILE_N], b6[REGTILE_N], b7[REGTILE_N];
            float b8[REGTILE_N], b9[REGTILE_N], b10[REGTILE_N], b11[REGTILE_N];
            float b12[REGTILE_N], b13[REGTILE_N], b14[REGTILE_N], b15[REGTILE_N];

            
#pragma unroll
            for (int i = 0; i < REGTILE_M; i++) {
                a0[i] = As[ty * REGTILE_M + i][k + 0];
                a1[i] = As[ty * REGTILE_M + i][k + 1];
                a2[i] = As[ty * REGTILE_M + i][k + 2];
                a3[i] = As[ty * REGTILE_M + i][k + 3];
                a4[i] = As[ty * REGTILE_M + i][k + 4];
                a5[i] = As[ty * REGTILE_M + i][k + 5];
                a6[i] = As[ty * REGTILE_M + i][k + 6];
                a7[i] = As[ty * REGTILE_M + i][k + 7];
                a8[i] = As[ty * REGTILE_M + i][k + 8];
                a9[i] = As[ty * REGTILE_M + i][k + 9];
                a10[i] = As[ty * REGTILE_M + i][k + 10];
                a11[i] = As[ty * REGTILE_M + i][k + 11];
                a12[i] = As[ty * REGTILE_M + i][k + 12];
                a13[i] = As[ty * REGTILE_M + i][k + 13];
                a14[i] = As[ty * REGTILE_M + i][k + 14];
                a15[i] = As[ty * REGTILE_M + i][k + 15];
            }

#pragma unroll
            for (int j = 0; j < REGTILE_N; j++) {
                b0[j] = Bs[k + 0][tx * REGTILE_N + j];
                b1[j] = Bs[k + 1][tx * REGTILE_N + j];
                b2[j] = Bs[k + 2][tx * REGTILE_N + j];
                b3[j] = Bs[k + 3][tx * REGTILE_N + j];
                b4[j] = Bs[k + 4][tx * REGTILE_N + j];
                b5[j] = Bs[k + 5][tx * REGTILE_N + j];
                b6[j] = Bs[k + 6][tx * REGTILE_N + j];
                b7[j] = Bs[k + 7][tx * REGTILE_N + j];
                b8[j] = Bs[k + 8][tx * REGTILE_N + j];
                b9[j] = Bs[k + 9][tx * REGTILE_N + j];
                b10[j] = Bs[k + 10][tx * REGTILE_N + j];
                b11[j] = Bs[k + 11][tx * REGTILE_N + j];
                b12[j] = Bs[k + 12][tx * REGTILE_N + j];
                b13[j] = Bs[k + 13][tx * REGTILE_N + j];
                b14[j] = Bs[k + 14][tx * REGTILE_N + j];
                b15[j] = Bs[k + 15][tx * REGTILE_N + j];
            }

            // Compute using ONLY registers (ZERO SMEM access during compute!)
            // 16× FMAs provide  ILP for warp scheduler
#pragma unroll
            for (int i = 0; i < REGTILE_M; i++) {
#pragma unroll
                for (int j = 0; j < REGTILE_N; j++) {
                    c[i][j] += a0[i] * b0[j];
                    c[i][j] += a1[i] * b1[j];
                    c[i][j] += a2[i] * b2[j];
                    c[i][j] += a3[i] * b3[j];
                    c[i][j] += a4[i] * b4[j];
                    c[i][j] += a5[i] * b5[j];
                    c[i][j] += a6[i] * b6[j];
                    c[i][j] += a7[i] * b7[j];
                    c[i][j] += a8[i] * b8[j];
                    c[i][j] += a9[i] * b9[j];
                    c[i][j] += a10[i] * b10[j];
                    c[i][j] += a11[i] * b11[j];
                    c[i][j] += a12[i] * b12[j];
                    c[i][j] += a13[i] * b13[j];
                    c[i][j] += a14[i] * b14[j];
                    c[i][j] += a15[i] * b15[j];
                }
            }
        }


        // PHASE 2: Start ASYNC LOAD for tile N+1
        // (happening in the background while we compute!)

        if (tile + 1 < numTiles) {
            int rowA = blockIdx.y * TILESIZE + (ty * REGTILE_M);
            int colA = (tile + 1) * TILESIZE + (tx * REGTILE_N);

            if (rowA < M && colA + 3 < K) {
#pragma unroll
                for (int i = 0; i < REGTILE_M; i++) {
                    __pipeline_memcpy_async(
                        &As[ty * REGTILE_M + i][tx * REGTILE_N],
                        &A[(rowA + i) * K + colA],
                        sizeof(float4)
                    );
                }
            }

            int rowB = (tile + 1) * TILESIZE + (ty * REGTILE_M);
            int colB = blockIdx.x * TILESIZE + (tx * REGTILE_N);

            if (rowB < K && colB + 3 < N) {
#pragma unroll
                for (int i = 0; i < REGTILE_M; i++) {
                    __pipeline_memcpy_async(
                        &Bs[ty * REGTILE_M + i][tx * REGTILE_N],
                        &B[(rowB + i) * N + colB],
                        sizeof(float4)
                    );
                }
            }

            __pipeline_commit();
        }


        // PHASE 3: Compute SECOND HALF (k=32 to 63)
        // ( async load happens in parallel)

#pragma unroll
        for (int k = k_split; k < TILESIZE; k += 16) {
            float a0[REGTILE_M], a1[REGTILE_M], a2[REGTILE_M], a3[REGTILE_M];
            float a4[REGTILE_M], a5[REGTILE_M], a6[REGTILE_M], a7[REGTILE_M];
            float a8[REGTILE_M], a9[REGTILE_M], a10[REGTILE_M], a11[REGTILE_M];
            float a12[REGTILE_M], a13[REGTILE_M], a14[REGTILE_M], a15[REGTILE_M];

            float b0[REGTILE_N], b1[REGTILE_N], b2[REGTILE_N], b3[REGTILE_N];
            float b4[REGTILE_N], b5[REGTILE_N], b6[REGTILE_N], b7[REGTILE_N];
            float b8[REGTILE_N], b9[REGTILE_N], b10[REGTILE_N], b11[REGTILE_N];
            float b12[REGTILE_N], b13[REGTILE_N], b14[REGTILE_N], b15[REGTILE_N];

#pragma unroll
            for (int i = 0; i < REGTILE_M; i++) {
                a0[i] = As[ty * REGTILE_M + i][k + 0];
                a1[i] = As[ty * REGTILE_M + i][k + 1];
                a2[i] = As[ty * REGTILE_M + i][k + 2];
                a3[i] = As[ty * REGTILE_M + i][k + 3];
                a4[i] = As[ty * REGTILE_M + i][k + 4];
                a5[i] = As[ty * REGTILE_M + i][k + 5];
                a6[i] = As[ty * REGTILE_M + i][k + 6];
                a7[i] = As[ty * REGTILE_M + i][k + 7];
                a8[i] = As[ty * REGTILE_M + i][k + 8];
                a9[i] = As[ty * REGTILE_M + i][k + 9];
                a10[i] = As[ty * REGTILE_M + i][k + 10];
                a11[i] = As[ty * REGTILE_M + i][k + 11];
                a12[i] = As[ty * REGTILE_M + i][k + 12];
                a13[i] = As[ty * REGTILE_M + i][k + 13];
                a14[i] = As[ty * REGTILE_M + i][k + 14];
                a15[i] = As[ty * REGTILE_M + i][k + 15];
            }

#pragma unroll
            for (int j = 0; j < REGTILE_N; j++) {
                b0[j] = Bs[k + 0][tx * REGTILE_N + j];
                b1[j] = Bs[k + 1][tx * REGTILE_N + j];
                b2[j] = Bs[k + 2][tx * REGTILE_N + j];
                b3[j] = Bs[k + 3][tx * REGTILE_N + j];
                b4[j] = Bs[k + 4][tx * REGTILE_N + j];
                b5[j] = Bs[k + 5][tx * REGTILE_N + j];
                b6[j] = Bs[k + 6][tx * REGTILE_N + j];
                b7[j] = Bs[k + 7][tx * REGTILE_N + j];
                b8[j] = Bs[k + 8][tx * REGTILE_N + j];
                b9[j] = Bs[k + 9][tx * REGTILE_N + j];
                b10[j] = Bs[k + 10][tx * REGTILE_N + j];
                b11[j] = Bs[k + 11][tx * REGTILE_N + j];
                b12[j] = Bs[k + 12][tx * REGTILE_N + j];
                b13[j] = Bs[k + 13][tx * REGTILE_N + j];
                b14[j] = Bs[k + 14][tx * REGTILE_N + j];
                b15[j] = Bs[k + 15][tx * REGTILE_N + j];
            }

#pragma unroll
            for (int i = 0; i < REGTILE_M; i++) {
#pragma unroll
                for (int j = 0; j < REGTILE_N; j++) {
                    c[i][j] += a0[i] * b0[j];
                    c[i][j] += a1[i] * b1[j];
                    c[i][j] += a2[i] * b2[j];
                    c[i][j] += a3[i] * b3[j];
                    c[i][j] += a4[i] * b4[j];
                    c[i][j] += a5[i] * b5[j];
                    c[i][j] += a6[i] * b6[j];
                    c[i][j] += a7[i] * b7[j];
                    c[i][j] += a8[i] * b8[j];
                    c[i][j] += a9[i] * b9[j];
                    c[i][j] += a10[i] * b10[j];
                    c[i][j] += a11[i] * b11[j];
                    c[i][j] += a12[i] * b12[j];
                    c[i][j] += a13[i] * b13[j];
                    c[i][j] += a14[i] * b14[j];
                    c[i][j] += a15[i] * b15[j];
                }
            }
        }

        // ========================================
        // PHASE 4: Wait for async load to finish
        // ========================================
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    // ============================================
    // WRITE BACK
    // ============================================
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

    dim3 block(TILESIZE / REGTILE_N, TILESIZE / REGTILE_M);
    dim3 grid(cuda::ceil_div(N, TILESIZE), cuda::ceil_div(M, TILESIZE));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < 5; i++) {
        matmul << <grid, block >> > (d_C, d_A, d_B, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        matmul << <grid, block >> > (d_C, d_A, d_B, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= 10.0f;

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