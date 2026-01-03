#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <cuda/cmath>
#define TILESIZE 32
#define REGTILE 2

//Understanding the code; matrix mul -> c = a x b 
// A is M x K  (M rows, K columns)A
// B is K x N  (K rows, N columns)
// C is M x N  (M rows, N columns)

__global__ void matmul(float* C, const float* A, const float* B, int M, int N, int K)
{

	int row = blockIdx.y * TILESIZE + (threadIdx.y * REGTILE);
	int col = blockIdx.x * TILESIZE + (threadIdx.x * REGTILE);
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float __shared__ As[TILESIZE][TILESIZE];
	float __shared__ Bs[TILESIZE][TILESIZE];

	float c[REGTILE][REGTILE] = { 0.0f };

	int numTiles = cuda::ceil_div(K, TILESIZE);
#pragma unroll
	for (int tile = 0; tile < numTiles; tile++) {
		int rowA = blockIdx.y * TILESIZE + (ty * REGTILE);
		int colA = tile * TILESIZE + (tx * REGTILE);
		//we need to transpose As, so that we can iterate through the columns (coalesce)
		// instead of iterating through the rows (k elements apart each time)



		// Store transposed: A[row][col] → As[col][row]
		float2 rowA0 = *reinterpret_cast<const float2*>(&A[rowA * K + colA]);
		As[ty * REGTILE + 0][tx * REGTILE + 0] = rowA0.x;  // 00
		As[ty * REGTILE + 0][tx * REGTILE + 1] = rowA0.y;  // 01
		float2 rowA1 = *reinterpret_cast<const float2*>(&A[(rowA + 1) * K + colA]);
		As[ty * REGTILE + 1][tx * REGTILE + 0] = rowA1.x;  // 10
		As[ty * REGTILE + 1][tx * REGTILE + 1] = rowA1.y;  // 11

		int rowB = tile * TILESIZE + (ty * REGTILE);
		int colB = blockIdx.x * TILESIZE + (tx * REGTILE);

		// Load row 0 (2 consecutive columns)
		float2 rowB0 = *reinterpret_cast<const float2*>(&B[rowB * N + colB]);
		Bs[ty * REGTILE + 0][tx * REGTILE + 0] = rowB0.x;  // colB + 0
		Bs[ty * REGTILE + 0][tx * REGTILE + 1] = rowB0.y;  // colB + 1

		// Load row 1 (2 consecutive columns)
		float2 rowB1 = *reinterpret_cast<const float2*>(&B[(rowB + 1) * N + colB]);
		Bs[ty * REGTILE + 1][tx * REGTILE + 0] = rowB1.x;  // colB + 0
		Bs[ty * REGTILE + 1][tx * REGTILE + 1] = rowB1.y;  // colB + 1



		__syncthreads();
#pragma unroll
		for (int k = 0; k < TILESIZE; k++) {
			// Load 4 consecutive values from As and Bs into registers
			float a0 = As[ty * REGTILE + 0][k];
			float a1 = As[ty * REGTILE + 1][k];


			float b0 = Bs[k][tx * REGTILE + 0];
			float b1 = Bs[k][tx * REGTILE + 1];


			// 16 independent FMAs (outer product)
			c[0][0] += a0 * b0;
			c[0][1] += a0 * b1;
			c[1][0] += a1 * b0;
			c[1][1] += a1 * b1;


		}
		__syncthreads();
	}
	C[row * N + col] = c[0][0];
	C[row * N + col + 1] = c[0][1];
	C[(row + 1) * N + col] = c[1][0];
	C[(row + 1) * N + col + 1] = c[1][1];
}




int main()
{
	const int M = 2048;
	const int N = 2048;
	const int K = 2048;

	float* h_A = new float[M * K];
	float* h_B = new float[K * N];
	float* h_C = new float[M * N];

	for (int i = 0; i < M * K; i++) {
		h_A[i] = 1.0f;
	}
	for (int i = 0; i < K * N; i++) {
		h_B[i] = 2.0f;
	}

	float* d_A;
	float* d_B;
	float* d_C;

	cudaMalloc(&d_A, M * K * sizeof(float));
	cudaMalloc(&d_B, K * N * sizeof(float));
	cudaMalloc(&d_C, M * N * sizeof(float));

	cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

	dim3 block(TILESIZE / REGTILE, TILESIZE / REGTILE); //blocks of 256
	dim3 grid(cuda::ceil_div(M, TILESIZE), cuda::ceil_div(N, TILESIZE)); // C =  M X N 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	matmul << <grid, block >> > (d_C, d_A, d_B, M, N, K); //warm up the kernel
	cudaDeviceSynchronize();

	cudaEventRecord(start);

	matmul << <grid, block >> > (d_C, d_A, d_B, M, N, K);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);



	cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

	float expected = K * 2.0f;
	bool correct = true;

	for (int i = 0; i < M * N; i++) {
		if (fabs(h_C[i] - expected) > 1e-3) {
			printf("wrong try again");
			correct = false;
			break;
		}
	}
	if (!correct) {
		printf("i failed");
	}
	else
	{
		printf("hooray\n");
		double flops = 2.0 * M * N * K;
		//flops = 2 ops (multiply + add) per element 
		double gflops = (flops / 1e9) / (milliseconds / 1000.0);
		printf("Time is: %.2f ms \n", milliseconds);
		printf("GFLOPS is: %.2f\n", gflops);
	}

	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);



	return 0;
}