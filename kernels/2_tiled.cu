#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <cuda/cmath>


//Understanding the code; matrix mul -> c = a x b 
// A is M x K  (M rows, K columns)
// B is K x N  (K rows, N columns)
// C is M x N  (M rows, N columns)

__global__ void matmul(float* C, const float* A, const float* B, int M, int N, int K)
{
	const int TILESIZE = 16;

	int row = blockIdx.y * TILESIZE + threadIdx.y;
	int col = blockIdx.x * TILESIZE + threadIdx.x;

	int ty = threadIdx.y;
	int tx = threadIdx.x;

	float sum = 0.0f;
	__shared__ float As[17][16];
	__shared__ float Bs[17][16];
	int numTiles = cuda::ceil_div(K, TILESIZE);

	for (int tile = 0; tile < numTiles; tile++) {
		int rowA = blockIdx.y * TILESIZE + ty;
		int colA = tile * TILESIZE + tx;

		As[ty][tx] = A[rowA * K + colA];

		int rowB = tile * TILESIZE + ty;
		int colB = blockIdx.x * TILESIZE + tx;

		Bs[ty][tx] = B[rowB * N + colB];

		__syncthreads();

		for (int k = 0; k < TILESIZE; k++) {
			float rowPlusOne = As[ty+1][tx];

			sum += As[ty][k] * Bs[k][tx];
		}
		__syncthreads();
	}
	if (row < M && col < N) {
		C[row * N + col] = sum;
	}
}




int main()
{
	const int M = 1024;
	const int N = 1024;
	const int K = 1024;

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

	dim3 block(16, 16); //blocks of 256
	dim3 grid(64, 64); //64x64 blocks of 256 threads 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	matmul << <grid, block >> > (d_C, d_A, d_B, M, N, K); //warm up the kernel
	cudaDeviceSynchronize();

	cudaEventRecord(start);

	matmul<<<grid, block>>>(d_C, d_A, d_B, M, N, K);
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