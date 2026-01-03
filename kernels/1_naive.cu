#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <cstdio>
#define TILE_SIZE 32

//Understanding the code; matrix mul -> c = a x b 
// A is M x K  (M rows, K columns)
// B is K x N  (K rows, N columns)
// C is M x N  (M rows, N columns)


//For each element C[i][j]:
//  k is changing here; on A k changes across columns and on B k changes across rows.
//  (indexed A[column][row])
// C[i][j] = sum of A[i][k] * B[k][j] for k = 0 to k - 1 
// 

// C is computed as row,col but index is in threadIdx.x = col and threadIdx.y = row so its (col,row)
// Similarly to threadIdx is blockIdx.x is col and blockIdx.y is row aka columns per block 
// And blockDim.x is threads per block width and blockDim.y is threads per block height aka rows per block
// putting this all together we have how many columns and rows is a block? then we ask, for this block, which group of columns and rows can we correspond to?
// and from that we ask, which exact column and row is our thread right now? 
// row major vs column major will only affect memory indexing and not the math or thread mapping.
__global__ void matmul_naive(float* C, const float* A, const float* B, int M, int N, int K)
{

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (row < M && col < N) {
		float sum = 0.0f;


		for (int k = 0; k < K; k++) {
			sum += A[row * K + k] * B[k * N + col];

		}

		C[row * N + col] = sum;
	}

}


__global__ void matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K)
{
	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.y * TILE_SIZE + ty;
	int col = blockIdx.x * TILE_SIZE + tx;

	float sum = 0.0f;

	int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

	for (int tile = 0; tile < numTiles; tile++)
	{
		// Load element from A into shared memory
		int aCol = tile * TILE_SIZE + tx;
		int aRow = row;

		if (aRow < M && aCol < K) {
			As[ty][tx] = A[aRow * K + aCol];
		}
		else
		{
			As[ty][tx] = 0.0f;
		}

		// Load element from B into shared memory
		int bRow = tile * TILE_SIZE + ty;
		int bCol = col;

		if (bRow < K && bCol < N) {
			Bs[ty][tx] = B[bRow * N + bCol];
		}
		else
		{
			Bs[ty][tx] = 0.0f;
		}

		// Ensure threads have loaded their data
		__syncthreads();

		// Compute partial dot product
		for (int k = 0; k < TILE_SIZE; k++)
		{
			sum += As[ty][k] * Bs[k][tx];
		}

		__syncthreads();
	}

	if (row < M && col < N)
	{
		C[row * N + col] = sum;
	}
}


int main()
{
	const int M = 1024; // rows in A and C 
	const int K = 1024; // Cols in A, Rows in B 
	const int N = 1024; // Cols in B and C 

	float* h_A = new float[M * K];
	float* h_B = new float[K * N];
	float* h_C = new float[M * N];

	printf("Initializing matrices...\n");
	for (int i = 0; i < M * K; i++) {
		h_A[i] = 1.0f;
	}
	for (int i = 0; i < K * N; i++) {
		h_B[i] = 2.0f;
	}

	float* d_A, * d_B, * d_C;
	cudaMalloc(&d_A, M * K * sizeof(float));
	cudaMalloc(&d_B, K * N * sizeof(float));
	cudaMalloc(&d_C, M * N * sizeof(float));

	cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 blocks_tiled((N + TILE_SIZE - 1) / TILE_SIZE,
		(M + TILE_SIZE - 1) / TILE_SIZE);

	printf("Kernel launched with grid(%d,%d) and blocks(%d,%d)\n",
		blocks_tiled.x, blocks_tiled.y, threads.x, threads.y);

	// Timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	matmul_tiled << <blocks_tiled, threads >> > (d_C, d_A, d_B, M, N, K);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

	printf("\nVerifying results...\n");
	float expected = K * 1.0f * 2.0f;
	bool correct = true;

	for (int i = 0; i < M * N; i++) {
		if (fabs(h_C[i] - expected) > 1e-5) {
			printf("ERROR at index %d: got %f, expected %f\n", i, h_C[i], expected);
			correct = false;
			break;
		}
	}

	if (correct) {
		printf("✓ PASS: All elements are correct!\n");
		printf("Kernel time: %.3f ms\n", milliseconds);

		// Calculate GFLOPS
		double gflops = (2.0 * M * N * K) / 1e9;
		printf("Performance: %.2f GFLOPS\n", gflops / (milliseconds / 1000.0));
	}
	else {
		printf("✗ FAIL: Results incorrect\n");
	}

	// Cleanup
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}