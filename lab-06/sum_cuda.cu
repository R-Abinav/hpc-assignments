#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//size of array - 10 million double precision floating point numbers
#define N 10000000

//parallel reduction kernel - each block reduces its chunk, result stored in block_sums
__global__ void sum_reduce(double *input, double *block_sums, int n)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    //load element into shared memory, 0 if out of bounds
    sdata[tid] = (id < n) ? input[id] : 0.0;
    __syncthreads();

    //reduction in shared memory
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    //write block result
    if(tid == 0)
    {
        block_sums[blockIdx.x] = sdata[0];
    }
}

//serial sum of n double precision values
double serial_sum(double *a, int n)
{
    double sum = 0.0;
    for(int i = 0; i < n; i++)
    {
        sum += a[i];
    }
    return sum;
}

//main program
int main()
{
    //number of bytes to allocate for n doubles
    size_t bytes = N * sizeof(double);

    //allocate memory for array on host
    double *A = (double*)malloc(bytes);

    //fill host array with large double precision values
    srand(42);
    for(int i = 0; i < N; i++)
    {
        A[i] = ((double)rand() / RAND_MAX) * 1000000.0;
    }

    printf("\nsum of n double precision floating point numbers\n");
    printf("n = %d elements\n", N);
    printf("values range: 0.0 to 1000000.0\n");

    //serial execution timing
    clock_t start_serial = clock();
    double serial_result = serial_sum(A, N);
    clock_t end_serial = clock();
    double serial_time = ((double)(end_serial - start_serial)) / CLOCKS_PER_SEC;

    printf("\nserial execution completed\n");
    printf("serial sum = %.6f\n", serial_result);
    printf("serial time: %f seconds\n", serial_time);

    //allocate device memory for input array
    double *d_A;
    cudaMalloc(&d_A, bytes);

    //copy input data to device
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);

    //configuration 1: threads = 256
    printf("\n");
    int threads_1 = 256;
    int blocks_1 = (int)ceil((float)N / threads_1);
    printf("configuration 1: blocks = %d, threads = %d\n", blocks_1, threads_1);
    printf("\n");

    //allocate device memory for block sums
    double *d_block_sums_1;
    cudaMalloc(&d_block_sums_1, blocks_1 * sizeof(double));
    double *h_block_sums_1 = (double*)malloc(blocks_1 * sizeof(double));

    cudaEvent_t start_1, stop_1;
    cudaEventCreate(&start_1);
    cudaEventCreate(&stop_1);
    cudaEventRecord(start_1);

    //launch reduction kernel
    sum_reduce<<<blocks_1, threads_1, threads_1 * sizeof(double)>>>(d_A, d_block_sums_1, N);

    cudaEventRecord(stop_1);
    cudaEventSynchronize(stop_1);
    float parallel_time_1 = 0;
    cudaEventElapsedTime(&parallel_time_1, start_1, stop_1);
    parallel_time_1 = parallel_time_1 / 1000.0;

    //copy block sums back and finish reduction on cpu
    cudaMemcpy(h_block_sums_1, d_block_sums_1, blocks_1 * sizeof(double), cudaMemcpyDeviceToHost);
    double parallel_result_1 = 0.0;
    for(int i = 0; i < blocks_1; i++)
    {
        parallel_result_1 += h_block_sums_1[i];
    }

    //verify results
    double tolerance = 1.0;
    double diff_1 = fabs(parallel_result_1 - serial_result);
    if(diff_1 < tolerance)
    {
        printf("verification: success (diff = %.6f)\n", diff_1);
    }
    else
    {
        printf("verification: failed (diff = %.6f)\n", diff_1);
    }
    printf("parallel sum = %.6f\n", parallel_result_1);
    printf("parallel time: %f seconds\n", parallel_time_1);
    printf("speedup: %.2fx\n", serial_time / parallel_time_1);

    //configuration 2: threads = 512
    printf("\n");
    int threads_2 = 512;
    int blocks_2 = (int)ceil((float)N / threads_2);
    printf("configuration 2: blocks = %d, threads = %d\n", blocks_2, threads_2);
    printf("\n");

    double *d_block_sums_2;
    cudaMalloc(&d_block_sums_2, blocks_2 * sizeof(double));
    double *h_block_sums_2 = (double*)malloc(blocks_2 * sizeof(double));

    cudaEvent_t start_2, stop_2;
    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);
    cudaEventRecord(start_2);

    sum_reduce<<<blocks_2, threads_2, threads_2 * sizeof(double)>>>(d_A, d_block_sums_2, N);

    cudaEventRecord(stop_2);
    cudaEventSynchronize(stop_2);
    float parallel_time_2 = 0;
    cudaEventElapsedTime(&parallel_time_2, start_2, stop_2);
    parallel_time_2 = parallel_time_2 / 1000.0;

    cudaMemcpy(h_block_sums_2, d_block_sums_2, blocks_2 * sizeof(double), cudaMemcpyDeviceToHost);
    double parallel_result_2 = 0.0;
    for(int i = 0; i < blocks_2; i++)
    {
        parallel_result_2 += h_block_sums_2[i];
    }

    double diff_2 = fabs(parallel_result_2 - serial_result);
    if(diff_2 < tolerance)
    {
        printf("verification: success (diff = %.6f)\n", diff_2);
    }
    else
    {
        printf("verification: failed (diff = %.6f)\n", diff_2);
    }
    printf("parallel sum = %.6f\n", parallel_result_2);
    printf("parallel time: %f seconds\n", parallel_time_2);
    printf("speedup: %.2fx\n", serial_time / parallel_time_2);

    //configuration 3: threads = 1024
    printf("\n");
    int threads_3 = 1024;
    int blocks_3 = (int)ceil((float)N / threads_3);
    printf("configuration 3: blocks = %d, threads = %d\n", blocks_3, threads_3);
    printf("\n");

    double *d_block_sums_3;
    cudaMalloc(&d_block_sums_3, blocks_3 * sizeof(double));
    double *h_block_sums_3 = (double*)malloc(blocks_3 * sizeof(double));

    cudaEvent_t start_3, stop_3;
    cudaEventCreate(&start_3);
    cudaEventCreate(&stop_3);
    cudaEventRecord(start_3);

    sum_reduce<<<blocks_3, threads_3, threads_3 * sizeof(double)>>>(d_A, d_block_sums_3, N);

    cudaEventRecord(stop_3);
    cudaEventSynchronize(stop_3);
    float parallel_time_3 = 0;
    cudaEventElapsedTime(&parallel_time_3, start_3, stop_3);
    parallel_time_3 = parallel_time_3 / 1000.0;

    cudaMemcpy(h_block_sums_3, d_block_sums_3, blocks_3 * sizeof(double), cudaMemcpyDeviceToHost);
    double parallel_result_3 = 0.0;
    for(int i = 0; i < blocks_3; i++)
    {
        parallel_result_3 += h_block_sums_3[i];
    }

    double diff_3 = fabs(parallel_result_3 - serial_result);
    if(diff_3 < tolerance)
    {
        printf("verification: success (diff = %.6f)\n", diff_3);
    }
    else
    {
        printf("verification: failed (diff = %.6f)\n", diff_3);
    }
    printf("parallel sum = %.6f\n", parallel_result_3);
    printf("parallel time: %f seconds\n", parallel_time_3);
    printf("speedup: %.2fx\n", serial_time / parallel_time_3);

    //free cpu memory
    free(A);
    free(h_block_sums_1);
    free(h_block_sums_2);
    free(h_block_sums_3);

    //free gpu memory
    cudaFree(d_A);
    cudaFree(d_block_sums_1);
    cudaFree(d_block_sums_2);
    cudaFree(d_block_sums_3);

    printf("\nlarge double precision floating point addition complete\n");
    printf("n = %d elements processed\n", N);

    return 0;
}