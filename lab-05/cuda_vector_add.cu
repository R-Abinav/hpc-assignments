#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//size of array
#define N 1048576

//kernel
__global__ void add_vectors(double *a, double *b, double *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) c[id] = a[id] + b[id];
}

//serial vector addition
void serial_add(double *a, double *b, double *c, int n)
{
    for(int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

//main program
int main()
{
    //number of bytes to allocate for n doubles
    size_t bytes = N*sizeof(double);
    
    //allocate memory for arrays a, b, and c on host
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);
    double *C_serial = (double*)malloc(bytes);
    
    //allocate memory for arrays d_a, d_b, and d_c on device
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    //fill host arrays a and b
    for(int i=0; i<N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }
    
    //serial execution timing
    clock_t start_serial = clock();
    serial_add(A, B, C_serial, N);
    clock_t end_serial = clock();
    double serial_time = ((double)(end_serial - start_serial)) / CLOCKS_PER_SEC;
    
    printf("\nserial execution completed\n");
    printf("serial time: %f seconds\n", serial_time);
    
    //copy data from host arrays a and b to device arrays d_a and d_b
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);
    
    //configuration 1: block = n and thread = 1
    printf("\n===========================================\n");
    printf("configuration 1: blocks = %d, threads = 1\n", N);
    printf("===========================================\n");
    int thr_per_blk_1 = 1;
    int blk_in_grid_1 = N;
    
    cudaEvent_t start_1, stop_1;
    cudaEventCreate(&start_1);
    cudaEventCreate(&stop_1);
    cudaEventRecord(start_1);
    
    add_vectors<<< blk_in_grid_1, thr_per_blk_1 >>>(d_A, d_B, d_C);
    
    cudaEventRecord(stop_1);
    cudaEventSynchronize(stop_1);
    float parallel_time_1 = 0;
    cudaEventElapsedTime(&parallel_time_1, start_1, stop_1);
    parallel_time_1 = parallel_time_1 / 1000.0;
    
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    //verify results
    double tolerance = 1.0e-14;
    int error_1 = 0;
    for(int i=0; i<N; i++)
    {
        if(fabs(C[i] - 3.0) > tolerance)
        { 
            printf("error: value of c[%d] = %f instead of 3.0\n", i, C[i]);
            error_1 = 1;
            break;
        }
    }
    
    if(!error_1)
    {
        printf("verification: success\n");
    }
    printf("parallel time: %f seconds\n", parallel_time_1);
    printf("speedup: %.2fx\n", serial_time / parallel_time_1);
    
    //configuration 2: blocks = ceil(n/1024) and threads = 1024
    printf("\n===========================================\n");
    printf("configuration 2: blocks = %d, threads = 1024\n", (int)ceil(float(N)/1024));
    printf("===========================================\n");
    int thr_per_blk_2 = 1024;
    int blk_in_grid_2 = ceil(float(N) / thr_per_blk_2);
    
    cudaEvent_t start_2, stop_2;
    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);
    cudaEventRecord(start_2);
    
    add_vectors<<< blk_in_grid_2, thr_per_blk_2 >>>(d_A, d_B, d_C);
    
    cudaEventRecord(stop_2);
    cudaEventSynchronize(stop_2);
    float parallel_time_2 = 0;
    cudaEventElapsedTime(&parallel_time_2, start_2, stop_2);
    parallel_time_2 = parallel_time_2 / 1000.0;
    
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    //verify results
    int error_2 = 0;
    for(int i=0; i<N; i++)
    {
        if(fabs(C[i] - 3.0) > tolerance)
        { 
            printf("error: value of c[%d] = %f instead of 3.0\n", i, C[i]);
            error_2 = 1;
            break;
        }
    }
    
    if(!error_2)
    {
        printf("verification: success\n");
    }
    printf("parallel time: %f seconds\n", parallel_time_2);
    printf("speedup: %.2fx\n", serial_time / parallel_time_2);
    
    //configuration 3: blocks = ceil(n/256) and threads = 256
    printf("\n===========================================\n");
    printf("configuration 3: blocks = %d, threads = 256\n", (int)ceil(float(N)/256));
    printf("===========================================\n");
    int thr_per_blk_3 = 256;
    int blk_in_grid_3 = ceil(float(N) / thr_per_blk_3);
    
    cudaEvent_t start_3, stop_3;
    cudaEventCreate(&start_3);
    cudaEventCreate(&stop_3);
    cudaEventRecord(start_3);
    
    add_vectors<<< blk_in_grid_3, thr_per_blk_3 >>>(d_A, d_B, d_C);
    
    cudaEventRecord(stop_3);
    cudaEventSynchronize(stop_3);
    float parallel_time_3 = 0;
    cudaEventElapsedTime(&parallel_time_3, start_3, stop_3);
    parallel_time_3 = parallel_time_3 / 1000.0;
    
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    //verify results
    int error_3 = 0;
    for(int i=0; i<N; i++)
    {
        if(fabs(C[i] - 3.0) > tolerance)
        { 
            printf("error: value of c[%d] = %f instead of 3.0\n", i, C[i]);
            error_3 = 1;
            break;
        }
    }
    
    if(!error_3)
    {
        printf("verification: success\n");
    }
    printf("parallel time: %f seconds\n", parallel_time_3);
    printf("speedup: %.2fx\n", serial_time / parallel_time_3);
    
    //free cpu memory
    free(A);
    free(B);
    free(C);
    free(C_serial);
    
    //free gpu memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("\n===========================================\n");
    printf("vector addition complete\n");
    printf("n = %d elements processed\n", N);
    printf("===========================================\n\n");
    
    return 0;
}