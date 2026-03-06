#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 10000

// parallel kernel - each thread computes one element of the result matrix
__global__ void matrix_add(double *a, double *b, double *c, int n)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < n && col < n)
    {
        c[row * n + col] = a[row * n + col] + b[row * n + col];
    }
}

// serial matrix addition
void serial_matrix_add(double **a, double **b, double **c, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

int main()
{
    int n = N;
    size_t bytes = n * n * sizeof(double);

    printf("matrix addition of %d x %d matrices\n", n, n);
    printf("element type: double precision\n");
    printf("total elements per matrix: %d\n\n", n * n);

    // allocate host matrices as 2d arrays
    double **a = (double**)malloc(n * sizeof(double *));
    double **b = (double**)malloc(n * sizeof(double *));
    double **c_serial = (double**)malloc(n * sizeof(double *));
    for(int i = 0; i < n; i++)
    {
        a[i]        = (double*)malloc(n * sizeof(double));
        b[i]        = (double*)malloc(n * sizeof(double));
        c_serial[i] = (double*)malloc(n * sizeof(double));
    }

    // fill matrices with random double precision values
    srand(42);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            a[i][j] = ((double)rand() / RAND_MAX) * 1000000.0;
            b[i][j] = ((double)rand() / RAND_MAX) * 1000000.0;
        }
    }

    // serial execution
    clock_t start_serial = clock();
    serial_matrix_add(a, b, c_serial, n);
    clock_t end_serial = clock();
    double serial_time = ((double)(end_serial - start_serial)) / CLOCKS_PER_SEC;

    printf("serial execution completed\n");
    printf("serial time: %f seconds\n\n", serial_time);

    // flatten matrices into 1d arrays for gpu transfer
    double *h_a = (double*)malloc(bytes);
    double *h_b = (double*)malloc(bytes);
    double *h_c = (double*)malloc(bytes);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            h_a[i * n + j] = a[i][j];
            h_b[i * n + j] = b[i][j];
        }
    }

    // allocate device memory
    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // copy input matrices to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // configuration: 16x16 thread blocks
    dim3 threads(16, 16);
    dim3 blocks((int)ceil((float)n / threads.x), (int)ceil((float)n / threads.y));

    printf("configuration: blocks = (%d, %d), threads = (%d, %d)\n",
        blocks.x, blocks.y, threads.x, threads.y);

    cudaEvent_t start_p, stop_p;
    cudaEventCreate(&start_p);
    cudaEventCreate(&stop_p);
    cudaEventRecord(start_p);

    // launch kernel
    matrix_add<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaEventRecord(stop_p);
    cudaEventSynchronize(stop_p);
    float parallel_ms = 0;
    cudaEventElapsedTime(&parallel_ms, start_p, stop_p);
    double parallel_time = parallel_ms / 1000.0;

    // copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // verify against serial result
    double max_diff = 0.0;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            double diff = fabs(h_c[i * n + j] - c_serial[i][j]);
            if(diff > max_diff) max_diff = diff;
        }
    }

    printf("\n");
    if(max_diff < 1e-6)
    {
        printf("verification: success (max diff = %e)\n", max_diff);
    }
    else
    {
        printf("verification: failed (max diff = %e)\n", max_diff);
    }

    printf("parallel time: %f seconds\n", parallel_time);
    printf("speedup: %.2fx\n", serial_time / parallel_time);

    // free host memory
    for(int i = 0; i < n; i++)
    {
        free(a[i]);
        free(b[i]);
        free(c_serial[i]);
    }
    free(a);
    free(b);
    free(c_serial);
    free(h_a);
    free(h_b);
    free(h_c);

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("\nmatrix addition complete\n");
    printf("n = %d x %d elements processed\n", n, n);

    return 0;
}