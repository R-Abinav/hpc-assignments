#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1000000

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = N / size;
    double *a = NULL, *b = NULL, *c = NULL;
    double *local_a = (double*)malloc(local_n * sizeof(double));
    double *local_b = (double*)malloc(local_n * sizeof(double));
    double *local_c = (double*)malloc(local_n * sizeof(double));

    if(rank == 0)
    {
        a = (double*)malloc(N * sizeof(double));
        b = (double*)malloc(N * sizeof(double));
        c = (double*)malloc(N * sizeof(double));
        for(int i = 0; i < N; i++)
        {
            a[i] = (double)i * 1.5;
            b[i] = (double)i * 2.5;
        }
    }

    double start = MPI_Wtime();

    MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, local_n, MPI_DOUBLE, local_b, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for(int i = 0; i < local_n; i++)
        local_c[i] = local_a[i] + local_b[i];

    MPI_Gather(local_c, local_n, MPI_DOUBLE, c, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if(rank == 0)
    {
        printf("vector addition\n");
        printf("n = %d\n", N);
        printf("processes = %d\n", size);
        printf("time: %f seconds\n", end - start);
        printf("sample: c[0] = %.2f, c[1] = %.2f, c[N-1] = %.2f\n", c[0], c[1], c[N-1]);
        free(a); free(b); free(c);
    }

    free(local_a); free(local_b); free(local_c);
    MPI_Finalize();
    return 0;
}