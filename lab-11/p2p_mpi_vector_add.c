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

        //copy rank 0 chunk directly
        for(int i = 0; i < local_n; i++)
        {
            local_a[i] = a[i];
            local_b[i] = b[i];
        }

        //send chunks to all other processes
        for(int p = 1; p < size; p++)
        {
            MPI_Send(a + p * local_n, local_n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            MPI_Send(b + p * local_n, local_n, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(local_a, local_n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_b, local_n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double start = MPI_Wtime();

    for(int i = 0; i < local_n; i++)
        local_c[i] = local_a[i] + local_b[i];

    double end = MPI_Wtime();

    if(rank == 0)
    {
        //copy rank 0 result
        for(int i = 0; i < local_n; i++)
            c[i] = local_c[i];

        //receive results from all other processes
        for(int p = 1; p < size; p++)
            MPI_Recv(c + p * local_n, local_n, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else
    {
        MPI_Send(local_c, local_n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    if(rank == 0)
    {
        printf("vector addition - point to point\n");
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