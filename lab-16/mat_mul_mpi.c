#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4

void print_matrix(float mat[N][N], const char *name) {
    printf("%s:\n", name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.2f ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;

    float A[N][N], B[N][N], C[N][N];
    float local_A[N][N], local_C[N][N];

    if (rank == 0) {
        printf("Initializing %dx%d matrices with float values...\n\n", N, N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = (float)(i * N + j + 1);          // 1.0, 2.0, ..., N^2
                B[i][j] = (float)(i == j ? 2.0 : 1.0);    // 2 on diagonal, 1 elsewhere
            }
        }
        print_matrix(A, "Matrix A");
        print_matrix(B, "Matrix B");
    }

    // Scatter rows of A to all processes
    MPI_Scatter(A, rows_per_proc * N, MPI_FLOAT,
                local_A, rows_per_proc * N, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Broadcast entire matrix B to all processes
    MPI_Bcast(B, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Each process computes its rows of C = A x B
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0.0f;
            for (int k = 0; k < N; k++) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }

    printf("Process %d: computed rows %d to %d\n",
           rank, rank * rows_per_proc, (rank + 1) * rows_per_proc - 1);

    // Gather result rows from all processes to process 0
    MPI_Gather(local_C, rows_per_proc * N, MPI_FLOAT,
               C, rows_per_proc * N, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n");
        print_matrix(C, "Result Matrix C = A x B");
    }

    MPI_Finalize();
    return 0;
}