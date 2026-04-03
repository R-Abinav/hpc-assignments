#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 16; // Total number of double precision floating point numbers
    double local_sum = 0.0;
    double total_sum = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process gets N/size elements
    int local_n = N / size;
    double *local_data = (double *)malloc(local_n * sizeof(double));

    // Process 0 initializes all data and distributes it
    if (rank == 0) {
        double *data = (double *)malloc(N * sizeof(double));

        // Initialize array: data[i] = i + 1 (1.0, 2.0, ..., N.0)
        printf("Initializing %d double precision numbers:\n", N);
        for (int i = 0; i < N; i++) {
            data[i] = (double)(i + 1);
            printf("data[%d] = %.1f\n", i, data[i]);
        }
        printf("\n");

        // Send local portion to each other process using point-to-point
        for (int dest = 1; dest < size; dest++) {
            MPI_Send(&data[dest * local_n], local_n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }

        // Process 0 keeps its own portion
        for (int i = 0; i < local_n; i++) {
            local_data[i] = data[i];
        }

        free(data);
    } else {
        // Other processes receive their portion from process 0
        MPI_Recv(local_data, local_n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Each process computes its local sum
    for (int i = 0; i < local_n; i++) {
        local_sum += local_data[i];
    }
    printf("Process %d: local_sum = %.6f\n", rank, local_sum);

    // Point-to-point gathering of partial sums to process 0 (NO MPI_Reduce)
    if (rank == 0) {
        total_sum = local_sum;

        // Receive partial sums from all other processes
        for (int src = 1; src < size; src++) {
            double recv_sum = 0.0;
            MPI_Recv(&recv_sum, 1, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += recv_sum;
        }

        // Calculate expected sum using formula: N*(N+1)/2
        double expected = (double)(N * (N + 1)) / 2.0;

        printf("\n=== RESULT ===\n");
        printf("Total number of elements : %d\n", N);
        printf("Number of processes      : %d\n", size);
        printf("Computed Sum             : %.6f\n", total_sum);
        printf("Expected Sum (N*(N+1)/2) : %.6f\n", expected);
    } else {
        // All other processes send their local sum to process 0
        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}