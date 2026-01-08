#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define n 1000000

#define r1 1
#define r2 2
#define r3 4
#define r4 6
#define r5 8
#define r6 10
#define r7 12
#define r8 16
#define r9 20
#define r10 24
#define r11 32
#define r12 64

int main(void){
    long double *a;
    long double *b;

    //using heap mem as stack is giving overflow 
    a = (long double*)malloc(n * sizeof(long double));
    b = (long double*)malloc(n * sizeof(long double));
    
    //init of a and b
    for(int i=0; i<n; i++){
        a[i] = ((long double)i * i) * 100000.00L;    
        b[i] = ((long double)i * i * i) * 999999.99L;
    }

    long double *c;
    c = (long double*)malloc(n * sizeof(long double));

    int threads[] = {r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12};
    int num_tests = sizeof(threads)/sizeof(threads[0]);

    for(int i=0; i<num_tests; i++){
        omp_set_num_threads(threads[i]);

        double s_time = omp_get_wtime();
        #pragma omp parallel 
        {
            #pragma omp for
            for(int i=0; i<n; i++)
            {
                c[i] = a[i] + b[i];
            }
        }
        double e_time = omp_get_wtime();

        double exec_time = e_time - s_time;
        printf("%d thread(s) exec time: %f\n", threads[i], exec_time);
    }
    return 0;
}