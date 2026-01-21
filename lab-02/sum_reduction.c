#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<time.h>

//program for sum of n numbers
#define n 100000000 

int main(){
	srand(time(NULL));
	double sum = 0;
	
	int threads[] = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};
	int num_tests = sizeof(threads)/sizeof(threads[0]);

	//serial exec time
	double t_1 = -1;

	double *numbers = (double*)malloc(n * sizeof(double));
	for(int i=0; i<n; i++){
		numbers[i] = (double)rand() * 1000000;
	}

	//small warm up -> without warm up, sometimes my 'f' goes above 1
	printf("Warm up\n");
	omp_set_num_threads(threads[0]);  
	#pragma omp parallel reduction(+:sum)
	{
		#pragma omp for
		for(int i=0; i<n; i++){
			sum += numbers[i];
		}
	}
	printf("Warm up complete\n");

	for(int i=0; i<num_tests; i++){
		sum = 0;

		omp_set_num_threads(threads[i]);
		double s_time = omp_get_wtime();

		#pragma omp parallel reduction(+:sum)
		{
			#pragma omp for
			for(int j=0; j<n; j++){
				sum += numbers[j];
			}
		}

		printf("Sum: %lf\n", sum);
		
		double e_time = omp_get_wtime();
		double exec_time = e_time - s_time;
		printf("%d thread(s) exec time: %lf\n", threads[i], exec_time);

		//calculate speedup
		double speedup = -1.00;
		if(i == 0){
			//serial exec
			t_1 = exec_time;
			speedup = 1.00;
			printf("The speedup (for %d thread(s)) is: %lf\n", threads[i], speedup);
		}else{
			speedup = (double)t_1/exec_time;
			printf("The speedup (for %d thread(s)) is: %lf\n", threads[i], speedup);
		}

		//parallelisation fraction
		double f = (threads[i] * (speedup - 1)) / ((threads[i] - 1) * speedup);
		printf("Parallelisation factor (thread(s) = %d): %lf\n", threads[i], f);

		printf("\n\n");
	}

	free(numbers);
	return 0;
}
