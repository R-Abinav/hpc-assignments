#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<time.h>

#define rows 12000
#define cols 12000

int main(){
	printf("====== Start of programme ======\n");
	srand(time(NULL));
	
	int threads[] = {1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 64};
	int size = sizeof(threads)/sizeof(threads[0]);
	
	//the two matrices to be added
	double ** a = malloc(rows * sizeof(double *));
	for(int i=0; i<rows; i++) a[i] = malloc(cols * sizeof(double));
	
	double ** b = malloc(rows * sizeof(double *));
	for(int i=0; i<rows; i++) b[i] = malloc(cols * sizeof(double));
	double t_1 = -1;
	
	//initialise the matrices a and b
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			a[i][j] = (double)rand() * 10000;
			b[i][j] = (double)rand() * 10110;
		}
	}
	
	for(int i=0; i<size; i++){
		//the resultant matrix 
		double ** c = malloc(rows * sizeof(double *));
		for(int a=0; a<rows; a++) c[a] = malloc(cols * sizeof(double));
	
		omp_set_num_threads(threads[i]);
		double s_time = omp_get_wtime();
		
		#pragma omp parallel for 
		for(int j=0; j<rows; j++){
			for(int k=0; k<cols; k++){
				c[j][k] = a[j][k] + b[j][k];
			}
		}
		
		double e_time = omp_get_wtime();
		
		//free matrix c
		for(int a=0; a<rows; a++) free(c[a]);
		free(c);
		
		double exec_time = e_time - s_time;
		printf("Execution time with %d thread (s): %lf\n", threads[i], exec_time);
		
		//speedup
		double speedup = -1.00;
		if(i == 0){
			t_1 = exec_time;
			speedup = (double)t_1/exec_time;
		}else{
			speedup = (double)t_1/exec_time;
		}
		
		printf("The speedup for %d thread (s) is: %lf\n", threads[i], speedup);
		
		//Parallelisation factor
		double f = (threads[i] * (speedup - 1)) / ((threads[i] - 1) * speedup);
		printf("The parallelisation fraction for %d thread (s) is: %lf\n", threads[i], f);	
		printf("\n\n");
	}
	
	for(int i=0; i<rows; i++) free(a[i]);
	free(a);
	for(int i=0; i<rows; i++) free(b[i]);
	free(b);

	printf("====== End of programme ======\n");
	return 0;
}