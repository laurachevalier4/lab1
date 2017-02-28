/*
1) Calculate x1 through xn using mode of calculation presented in step 1
2) Substitute initial values for each xj when calculating each xi (do NOT use the newly calculated values of previous xi until the next round of calculations)
3) Calculate the percent error using step 2. If the error is greater than the given %e or equal to the previously calculated %e, repeat steps 1 and two (using the newly calculated x1 through xn as initial values)

INPUT: A text file xxxx.txt
- must be able to run as mpirun -n x my_program inputfile.txt
  (where x is the number of processes)

OUTPUT: The value of each unknown
x1
x2
x3 
.
.
.
xn

ONCE COMPLETE: Check the correctness of your code with gsref
- ssh to a crunchy
- type: module load openmpi-x85_64
- compile your program: mpicc -o my_program my_program.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


/*** Skeleton for Lab 1 ***/

/***** Globals ******/
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */
int comm_sz;
int my_rank;
int nit = 0; /* number of iterations */ // how to keep track of this? Should each process iterate the same number of times? If not, how can we only keep track of the number of iterations of the program that iterates the most?


/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */

/********************************/



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/* 
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix()
{
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;
  
  for(i = 0; i < num; i++)
  {
    sum = 0;
    aii = fabs(a[i][i]);
    
    for(j = 0; j < num; j++)
       if( j != i)
	 sum += fabs(a[i][j]);
       
    if( aii < sum)
    {
      printf("The matrix will not converge.\n");
      exit(1);
    }
    
    if(aii > sum)
      bigger++;
    
  }
  
  if( !bigger )
  {
     printf("The matrix will not converge\n");
     exit(1);
  }
}


/******************************************************/
/* Read input from file */
/* After this function returns:
 * a[][] will be filled with coefficients and you can access them using a[i][j] for element (i,j)
 * x[] will contain the initial values of x
 * b[] will contain the constants (i.e. the right-hand-side of the equations
 * num will have number of variables
 * err will have the absolute error that you need to reach
 */
void get_input(char filename[])
{
  FILE * fp;
  int i,j;  
 
  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

 fscanf(fp,"%d ",&num); // number of unknowns (number of rows in our matrix)
 fscanf(fp,"%f ",&err);

 /* Now, time to allocate the matrices and vectors */
 a = (float**)malloc(num * sizeof(float*)); // puts the data into a two-dimensional array
 if(!a)
  {
	printf("Cannot allocate a!\n");
	exit(1);
  }

 for(i = 0; i < num; i++) 
  {
    a[i] = (float *) malloc(num * sizeof(float)); // a[i] is a row of coefficients
    if(!a[i])
  	{
		printf("Cannot allocate a[%d]!\n",i);
		exit(1);
  	}
  }
 
 x = (float *) malloc(num * sizeof(float)); // initial x values
 if(!x)
  {
	printf("Cannot allocate x!\n");
	exit(1);
  }


 b = (float *) malloc(num * sizeof(float)); // constants (b in Ax = b)
 if( !b)
  {
	printf("Cannot allocate b!\n");
	exit(1);
  }

 /* Filling in the blanks */ 

 /* The initial values of Xs */
 for(i = 0; i < num; i++)
	fscanf(fp,"%f ", &x[i]);
 
 for(i = 0; i < num; i++)
 {
   for(j = 0; j < num; j++)
     fscanf(fp,"%f ",&a[i][j]);
   
   /* reading the b element */
   fscanf(fp,"%f ",&b[i]);
 }
 
 fclose(fp); 

}


/************************************************************/


int parallelize() {
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int preverror = 0;

  
  // determine number of elements each process will work with (need to add its share of remainder)
  int remainder = num % comm_sz; 
  int distr = num / comm_sz; 
  int start;
  if (my_rank < remainder) {
  	start = my_rank * distr + my_rank;
  } else {
  	start = my_rank * distr + remainder;
  }
  int finish = start + distr;
  if (my_rank < remainder) finish += 1;

  // a and b don't change so we can use their global values and not have to distribute them
  // each process has an array x

  // put new x's in a new array
  // do error calculation in each process
  // if the percentage is over, set a flag that says to loop again
  // use reduce to find the max of those flags (1 means loop again)
  // Allgather x-news and put them into x-olds
  // Allgather automatically synchronizes everything!
  int i;
  int j;
  int sum_x;
  int maxerr = 0;
  int repeat = 1;
  while (repeat) { // change this to accept a flag
  	if (my_rank == 0) nit++;

  	float* new_x = (float*)malloc(num * sizeof(float*)); // keep track of newly calculated x values; use this in call to Allgather to create new x
    
    // calculate new x values based on old ones
    for (i = start; i < finish; i++) { // <= or < ?
      // subtract all x's (except the x at j) * their corresponding a[j] and divide by a[i]
      sum_x = 0;
      for (j = 0; j < num; j++) {
      	if (j != i) {
      	  sum_x -= x[j] * a[i][j]; // follow calculations from step 1 in instructions
      	}
      }
      new_x[i] = (b[i] + sum_x) / x[i];
    }

    // calculate percent error for each x in local_x, keeping track of the maximum
    // use this maximum to set the flag for whether or not to continue the while loop
    for (i = start; i < finish; i++) {
      if ((new_x[i] - x[i]) / new_x[i] > maxerr) { 
      	maxerr = (new_x[i] - x[i]) / new_x[i];
      }
    }

    int flag = 0;
    if (maxerr >= err) flag = 1; // if the highest percent error is greater than err, we need to keep looping

    // set error to the result of reducing the flags, based on whether or not maxerr <= err
    MPI_Allreduce(&flag, &repeat, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); // use flag from each process and * distribute result to each process *
    // if any of the processes returns 1 for the flag, repeat will be set to 1 and the loop will continue,
    // otherwise, repeat will be set to 0 and the loop will not continue

    int count = finish - start;
    printf("num: %f\n", num); // num is 0.000 here but correct in main...
    printf("%d\n", count);
    MPI_Allgather(new_x, count, MPI_FLOAT, x, count, MPI_FLOAT, MPI_COMM_WORLD); // concatenate all our new x values (from new_x[]) and put them into x[] (so they will be treated as the initial values of our next iteration)

    free(new_x); // values from new_x have been put into x so it is safe to free it
  }

  // x should now hold all values within the proper percent error
  // output these values to stdout (in main)

}

/************************************************************/


int main(int argc, char *argv[])
{
 int i;
 
  
 if( argc != 2)
 {
   printf("Usage: gsref filename\n");
   exit(1);
 }
  
 /* Read the input file and fill the global data structure above */
 get_input(argv[1]);
 // is it better to get all the input before using MPI and then make MPI calls on that global variable? Or use MPI to get the input? Can use MPI to get the input from the global arrays, divide the work accordingly
 
 /* Check for convergence condition */
 /* This function will exit the program if the coefficient will never converge to 
  * the needed absolute error. 
  * This is not expected to happen for this programming assignment.
  */

 printf("num in main:%d", num);
 check_matrix();
 
 MPI_Init(&argc, &argv);

 parallelize();

 MPI_Finalize();
 
 /* Writing to the stdout */
 /* Keep that same format */
 for( i = 0; i < num; i++) // why does this print for each core AFTER MPI_Finalize() has been called?
   printf("%f\n",x[i]);
 
 printf("total number of iterations: %d\n", nit);
 
 exit(0);

}