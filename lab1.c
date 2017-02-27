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
  float* local_x;
  int recvcount;
  int preverror = 0;

  /*
  // determine number of elements each process will work with (need to add its share of remainder)
  int remainder = num % comm_sz;
  int distr = num / comm_sz; 
  int start = my_rank * distr + min(my_rank, remainder); 
  int finish = start + distr;
  if (my_rank < remainder) finish += 1;
  */

  // allocate space for arrays to pass to MPI_Scatterv
  int* counts = (int*)malloc(comm_sz * sizeof(int*));
  int* displace = (int*)malloc(comm_sz * sizeof(int*));
  if (my_rank == comm_sz) { // if last process, recvcount is all the elements the previous processes didn't receive
  	recvcount = num - ((num - 1) * floor(num / comm_sz));
  } else {
  	recvcount = floor(num / comm_sz);
  }

  int i;
  int j;
  for (i = 0; i < sizeof(counts); i++) {
    if (i == sizeof(counts) - 1) {
   	  counts[i] = num - ((num - 1) * floor(num / comm_sz)); // all remaining elements of a
    } else {
   	  counts[i] = floor(num / comm_sz);
    }
    displace[i] = i * num * floor(num / comm_sz); // displace[my_rank] gives the displacement of the x value in local_x
  }

  // even though a is an array of pointers, it should still work to pass MPI_FLOAT as the datatype
  // a and b don't change so we can use their global values and not have to distribute them
  MPI_Scatterv(x, counts, displace, MPI_FLOAT, local_x, recvcount, MPI_FLOAT, 0, MPI_COMM_WORLD); // divide x amongst processes


  // calculate which portion of x's it is going to find using rank

  int sum_x;
  int maxerr = 0;
  while (preverror > err) {
  	if (my_rank == 0) nit++;

  	// calculate each X value using values from b and a until percent error is less than err or error is same as previously calculated error
  	float* temp_x = (float*)malloc(sizeof(local_x) * sizeof(float*)); 
  	for (i = 0; i < sizeof(local_x); i++) {
      temp_x[i] = local_x[i]; // keep original values in temp_x
    }
    
    for (i = 0; i < sizeof(local_x); i++) {
      // subtract all x's (except the x at j) * their corresponding a[j] and divide by a[i]
      sum_x = 0;
      for (j = 0; j < sizeof(local_x); j++) {
      	if (j != i) {
      		sum_x -= temp_x[j] * local_a[i][j]; // follow calculations from step 1 in instructions
      		// ** THIS SHOULD BE BASED ON ALL X'S --> You will have to get values from x actually and update x for the next iteration... but that would mean we want each process to have their own copy of all of x so that they don't overwrite one another in case they're on different iterations.
      		// ** SOMEWHERE ABOVE you'll want to use MPI_Broadcast to send a copy of x to each process 
      		// either broadcast to every process and each has a local copy of
      	}
      }
      local_x[i] = (b[i] + sum_x) / temp_x[i];
    }

    // calculate percent error with new local_x values
    // preverror is the maximum of all the errors for each x calculated individually
    // ** preverror IS based on a single block of x values, but programs should not end until ALL of the preverrors are less than maxerr (?)

    for (i = 0; i < sizeof(local_x); i++) { // calculate percent error for each x in local_x, keeping track of the maximum
      if ((local_x[i] - temp_x[i]) / local_x[i] > maxerr) { 
      	// Can we calculate percent errors from different iterations at the same time? Sure! Once we are done with the local_x's from one process, that process will be over and its x's finalized -- each having percent error less than err. The other processes must keep going until their own percent errors are smaller than err PROBLEM: I'M CALCULATING VALUES BASED ON ONLY A SUBSET OF X'S, BUT SHOULD BE BASING IT OFF OF ALL X VALUES (each value of x depends on every other value of x, not just the one in the same block) -- ** see notes in previous for loop!!
      	preverror = (local_x[i] - temp_x[i]) / local_x[i];
      }
    }

    /*
	float totalerr;
    // Determine overall preverr across all x's by summing with MPI_Reduce, then dividing by the number of processes
    MPI_Allreduce(preverror, totalerr, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    // Allreduce because we want each process to end if totalerr <= err

    preverror = totalerr; // use maxerr as preverr on next iteration
    */

    preverror = maxerr; // use maxerr as preverr on next iteration

    free(temp_x);
  }

  // call MPI_Gather on final local_x's -- ** How to make sure this is only done once ALL processes have finished the while loop?
  if (my_rank == 0) {
  	MPI_Gatherv(local_x, sizeof(local_x), MPI_FLOAT, x, counts, displace, MPI_FLOAT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Gatherv(local_x, sizeof(local_x), MPI_FLOAT, x, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
  }

  // x should now hold all values within the proper percent error
  // output these values to stdout (in main)

  free(counts);
  free(displace);

  /* 


  Keep a local array that is made up of a single group of rows in a and the corresponding b values (a[i] through a[j] and b[i] through b[j]); these also correspond to x values j through i in our global array x (but each process will have their own temp array)

  use these values to calculate the new x's, which we can store in a temp array and pass to the next call of parallelize, along with the percent error most recently achieved

  calculate the new percent error and compare to the one passed in. If same, exit. Otherwise, check if less than the given %e. If not, copy the temp array into the global array of x's / replace the pointer reference with a reference to this new array and pass that array to another call to parallelize(), freeing the temp array because it is already store in global x (the next call to parallelize will create a new temp array)
    - make sure that you're using the values from the array passed in in your calculations and only changing the temp array
  */
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
 check_matrix();
 
 MPI_Init(&argc, &argv);

/*  
  ** See page 112 for implementation of scatter, 113 for gather

  use MPI_Scatterv to distribute the calculations amongst processes
  int MPI Scatterv(
	void∗ sendbuf /∗ in ∗/,
	int∗ sendcounts /∗ in ∗/, // need to calculate this :(
	int∗ displacements /∗ in ∗/, // and these
	MPI Datatype sendtype /∗ in ∗/,
	void∗ recvbuf /∗ out ∗/,
	int recvcount /∗ in ∗/,
	MPI Datatype recvtype /∗ in ∗/,
	int root /∗ in ∗/,
	MPI Comm comm /∗ in ∗/);	

IN MY PROGRAM:
  int* counts = (int*)malloc(comm_sz * sizeof(int*)); 
  int* displace = (int*)malloc(comm_sz, sizeof(int*));

  // allocate number of elements that will be treated by each process
  MPI_Scatterv(a, )
  
  The single sendcount argument in a call to MPI Scatter is replaced by two array
arguments: sendcounts and displacements. Both of these arrays contain comm_sz
elements: sendcounts[q] is the number of objects of type sendtype being sent to
process q. Furthermore, displacements[q] specifies the start of the block that is
being sent to process q. The displacement is calculated in units of type sendtype.
So, for example, if sendtype is MPI INT, and sendbuf has type int∗, then the data
that is sent to process q will begin in location sendbuf + displacements[q]

Use MPI_Gatherv:
  int MPI Gatherv(
	void∗ sendbuf /∗ in ∗/,
	int sendcount /∗ in ∗/, 
	MPI Datatype sendtype /∗ in ∗/,
	void∗ recvbuf /∗ out ∗/,
	int∗ recvcounts /∗ in ∗/,
	int∗ displacements /∗ in ∗/,
	MPI Datatype recvtype /∗ in ∗/,
	int root /∗ in ∗/,
	MPI Comm comm /∗ in ∗/);
*/

 parallelize();

 MPI_Finalize();
 
 /* Writing to the stdout */
 /* Keep that same format */
 for( i = 0; i < num; i++) // why does this print for each core AFTER MPI_Finalize() has been called?
   printf("%f\n",x[i]);
 
 printf("total number of iterations: %d\n", nit);
 
 exit(0);

}