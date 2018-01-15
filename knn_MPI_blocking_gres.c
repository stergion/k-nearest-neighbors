#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define MASTER 0
#define NLAB 10


struct timeval startwtime, endwtime;
double seq_time;


void mpi_file_read_obs(double **obs, int rank, int chunk, int cols, char *filename, int numtasks);
void mpi_file_read_L(int *L, int rank, int chunk, char *filename, int numtasks);
void fill_obs(double **obs, int rank, int chunk, int cols, char *filename);
void fill_labels(int *labels, int rank, int chunk, char *filename);
void print_dmatrix(double **matrix, int rows, int cols);
void print_imatrix(int **matrix, int rows, int cols);
double** alloc_matrix_seq_double(int rows, int cols);
int** alloc_matrix_seq_int(int rows, int cols);
void free_dMatrix(double **matrix);
void free_intMatrix(int **matrix);
void knn_search(double **X, double **Y, int *L, int rows, int cols, int **IDX, double **DIST, int **LBL,
				int Nbr, int rank);
int most_freq_element(int **Mat, int row, int mcols, int *arr, int acols);
float find_match(int *Arr, int **Mat, int rows, int matCols);
void init_dmat_inf(double **mat, int rows, int cols);


int main(int argc, char **argv){
	
	int cols, rows, kNbr, i;
	char *labFilename, *pointsFilename;
	int numtasks, rank = -1, next, prev, chunkSize, tagP = 1, tagL = 2, prov;
	
	if(argc != 6) {
		printf("----------------------------------------------------------------------\n");
		printf("This program needs 5 arguments.\n");
		printf("1st argument: the .txt file with the points as 'filename.txt'\n");
		printf("2nd argument: the .txt file with the labels* as 'filename.txt'\n	*for this version labels must be from 1-10\n");
		printf("3rd argument: the number of pionts\n");
		printf("4th argument: the dimentions of points\n");
		printf("5th argument: the k nearest neighbors\n");
		printf("----------------------------------------------------------------------\n");
		exit(1);
	}
	
	// get the arguments.
	pointsFilename = argv[1];	// filename of the poins
	labFilename = argv[2];		// filename of the labels
	rows = atoi(argv[3]);		// rows is the number of points
	cols = atoi(argv[4]);		// cols is the number of dimensions of each point
	kNbr = atoi(argv[5]);		// kNbr is number of nearest neighbors
	
	MPI_Status Stats[2];
	
	// initialise MPI
	MPI_Init(&argc, &argv);
	//MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &prov);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// determine previous and next process.
    prev = rank-1;
    next = rank+1;
    if (rank == 0)  prev = numtasks - 1;
    if (rank == (numtasks - 1))  next = 0;
	
	chunkSize = rows / numtasks;
	//printf("MPI_numtasks %d, %d\n", numtasks, chunkSize);
	
	// allocate memory for the matrixes 
	double **Xb, **Yb, **tempMat, **DIST, **Mtemp;
	int **IDX, **LBL, *L, *tempL, *Ltemp;
	
	Xb = alloc_matrix_seq_double(chunkSize, cols);
	Yb = alloc_matrix_seq_double(chunkSize, cols);
	L = malloc(chunkSize * sizeof(int));

	tempMat = alloc_matrix_seq_double(chunkSize, cols);
	tempL = malloc(chunkSize * sizeof(int));
	
	DIST = alloc_matrix_seq_double(chunkSize, kNbr);
	IDX = alloc_matrix_seq_int(chunkSize, kNbr);
	LBL = alloc_matrix_seq_int(chunkSize, kNbr);
	
	
	// fill matrixes with values
	mpi_file_read_obs(Xb, rank, chunkSize, cols, pointsFilename, numtasks);
	mpi_file_read_obs(Yb, rank, chunkSize, cols, pointsFilename, numtasks);
	mpi_file_read_L(L, rank, chunkSize, labFilename, numtasks);
	init_dmat_inf(DIST, chunkSize, kNbr);
	/* if(rank == 3) {
		for (i=0;i<chunkSize;i++){
			printf("%d", L[i]);
		}
	} */
	
	
	//printf("MPI_task %d: I WAIT FOR OTHER TASKS \n", rank);
	MPI_Barrier(MPI_COMM_WORLD);
	
	if(rank == MASTER){
		//printf("start knn... \n");
		gettimeofday (&startwtime, NULL);
	}
	
	
	for(i = 0; i < numtasks; i++){
		
		if(rank == MASTER) {
			MPI_Send(&(Yb[0][0]), chunkSize * cols, MPI_DOUBLE, next, tagP, MPI_COMM_WORLD);
			MPI_Send(&(L[0]), chunkSize, MPI_INT, next, tagL, MPI_COMM_WORLD);
			MPI_Recv(&(tempMat[0][0]), chunkSize * cols, MPI_DOUBLE, prev, tagP, MPI_COMM_WORLD, &Stats[0]);
			MPI_Recv(&(tempL[0]), chunkSize, MPI_INT, prev, tagL, MPI_COMM_WORLD,  &Stats[1]);
		}else {
			MPI_Recv(&(tempMat[0][0]), chunkSize * cols, MPI_DOUBLE, prev, tagP, MPI_COMM_WORLD, &Stats[0]);
			MPI_Recv(&(tempL[0]), chunkSize, MPI_INT, prev, tagL, MPI_COMM_WORLD, &Stats[1]);
			MPI_Send(&(Yb[0][0]), chunkSize * cols, MPI_DOUBLE, next, tagP, MPI_COMM_WORLD);
			MPI_Send(&(L[0]), chunkSize, MPI_INT, next, tagL, MPI_COMM_WORLD);
		}
		
		// if i==0 Yb is same as Xb, else
		//	Yb comes from the pre node
		if(i == 0) {
			//printf("task %d: %d \n",rank, i);
			knn_search(Xb, Yb, L, chunkSize, cols, IDX, DIST, LBL, kNbr, rank); 
		}else {
			//printf("task %d: %d \n",rank, i);
			knn_search(Xb, Yb, L, chunkSize, cols, IDX, DIST, LBL, kNbr, prev);
		}
		
		
		// Swap addresses
		Mtemp = Yb;
		Yb = tempMat;
		tempMat = Mtemp;
		
		Ltemp = L;
		L = tempL;
		tempL = Ltemp; 
	}
	
	// wait for all processes to finish and then get time.
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == MASTER){
		gettimeofday (&endwtime, NULL);
		seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
							+ endwtime.tv_sec - startwtime.tv_sec);
		printf("knn search:%.4f sec", seq_time);
	}
	
	// each processes finds a partial match percentage
	float match, wholeMatch;
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == MASTER){
		gettimeofday (&startwtime, NULL);
	}
	
	match = find_match(L, LBL, chunkSize, kNbr) * 100;

	//printf("task %d:partial match percentage is %f2%%.\n", rank, match);
	
	// and then combine all percentages
	MPI_Reduce (&match,&wholeMatch,1,MPI_FLOAT ,MPI_SUM, 0, MPI_COMM_WORLD);
	wholeMatch = wholeMatch / numtasks;
	
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == MASTER){
		gettimeofday (&endwtime, NULL);
		seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
							+ endwtime.tv_sec - startwtime.tv_sec);
		printf("\t label time: %.4f sec", seq_time);
		printf("\t match percentage: %.2f%% \n", wholeMatch);
	}
	
	
	// free allocated memory.
	free_dMatrix(Xb);
	free_dMatrix(Yb);
	free(L);
	
	free_dMatrix(tempMat);
	free(tempL);
	
	free_dMatrix(DIST);
	free_intMatrix(IDX);
	free_intMatrix(LBL);
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	
	return 0;
}


void mpi_file_read_obs(double **obs, int rank, int chunk, int cols, char *filename, int numtasks){
	int file_free = 0;
	MPI_Status status;
	
	if(rank == MASTER){
		file_free = 1;
	} else{
		//printf("task %d: Cant read file.\n", rank);
		MPI_Recv(&file_free, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &status);
	}
	
	if(file_free == 1){
		//printf("task %d: i got to fill my matrix.\n", rank);
		fill_obs(obs, rank, chunk, cols, filename);
	}
	
	if(rank != numtasks-1) MPI_Send(&file_free, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);
}


void mpi_file_read_L(int *L, int rank, int chunk, char *filename, int numtasks){
	int file_free = 0;
	MPI_Status status;
	
	if(rank == MASTER){
		file_free = 1;
	} else{
		//printf("task %d: Cant read label file.\n", rank);
		MPI_Recv(&file_free, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &status);
	}
	
	if(file_free == 1){
		//printf("task %d: i got to fill my labels.\n", rank);
		fill_labels(L, rank, chunk, filename);
	}
	
	if(rank != numtasks-1) MPI_Send(&file_free, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);
}
	

//Function to fill the table with the observations.
void fill_obs(double **obs, int rank, int chunk, int cols, char *filename)
{
 //printf("start filling matrix...\n");
 FILE *fin;
  /** The txt file contains all the observations in decimal numbers.
      Each line refers to a specific point and the dimensions of each
      point are splitted by a tab.
  **/

  char *str = (char *)malloc(2 * cols * sizeof(double));
  char *token = (char *)malloc(sizeof(double));
  fin = fopen(filename, "r");
  if (fin == NULL)
  {
    printf("Error opening the file...");
    exit(1);
  }
  int i = 0;
  int j;
  
  
  for(i = 0; i < (rank * chunk); i++){
	str = fgets(str, 2 * cols * sizeof(double), fin);
  }
  str = fgets(str, 2 * cols * sizeof(double), fin); //get a line of the txt file, which refers to one point.
  i = 0;
  while (str != NULL && i < chunk)
  {
	token = strtok(str, "\t"); //get one dimension per recursion.
	j = 0;
	while (token != NULL && j < cols)
	{
		obs[i][j] = atof(token);
		token = strtok(NULL, "\t");
		j++;
	}
	
    str = fgets(str, 2 * cols * sizeof(double), fin);
    i++;
  }
  fclose(fin);
  free(str);
  free(token);
}


void fill_labels(int *labels, int rank, int chunk, char *filename)
{
  int i;
  FILE *fin;
  char *str = (char *)malloc(sizeof(int)+1);
  
  fin = fopen(filename, "r");
  
  if (fin == NULL)
  {
    printf("Error opening the file...");
    exit(1);
  }
  
  for(i = 0; i < (rank * chunk); i++){
	str = fgets(str, sizeof(int)+1, fin);
  }
  
  str = fgets(str, sizeof(int)+1, fin);
  i = 0;
  while (str != NULL && i < chunk)
  {
    labels[i] = atoi(str);
    str = fgets(str, sizeof(int)+1, fin);
    i++;
  }
  
  fclose(fin);
  free(str);
}


void print_dmatrix(double **matrix, int rows, int cols){
	int i,j;
	
	printf("printing matrix... \n");
	
	for(i = 0; i < rows; i++){
		for(j = 0; j < cols; j++){
			
			printf("%f \t", matrix[i][j]);
		}
		
		printf("\n");
	}
}

void print_imatrix(int **matrix, int rows, int cols){
	int i,j;
	
	printf("printing matrix... \n");
	
	for(i = 0; i < rows; i++){
		for(j = 0; j < cols; j++){
			
			printf("%d \t", matrix[i][j]);
		}
		
		printf("\n");
	}
}


double **alloc_matrix_seq_double(int rows, int cols){
	int i;
	
	double **matrix= malloc(rows * sizeof(*matrix));
	if(matrix == NULL){
		printf("cant allocate memory.");
		exit(1);
	}
	matrix[0] = malloc(rows * cols * sizeof(**matrix));
	if(matrix[0] == NULL){
		printf("cant allocate memory.");
		exit(1);
	}
	for(i = 1; i < rows; i++){
		matrix[i] = matrix[0] + i * cols;
	}
	
	return matrix;
}

int** alloc_matrix_seq_int(int rows, int cols){
	int i;
	
	int **matrix= malloc(rows * sizeof(*matrix));
	if(matrix == NULL){
		printf("cant allocate memory.");
		exit(1);
	}
	matrix[0] = malloc(rows * cols * sizeof(**matrix));
	if(matrix[0] == NULL){
		printf("cant allocate memory.");
		exit(1);
	}
	for(i = 1; i < rows; i++){
		matrix[i] = matrix[0] + i * cols;
	}
	
	return matrix;
}

void free_dMatrix(double **matrix) {
	free(*matrix);
	free(matrix);
}


void free_intMatrix(int **matrix) {
	free(*matrix);
	free(matrix);
}

void knn_search(double **X, double **Y, int *L, int rows, int cols, int **IDX, double **DIST, int **LBL,
				int Nbr, int rank) {
	int i, j, k, index;
	double d;
	
	for(i = 0; i < rows; i++){
		index = 0;
		
		for(j = 0; j < rows; j++){
			if(i != j){
				d = 0;		// contains the euclidean distance between two points
				
				for(k = 0; k < cols; k++){		// cols is the number of dimensions of each point
					d = d + pow(X[i][k] - Y[j][k],2);
				}
				
				d = sqrt(d);
				
				// check if distance is smaller than the previous Nbr distances,
				// if true save it and forget the max distance
				for(index = 0; index < Nbr; index++){
					if(d < DIST[i][index]){
						for(k = Nbr-1; k > index; k--){
							DIST[i][k] = DIST[i][k-1];
							LBL[i][k] = LBL[i][k-1];
						}
						DIST[i][index] = d;
						IDX[i][index] = rank * rows + j;
						LBL[i][index] = L[j];
						//printf("%d",L[j]);
						break;
					}
				}
				
				
				
			}
		}
		
		
	}
}


int most_freq_element(int **Mat, int row, int mcols, int *arr, int acols){
	int i, max_freq, element = 0;
	
	// fill arr with zeros
	for(i = 0; i < acols; i++){
		arr[i] = 0;
	}
	
	for(i = 0; i < mcols; i++){
		arr[Mat[row][i]-1]++;
	}
	
	/* for(i = 0; i < acols; i++){
		printf("%d", arr[i]);
	}
	printf("\n"); */
	// find most frequent element
	max_freq = 0;
	for(i = 0; i < acols; i++){
		if(arr[i] > max_freq){
			max_freq = arr[i];
			element = i + 1;
			//printf("%d    %d\n", arr[i],element);
		}
	}
	//printf("max freq:%d    elemend:%d\n", max_freq,element);
	return element;
}

float find_match(int *Arr, int **Mat, int rows, int matCols){
	int i, element = 0;
	float count = 0;
	
	int *arr = malloc(NLAB * sizeof(int));
	
	for(i = 0; i < rows; i++){
		element = most_freq_element(Mat, i, matCols, arr, NLAB);
		//printf("%d", element);
		if(Arr[i] == element) {
			count++;
		}
		//printf("Arr[%d]:%d    elemend:%d	count:%f\n",i, Arr[i],element, count);
	}
	
	//free(arr);
	return count/(float)rows;
}


void init_dmat_inf(double **mat, int rows, int cols){
	int i,j;
	
	for(i = 0; i < rows; i++){
		for(j = 0; j < cols; j++){
			mat[i][j] = INFINITY;
		}
	}
}

