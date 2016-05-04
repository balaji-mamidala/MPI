// Distributed two-dimensional Discrete FFT transform
// Balaji Mamidala 
// ECE8893 Project 1

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;

void Transpose(Complex* h, int height, int width);
void Transform1D(Complex* h, int w, Complex* H);
void Transform1D_Inverse(Complex* H, int w, Complex* h);
void Truncate(Complex* data, int heigth, int width);

void Transform2D(const char* inputFN) 
{  
  InputImage image(inputFN);  // Create the helper object for reading the image 
  
  //Get data from the image width and height. Note height=nRows , width=nColumns
  int width = image.GetWidth();
  int height = image.GetHeight();
  //Assumption: The image data is a square matrix i.e. width = height

  //Data from the image is real. But it is declared as complex for ease of computation of  DFT
  Complex* data = new Complex[height * width];
  data = image.GetImageData();

  // nCpus is the number of CPUs. myRank is the rank of process/CPU
  int  nCpus, myRank;

  MPI_Comm_size(MPI_COMM_WORLD,&nCpus);
  MPI_Comm_rank(MPI_COMM_WORLD,&myRank);

  cout << "Number of CPUs= " << nCpus << " My rank= " << myRank << endl;

  //sRow is the first row to compute DFT for the process/CPU
  //eRow is last row to compute DFT for the process/CPU
  //Row_Cpu is the rows per process/CPU
  //Assumption: Number of rows and columns is exactly divisible by number of CPUs
  int sRow, eRow, Row_Cpu = 0;
  int nRows = height;

  Row_Cpu = nRows/nCpus;
  sRow = Row_Cpu * myRank;
  eRow = sRow + Row_Cpu - 1;
  
  cout << "Start row: " << sRow << " End Row: " << eRow << endl;

  //Array to store DFT of the row
  Complex* dft_1d =  new Complex[(eRow-sRow+1) * width];

  int h,w;
  //Calculate 1d dft for the rows from sRow to eRow
  for(h=sRow; h <= eRow; h++)
  {
    Transform1D( (data+(h*width)) , width , dft_1d+((h-sRow)*width) );
  }

  //Ensure that all the CPUs have finished calculating 1d dft for thier respective rows
  MPI_Barrier(MPI_COMM_WORLD);
  //Now all the CPUs have finished calcualting 1d dft for their respective rows

  //Array to store 1d dft from collected from all CPUs
  Complex* dft_all = new Complex[height * width]; 

  //Code to get Size of object Complex and MPI data types MPI_COMPLEX & MPI_LONG_DOUBLE
  //int size;
  //MPI_Type_size(MPI_LONG_DOUBLE, &size);
  //cout << "Size of Complex: " << sizeof(dft_r_all[0]) << endl;
  //cout << "Size of MPI_LONG_DOUBLE: " << size << endl;
 
  //Collect 1d dft data from all CPUs and store it in dft_r_all
  MPI_Allgather(dft_1d, (eRow-sRow+1)*width , MPI_LONG_DOUBLE , dft_all , (eRow-sRow+1)*width , MPI_LONG_DOUBLE , MPI_COMM_WORLD);
  
  //Write complete 1d dft to output.txt only for a particular rank
  if(0==myRank)
  {
    image.SaveImageData("MyAfter1d.txt", dft_all, width, height);
  }

  //Transpose dft_all so that rows and columns are swapped
  Transpose(dft_all, height, width);  

  //Calculate 1d dft on all rows from sRow to eRow
  for(h=sRow; h <= eRow; h++)
  {
    Transform1D( (dft_all+(h*width)) , width , dft_1d+((h-sRow)*width) );
  }

  //Ensure that all the CPUs have finished calculating 1d dft for thier respective rows
  MPI_Barrier(MPI_COMM_WORLD);
  //Now all the CPUs have finished calcualting 1d dft for their respective rows

  //Collect 1d dft data from all CPUs and store it in dft_r_all
  MPI_Allgather(dft_1d, (eRow-sRow+1)*width , MPI_LONG_DOUBLE , dft_all , (eRow-sRow+1)*width , MPI_LONG_DOUBLE , MPI_COMM_WORLD);
 
  //Transpose dft_all so that rows and columns are swapped
  Transpose(dft_all, height, width); 

  //2D dft calculated. Write to file.
  if(0==myRank)
  {
    image.SaveImageData("MyAfter2d.txt", dft_all, width, height);
  }
  

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  //Now compute 2D DFT inverse

  //Using dft_1d to store 1D inverse DFT of evry row for CPU
  //Calculate 1d inverse dft for the rows from sRow to eRow
  for(h=sRow; h <= eRow; h++)
  {
    Transform1D_Inverse( (dft_all+(h*width)) , width , dft_1d+((h-sRow)*width) );
  }

  //Ensure that all the CPUs have finished calculating 1d dft for thier respective rows
  MPI_Barrier(MPI_COMM_WORLD);
  //Now all the CPUs have finished calcualting 1d dft for their respective rows

  //Using dft_all to store 1d inverse dft collected from all CPUs
  //Collect 1d inverse dft from all CPUs
  MPI_Allgather(dft_1d, (eRow-sRow+1)*width , MPI_LONG_DOUBLE , dft_all , (eRow-sRow+1)*width , MPI_LONG_DOUBLE , MPI_COMM_WORLD);

  //Transpose dft_all so that rows and columns are swapped
  Transpose(dft_all, height, width);
  
  //Calculate 1d inverse  dft on all rows from sRow to eRow
  for(h=sRow; h <= eRow; h++)
  {
    Transform1D_Inverse( (dft_all+(h*width)) , width , dft_1d+((h-sRow)*width) );
  }

  //Ensure that all the CPUs have finished calculating 1d dft for thier respective rows
  MPI_Barrier(MPI_COMM_WORLD);
  //Now all the CPUs have finished calcualting 1d dft for their respective rows

  //Collect 1d inverse dft data from all CPUs and store it
  MPI_Allgather(dft_1d, (eRow-sRow+1)*width , MPI_LONG_DOUBLE , dft_all , (eRow-sRow+1)*width , MPI_LONG_DOUBLE , MPI_COMM_WORLD); 

  //Transpose dft_all so that rows and columns are swapped
  Transpose(dft_all, height, width); 
  
  Truncate(dft_all, height, width);
 
  //2D dft calculated. Write to file.
  if(0==myRank)
  {
    image.SaveImageData("MyAfterInverse.txt", dft_all, width, height);
  } 


  delete(data);
  delete(dft_1d);
  delete(dft_all);
}


void Truncate(Complex* data, int height, int width)
{
  int r,c;

  for(r=0; r< height; r++)
  {
    for(c=0; c<width; c++)
    {  
      if(data[(r*height) + c].Mag().real < (double)0.1) 
      {
        data[(r*height) + c].real = 0; 
        data[(r*height) + c].imag = 0;
      }
   }
  }

}


void Transpose(Complex* h, int height, int width)
{
  int i,j;
  for(i=0; i< height; i++)
  {
    for(j=0; j<i; j++)
    {
      Complex temp =  h[(j*height) + i];
      h[(j*height) + i] =  h[(i*width) + j];
      h[(i*width) + j] = temp;
    }
  }
}


void Transform1D(Complex* h, int w, Complex* H)

{
  int n,k;
  //H(k) = Sum(n)(0:N-1) {TW_nk * h(n)} 
  for(k=0; k<w; k++)
  {
    //Initialize H[k] to 0
    H[k].real = 0;
    H[k].imag = 0;

    for(n=0; n<w; n++)
    {
      Complex TW_nk; // (Twiddle_facor)^(n*k)
      TW_nk.real = cos(2*M_PI*(n*k)/w);
      TW_nk.imag = -sin(2*M_PI*(n*k)/w);
      H[k] = H[k] + (TW_nk * h[n]);
    }
  }

}


void Transform1D_Inverse(Complex* H, int w, Complex* h)
{
  int n,k;
  //h(k) = 1/K * Sum(k)(0:K-1) {TW_nk * H(k)} 
  for(n=0; n<w; n++)
  {
    //Initialize h[n] to 0
    h[n].real = 0;
    h[n].imag = 0;

    for(k=0; k<w; k++)
    {
      Complex TW_nk; // (Twiddle_facor)^(n*k)
      TW_nk.real = cos(2*M_PI*(n*k)/w);
      TW_nk.imag = sin(2*M_PI*(n*k)/w);
      h[n] = h[n] + (TW_nk * H[k]);
    }

    h[n].real = h[n].real / (double)w;
    h[n].imag = h[n].imag / (double)w;

  }
}


int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line

  int rc;
  //Initialize MPI
  rc = MPI_Init(&argc,&argv);

  if (rc != MPI_SUCCESS)
  {
    cout << "Error starting MPI program. Terminating.\n";
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  Transform2D(fn.c_str()); // Perform the transform.

  // Finalize MPI here
  //cout << "Rank " << rank << " exiting normally" << endl;

  MPI_Finalize();
}

