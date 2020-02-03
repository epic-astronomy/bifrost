#include <stdio.h>  
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string.h>  
#include <ctype.h>
#include <unistd.h> 
#include <sys/types.h> 
#include <fcntl.h>   
#include <errno.h>   
#include <termios.h> 
#include <fstream>
#include <cuda_runtime.h>
#include <complex.h>
#include <cuComplex.h>
#include <cuda.h>
#include <cufft.h>
#include <sys/time.h>
#include <cuda.h>
#include <cufft.h>
//#include <stdint>



//nvcc -o fft_mac_16_Nint256 fpa_fft_uGcorr_v16.cu --ptxas-options=-v --use_fast_math -L/usr/local/cuda/lib -lcudart -lcufft

typedef float2 Complex;

//#define _FILE_OFFSET_BITS 64	// for handling files greater than 2 GB

// The filter size is assumed to be a number smaller than the signal size
//#define SIGNAL_SIZE        65536

//#define ant_num            128
//#define freq_ch            512
//#define accum_len          1024  
//#define pattern_freq        64   
//#define nStreams           2


//#define f_tile            16
//#define a_tile            16   
         

//__device__ const int ant_blk_count = ant_num/(a_tile*nStreams);
//__device__ const uint stride_freq = freq_ch;
//__device__ const uint in_stride2  = accum_len*stride_freq; 
//__device__ const uint out_stride2 = ant_num*stride_freq/nStreams;  


//#define stride_freq    freq_ch
//#define in_stride2     accum_len*stride_freq
//#define out_stride2    ant_num*stride_freq    


#define MAX_THREADS_PER_BLOCK 512
#define MIN_BLOCKS_PER_MP     4


/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}



template<typename Complex, typename Complex>
__global__ void 
//__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
Corr(Complex<float>* a_ptr, Complex<float>* b_ptr)
{

  	int bid_x = blockIdx.x, bid_y = blockIdx.y, bid_z = blockIdx.z ;
  	int blk_x = blockDim.x, blk_y = blockDim.y, blk_z = blockDim.z ;
  	int grid_x = gridDim.x, grid_y = gridDim.y, grid_z = gridDim.z ;
  	int tid_x=threadIdx.x, tid_y = threadIdx.y, tid_z = threadIdx.z;
  	int tid_yz = tid_y*blk_z+tid_z;
  
  	extern  __shared__ Complex<float> shared[];
  	const int npol_in = 2, npol_out = 4 ;
        Complex<float>* xx=shared;
  	Complex<float> temp[npol_out];
  	
  	  
  	//int bid3=blockIdx.y*blockDim.y+blockIdx.z;
  	
  	int bid4=(bid_x*blk_x*grid_z*grid_y+bid_y*grid_z+bid_z)*blk_y*blk_z;
        int bid5= (bid_x*npol_out*grid_z*grid_y+bid_y*grid_z+bid_z)*blk_y*blk_z;
  
 	int tid2 =  tid_x%2*blk_z*blk_y+tid_yz ;
  	int tid  =  tid_x%2*blk_z*blk_y*grid_y*grid_z+tid_yz ;
  	xx[tid2]=b_ptr[bid4+tid];}
  	__syncthreads(); 
        
        //int tid3 = tid_x%2*blk_z*blk_y+tid_yz ;
        int tid4 = tid_x/2*blk_z*blk_y+tid_yz ;
        int tid5 = tid_x*blk_x*blk_y+tid_yz ;
         
        temp[tid_x] = a_ptr[bid5+tid5];
        temp[tid_x].x += xx[tid4].x*xx[tid2].x + xx[tid4].y*xx[tid2].y ; 
        temp[tid_x].y += xx[tid4].y*xx[tid2].x - xx[tid4].x*xx[tid2].y;
        a_ptr[bid5+tid5] += temp[tid_x];}
       __syncthreads();
}

// main routine
/*
int timing (int argc, char **argv)
{
       
   int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda( cudaSetDevice(devId) );

  // create events and streams
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[nStreams];
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  checkCuda( cudaEventCreate(&dummyEvent) );
  for (int i = 0; i < nStreams; ++i)
    checkCuda( cudaStreamCreate(&stream[i]) );



 
     cudaEvent_t start,stop;

       cufftHandle* handle = (cufftHandle*) malloc(sizeof(cufftHandle)*nStreams);
cudaEventCreate(&start);  cudaEventCreate(&stop); cudaEventRecord(start, 0);
    cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&ms, start, stop);
  

  
  
    
cudaEventCreate(&start);  cudaEventCreate(&stop); cudaEventRecord(start, 0); 
  
  cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&ms, start, stop);
 
 
  printf("\n\nTime for FFT & MAC Execution in loop : %f ms\n\n",ms);

  printf("\n\nProcess completed !! \n\n");


 cudaDeviceSynchronize();
 
//  cudaFreeHost(h_signal);
//  cudaFreeHost(h_Corr_p);
//  gpuErrchk(cudaFree(d_fft_signal));
//  gpuErrchk(cudaFree(d_Corr_p));
//  gpuErrchk(cudaFree(d_signal));
 cudaDeviceReset();
  
  return 0 ;
}*/
