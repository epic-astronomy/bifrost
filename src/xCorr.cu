/* 

Implements the grid-Correlation onto a GPU using CUDA. 

*/
#include <iostream>
#include "bifrost/xCorr.h"
#include "assert.hpp"
#include "trace.hpp"
#include "utils.hpp"
#include "cuda.hpp"
#include "cuda/stream.hpp"
//#include <complex.h>
#include "Complex.hpp"

#include <thrust/device_vector.h>

#define tile 256 // Number of threads per thread-block
#define tile_grid 65536 //Tiling the grid

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

struct __attribute__((aligned(1))) nibble2 {
    // Yikes!  This is dicey since the packing order is implementation dependent!  
    signed char y:4, x:4;
};

struct __attribute__((aligned(1))) blenib2 {
    // Yikes!  This is dicey since the packing order is implementation dependent!
    signed char x:4, y:4;
};
template<typename RealType>
__host__ __device__
inline Complex<RealType> ComplexMul(Complex<RealType> x, Complex<RealType> y, Complex<RealType> d) {
    RealType real_res;
    RealType imag_res;

    real_res = (x.x *  y.x) + d.x;
    imag_res = (x.x *  y.y) + d.y;

    real_res =  (x.y * y.y) + real_res;
    imag_res = -(x.y * y.x) + imag_res;

    return Complex<RealType>(real_res, imag_res);
}

template<typename In, typename Out>
__global__ void XCorr(int npol, int gridsize, int nbatch,
		     cudaTextureObject_t   data_in,
		     Out* d_out){

        int bid_x = blockIdx.x, bid_y = blockIdx.y, bid_z = blockIdx.z ;
        int blk_x = blockDim.x;
        int grid_x = gridDim.x, grid_y = gridDim.y, grid_z = gridDim.z ;
        int tid_x = threadIdx.x ;
        int pol_skip = grid_y*grid_z*blk_x;
// Making use of shared memory for faster memory accesses by the threads

        extern __shared__  float2 shared[] ;
        float2* xx=shared;      
        float2* yy=xx+tile ;  
// Access pattern is such that coaelescence is achieved both for read and writes to global and shared memory
        int bid = bid_x*npol*grid_y*grid_z*blk_x + bid_y*grid_z*blk_x + bid_z*blk_x;
        int bid_2 ;
	int bid_3 = bid_z*blk_x;
	// Reading texture cache as 2D with 1D thread-block indexing and copying it to shared memory
	if(gridsize>128)
	{ 
		bid_2=bid_x*grid_y*(int)npol/2+bid_y ; 
		xx[tid_x]= tex2D<float2>(data_in,bid_3+tid_x,bid_2);
                yy[tid_x]= tex2D<float2>(data_in,bid_3+tid_x,bid_2+grid_y);

	}
	else{ 
		bid_2=bid_x*grid_y+bid_y ;
		xx[tid_x]= tex2D<float2>(data_in,bid_3+tid_x,bid_2);
                yy[tid_x]= tex2D<float2>(data_in,bid_3+tid_x+grid_z*blk_x,bid_2);
	}

// Estimate polarizations to estimate XX*, YY*, XY*		
        d_out[bid+tid_x].x=xx[tid_x].x*xx[tid_x].x+xx[tid_x].y*xx[tid_x].y;
	d_out[bid+tid_x].y=0;
	d_out[bid+pol_skip+tid_x].x=yy[tid_x].x*yy[tid_x].x+yy[tid_x].y*yy[tid_x].y;
	d_out[bid+pol_skip+tid_x].y=0;
        d_out[bid+2*pol_skip+tid_x].x +=  xx[tid_x].x*yy[tid_x].x + xx[tid_x].y*yy[tid_x].y;
      	d_out[bid+2*pol_skip+tid_x].y +=  xx[tid_x].y*yy[tid_x].x - xx[tid_x].x*yy[tid_x].y;   
	 __syncthreads();
//  YX* is the same as XY*; just negate the imaginary part
   	 d_out[bid+3*pol_skip+tid_x].x=d_out[bid+2*pol_skip+tid_x].x;  
         d_out[bid+3*pol_skip+tid_x].y=(-1)*d_out[bid+2*pol_skip+tid_x].y;
}
 
template<typename In, typename Out>
inline void launch_xcorr_kernel(int npol, bool polmajor, int gridsize, int nbatch,
                               In*  d_in,
                               Out* d_out,
                               cudaStream_t stream=0) {
    cudaDeviceProp dev;
    cudaError_t error;
    error = cudaGetDeviceProperties(&dev, 0);
     if(error != cudaSuccess)
     {
        printf("Error: %s\n", cudaGetErrorString(error));
      }
    size_t grid_count = gridsize*gridsize ;
    size_t thread_bound = dev.maxThreadsPerBlock;
    size_t tile_grid_z ;
    size_t tile_grid_y;
  size_t width, height ;
    if(gridsize>128){ 
	    tile_grid_z = tile_grid/tile;
	    tile_grid_y = grid_count/tile_grid;
	    width =  tile_grid;  // number of data columns in matrix
	    height = nbatch*tile_grid_y*(int)npol/2 ;// number of data rows in matrix
    }
    else{   
	    width = grid_count*(int)npol/2; // number of data columns in matrix
	    tile_grid_z = grid_count/tile;
	    tile_grid_y= grid_count/(tile_grid_z*tile);
	    height = nbatch*tile_grid_y ;// number of data rows in matrix
    }
    
   
    dim3 block(tile,1); /// Flattened one-D to reduce indexing arithmetic
    dim3 grid(nbatch,tile_grid_y,tile_grid_z);// 2D grid for time, frequency and pixel indexing
   
    //Create Data Channel Format for texture chache 
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<In>();//cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat );
  // Create 2D-texture map for the input-data
    size_t pitch;    
    In* dataDev;
    checkCuda( cudaMallocPitch(&dataDev, &pitch, width * sizeof(In),  height) );
    checkCuda( cudaMemcpy2D(dataDev, pitch, d_in, width*sizeof(In), width*sizeof(In),height, cudaMemcpyDeviceToDevice) );
   
   // Create 2D texture object for the pitched 2D texture map
   
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc)) ;
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dataDev;
    resDesc.res.pitch2D.width  = width;
    resDesc.res.pitch2D.height =  height;
    resDesc.res.pitch2D.desc = channelDesc;
    resDesc.res.pitch2D.pitchInBytes = pitch;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t data_in;
    BF_CHECK_CUDA_EXCEPTION(cudaCreateTextureObject(&data_in, &resDesc, &texDesc, NULL), BF_STATUS_INTERNAL_ERROR);
    void* args[] = {&npol,
                    &gridsize, 
                    &nbatch,
		    &data_in,
                    &d_out};
     size_t loc_size=int(npol/2)*block.x; // Shared memory size to be allocated for the kernel
	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)XCorr<In,Out>,
						 grid, block,
						 &args[0], loc_size*sizeof(float2), stream),BF_STATUS_INTERNAL_ERROR);
   
 cudaFree(dataDev); // Clear the texture cache to resuse the resources
 BF_CHECK_CUDA_EXCEPTION(cudaDestroyTextureObject(data_in), BF_STATUS_INTERNAL_ERROR);

}

class BFxcorr_impl {
    typedef int    IType;
    typedef double FType;
public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
    typedef float  DType;
private:
    IType        _npol;
    bool         _polmajor;
    IType        _gridsize;
  //  BFdtype      _tkernels = BF_DTYPE_INT_TYPE;
    cudaStream_t _stream;
public:
    BFxcorr_impl() : _npol(1), _polmajor(true), _stream(g_cuda_stream) {}
    inline IType npol()       const { return _npol;       }
    inline bool polmajor()    const { return _polmajor;   }
    inline IType gridsize()   const { return _gridsize;   }
  //  inline IType tkernels()   const { return _tkernels;   }
    void init(IType npol, bool  polmajor, IType gridsize) {
        BF_TRACE();
        _npol       = npol;
        _polmajor   = polmajor;
        _gridsize   = gridsize;
    }
   void execute(BFarray const* in, BFarray const* out) {
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        BF_ASSERT_EXCEPTION(out->dtype == BF_DTYPE_CF32 \
                                          || BF_DTYPE_CF64, BF_STATUS_UNSUPPORTED_DTYPE);
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
        
        int nbatch = in->shape[1]*in->shape[2];
        
#define LAUNCH_XCORR_KERNEL(IterType,OterType) \
        launch_xcorr_kernel(_npol, _polmajor, _gridsize, nbatch, \
                             (IterType)in->data, (OterType)out->data, \
                             _stream)
        
        switch( in->dtype ) {
            case BF_DTYPE_CI4:
                if( in->big_endian ) {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_XCORR_KERNEL(nibble2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_XCORR_KERNEL(nibble2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                } else {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_XCORR_KERNEL(blenib2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_XCORR_KERNEL(blenib2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                }
                break;
            case BF_DTYPE_CI8:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_XCORR_KERNEL(char2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_XCORR_KERNEL(char2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                };
                break;
            case BF_DTYPE_CI16:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_XCORR_KERNEL(short2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_XCORR_KERNEL(short2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_XCORR_KERNEL(int2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_XCORR_KERNEL(int2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_XCORR_KERNEL(long2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_XCORR_KERNEL(long2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_XCORR_KERNEL(float2*, Complex32*);   break;
                    case BF_DTYPE_CF64: LAUNCH_XCORR_KERNEL(float2*, Complex64*);   break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_XCORR_KERNEL(double2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_XCORR_KERNEL(double2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        }
#undef LAUNCH_XCORR_KERNEL
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
    }
    void set_stream(cudaStream_t stream) {
        _stream = stream;
    }
};

BFstatus bfxCorrCreate(BFxcorr* plan_ptr) {
    BF_TRACE();
    BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN_ELSE(*plan_ptr = new BFxcorr_impl(),
                       *plan_ptr = 0);
}
BFstatus bfxCorrInit(BFxcorr       plan,
                      BFsize         gridsize,
                      BFbool         polmajor) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
        
    int npol=4 ;
     
    BF_TRY_RETURN(plan->init(npol, polmajor, gridsize));
}
BFstatus bfxCorrSetStream(BFxcorr    plan,
                           void const* stream) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}
BFstatus bfxCorrExecute(BFxcorr          plan,
                         BFarray const* in,
                         BFarray const* out) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
    BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
    BF_ASSERT( in->ndim == 6,          BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->ndim == in->ndim-1, BF_STATUS_INVALID_SHAPE);
    BFarray in_flattened;
    //cout << endl << " Input Dimension " << in->ndim << " Output Dimension " << out->ndim << endl ;
    if( in->ndim > 5 ) {
        // Keep the last three dim but attempt to flatten all others
        unsigned long keep_dims_mask = padded_dims_mask(out);
        keep_dims_mask |= 0x1 << (in->ndim-1);
        keep_dims_mask |= 0x1 << (in->ndim-2);
        keep_dims_mask |= 0x1 << (in->ndim-3);
        keep_dims_mask |= 0x1 << (in->ndim-4);
        keep_dims_mask |= 0x1 << (in->ndim-5);
        keep_dims_mask |= 0x1 << (in->ndim-6);
	flatten(in,   &in_flattened, keep_dims_mask);
        in  =  &in_flattened;
       BF_ASSERT(in_flattened.ndim == 6, BF_STATUS_UNSUPPORTED_SHAPE); 
    }

    BFarray out_flattened;
    if( out->ndim > 4 ) {
        // Keep the last three dim but attempt to flatten all others
        unsigned long keep_dims_mask = padded_dims_mask(out);
        keep_dims_mask |= 0x1 << (out->ndim-1);
        keep_dims_mask |= 0x1 << (out->ndim-2);
        keep_dims_mask |= 0x1 << (out->ndim-3);
        keep_dims_mask |= 0x1 << (out->ndim-4);
        flatten(out,   &out_flattened, keep_dims_mask);
        out  =  &out_flattened;
        BF_ASSERT(out_flattened.ndim == 4, BF_STATUS_UNSUPPORTED_SHAPE);
    }
// cout << out->shape[0] << "  " << out->shape[1] << "  " << out->shape[2] << "  " << out->shape[3] << endl ;
//   cout <<  in->shape[0] << "  " << in->shape[1] << "  " << in->shape[2]  << "  " << in->shape[3] << "  " << in->shape[4] << "  " << in->shape[5] << endl ;


//    std::cout << "OUT ndim = " << out->ndim << std::endl;
//    std::cout << "   0 = " << out->shape[0] << std::endl;
//    std::cout << "   1 = " << out->shape[1] << std::endl;
//    std::cout << "   2 = " << out->shape[2] << std::endl;
//    std::cout << "   3 = " << out->shape[3] << std::endl;


//    std::cout << "IN ndim = " << in->ndim << std::endl;
//    std::cout << "   0 = " << in->shape[0] << std::endl;
//    std::cout << "   1 = " << in->shape[1] << std::endl;
//    std::cout << "   2 = " << in->shape[2] << std::endl;
//   std::cout << "   3 = " << in->shape[3] << std::endl;
//    std::cout << "   4 = " << in->shape[4] << std::endl;




//    BF_ASSERT(out->shape[1] == plan->npol(),     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[2] == plan->gridsize(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[3] == plan->gridsize(), BF_STATUS_INVALID_SHAPE);
    
   // BF_ASSERT(out->dtype == plan->tkernels(),    BF_STATUS_UNSUPPORTED_DTYPE);
    
    BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
   // cout<<"Memory accessible from " << space_accessible_from( in->space, BF_SPACE_CUDA)<<endl ;
    BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
   // cout<<"Memory accessible for output " <<space_accessible_from(out->space, BF_SPACE_CUDA) << endl;
    BF_TRY_RETURN(plan->execute(in, out));
}

BFstatus bfxCorrDestroy(BFxcorr plan) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    delete plan;
    return BF_STATUS_SUCCESS;
}
