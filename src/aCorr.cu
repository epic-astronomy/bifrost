/* 

Implements the grid-Correlation onto a GPU using CUDA. 

*/
#include <iostream>
#include "bifrost/aCorr.h"
#include "assert.hpp"
#include "trace.hpp"
#include "utils.hpp"
#include "cuda.hpp"
#include "cuda/stream.hpp"
//#include <complex.h>
#include "Complex.hpp"

#include <thrust/device_vector.h>

#define tile 256 // Number of threads per thread-block

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
__global__ void ACorr(int nbaseline, int npol, int nbatch,
		     In* d_in,//cudaTextureObject_t   data_in,//In* d_in,
		     Out* d_out){

        int bid_x = blockIdx.x, bid_y = blockIdx.y ;
        int blk_x = blockDim.x;
        int grid_x = gridDim.x, grid_y = gridDim.y ;
        int tid_x = threadIdx.x ;
        int pol_skip = grid_y*blk_x;
// Making use of shared memory for faster memory accesses by the threads

/*        extern __shared__  float2 shared[] ;
        float2* xx=shared;      
        float2* yy=xx+ blk_x;
*/
        
	extern  __shared__ float2 shared[] ;
        In* xx = reinterpret_cast<In *>(shared);
        In* yy= xx+ blk_x; 
	  

// Access pattern is such that coaelescence is achieved both for read and writes to global and shared memory
        int bid1= bid_x*npol*pol_skip + bid_y*blk_x ;
	int bid2 = bid_x*pol_skip*npol*npol+bid_y*blk_x;
	// Reading texture cache as 2D with 1D thread-block indexing and copying it to shared memory
	xx[tid_x]=d_in[bid1+tid_x];// tex1Dfetch<float2>(data_in, bid1+tid_x); 
        yy[tid_x]=d_in[bid1+pol_skip+tid_x];// tex1Dfetch<float2>(data_in, bid1+pol_skip+tid_x);

// Estimate polarizations to estimate XX*, YY*, XY*		
        d_out[bid2+tid_x].x=xx[tid_x].x*xx[tid_x].x+xx[tid_x].y*xx[tid_x].y;
	d_out[bid2+tid_x].y=0;
	d_out[bid2+pol_skip+tid_x].x=yy[tid_x].x*yy[tid_x].x+yy[tid_x].y*yy[tid_x].y;
	d_out[bid2+pol_skip+tid_x].y=0;
        d_out[bid2+2*pol_skip+tid_x].x +=  xx[tid_x].x*yy[tid_x].x + xx[tid_x].y*yy[tid_x].y;
      	d_out[bid2+2*pol_skip+tid_x].y +=  xx[tid_x].y*yy[tid_x].x - xx[tid_x].x*yy[tid_x].y;   
	 __syncthreads();
//  YX* is the same as XY*; just negate the imaginary part
   	 d_out[bid2+3*pol_skip+tid_x].x=d_out[bid2+2*pol_skip+tid_x].x;  
         d_out[bid2+3*pol_skip+tid_x].y=(-1)*d_out[bid2+2*pol_skip+tid_x].y;
}
 
template<typename In, typename Out>
inline void launch_acorr_kernel(int nbaseline, int npol, bool polmajor, int nbatch,
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
     size_t thread_bound = dev.maxThreadsPerBlock;
     size_t tile_grid_y = nbaseline/tile;


    dim3 block(tile,1); /// Flattened one-D to reduce indexing arithmetic
    dim3 grid(nbatch,tile_grid_y);// 2D grid for time, frequency and antennas
    
/*


    // Determine how to create the texture object
    // NOTE:  Assumes some type of complex float
    cudaChannelFormatKind channel_format = cudaChannelFormatKindFloat;
    int dx = 32;
    int dy = 32;
    int dz = 0;
    int dw = 0;
    if( sizeof(In) == sizeof(Complex64) )
    {
        channel_format = cudaChannelFormatKindUnsigned;
        dz = 32;
        dw = 32;
    }
    // Create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_in;
    //resDesc.res.linear.desc = cudaCreateChannelDesc<In>(); //cudaCreateChannelDesc( 32, 32, 0, 0, cudaChannelFormatKindFloat );
    
    resDesc.res.linear.desc.f = channel_format;
    resDesc.res.linear.desc.x = dx;
    resDesc.res.linear.desc.y = dy;
    resDesc.res.linear.desc.z = dz;
    resDesc.res.linear.desc.w = dw;
    resDesc.res.linear.sizeInBytes = nbaseline*npol*nbatch*sizeof(In);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t data_in;
    BF_CHECK_CUDA_EXCEPTION(cudaCreateTextureObject(&data_in, &resDesc, &texDesc, NULL), BF_STATUS_INTERNAL_ERROR);


*/
    void* args[] = {&nbaseline,
	            &npol,
                    &nbatch,
		    &d_in,//&data_in,
                    &d_out};
     size_t loc_size=int(npol)*block.x; // Shared memory size to be allocated for the kernel
	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)ACorr<In,Out>,
						 grid, block,
						 &args[0], loc_size*sizeof(float2), stream),BF_STATUS_INTERNAL_ERROR);
   

//   BF_CHECK_CUDA_EXCEPTION(cudaDestroyTextureObject(data_in),BF_STATUS_INTERNAL_ERROR);



}

class BFaCorr_impl {
    typedef int    IType;
    typedef double FType;
public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
    typedef float  DType;

private:
    IType        _nbaseline;
    IType        _npol;
    bool         _polmajor;
    cudaStream_t _stream;
public:
    BFaCorr_impl() : _nbaseline(1), _npol(1), _polmajor(true), \
                      _stream(g_cuda_stream) {}
    inline IType nbaseline()  const { return _nbaseline;  }
    inline IType npol()       const { return _npol;       }
    inline bool polmajor()    const { return _polmajor;   }
    void init(IType nbaseline,
              IType npol,
              bool  polmajor) {
        BF_TRACE();
        _nbaseline  = nbaseline;
        _npol       = npol;
        _polmajor   = polmajor;
    }
   void execute(BFarray const* in, BFarray const* out) {
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        BF_ASSERT_EXCEPTION(out->dtype == BF_DTYPE_CF32 \
                                          || BF_DTYPE_CF64, BF_STATUS_UNSUPPORTED_DTYPE);
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
        
        int nbatch = in->shape[1]*in->shape[2];
        
#define LAUNCH_ACORR_KERNEL(IterType,OterType) \
        launch_acorr_kernel(_nbaseline, _npol, _polmajor, nbatch, \
                             (IterType)in->data, (OterType)out->data, \
                             _stream)
        
        switch( in->dtype ) {
            case BF_DTYPE_CI4:
                if( in->big_endian ) {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(nibble2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(nibble2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                } else {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(blenib2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(blenib2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                }
                break;
            case BF_DTYPE_CI8:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(char2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(char2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                };
                break;
            case BF_DTYPE_CI16:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(short2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(short2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(int2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(int2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(long2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(long2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(float2*, Complex32*);   break;
                    case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(float2*, Complex64*);   break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_ACORR_KERNEL(double2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_ACORR_KERNEL(double2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        }
#undef LAUNCH_ACORR_KERNEL
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
    }
    void set_stream(cudaStream_t stream) {
        _stream = stream;
    }
};

BFstatus bfaCorrCreate(BFacorr* plan_ptr) {
    BF_TRACE();
    BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN_ELSE(*plan_ptr = new BFaCorr_impl(),
                       *plan_ptr = 0);
}
BFstatus bfaCorrInit(BFacorr       plan,
                      BFarray const* positions,
                      BFbool         polmajor) {
  
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(positions,                                BF_STATUS_INVALID_POINTER);
    BF_ASSERT(positions->ndim >= 4,                     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(positions->shape[0] == 3, BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(space_accessible_from(positions->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    
    // Discover the dimensions of the positions/kernels.
    int npositions, nbaseline, npol;
    npositions = positions->shape[1];
    for(int i=2; i<positions->ndim-2; ++i) {
        npositions *= positions->shape[i];
    }
    if( polmajor ) {
         npol = positions->shape[positions->ndim-2];
         nbaseline = positions->shape[positions->ndim-1];
    } else {
        nbaseline = positions->shape[positions->ndim-2];
        npol = positions->shape[positions->ndim-1];
    }

    // Validate
    BF_TRY_RETURN(plan->init(nbaseline, npol, polmajor));
}
BFstatus bfaCorrSetStream(BFacorr    plan,
                           void const* stream) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}
BFstatus bfaCorrExecute(BFacorr          plan,
                         BFarray const* in,
                         BFarray const* out) {

    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
    BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
    BF_ASSERT( in->ndim >= 3,          BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->ndim == in->ndim, BF_STATUS_INVALID_SHAPE);

    BFarray in_flattened;
    if( in->ndim > 3 ) {
        // Keep the last two dim but attempt to flatten all others
        unsigned long keep_dims_mask = padded_dims_mask(in);
        keep_dims_mask |= 0x1 << (in->ndim-1);
        keep_dims_mask |= 0x1 << (in->ndim-2);
        keep_dims_mask |= 0x1 << (in->ndim-3);
        flatten(in,   &in_flattened, keep_dims_mask);
        in  =  &in_flattened;
        BF_ASSERT(in_flattened.ndim == 3, BF_STATUS_UNSUPPORTED_SHAPE);
    }

    if( plan->polmajor() ) {
        BF_ASSERT( in->shape[1] == plan->npol(),      BF_STATUS_INVALID_SHAPE);
        BF_ASSERT( in->shape[2] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE);
    } else {
        BF_ASSERT( in->shape[1] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE);
        BF_ASSERT( in->shape[2] == plan->npol(),      BF_STATUS_INVALID_SHAPE);
    }


    BFarray out_flattened;

    if( out->ndim > 3 ) {
        // Keep the last two dim but attempt to flatten all others
        unsigned long keep_dims_mask = padded_dims_mask(in);
        keep_dims_mask |= 0x1 << (out->ndim-1);
        keep_dims_mask |= 0x1 << (out->ndim-2);
        keep_dims_mask |= 0x1 << (out->ndim-3);
        flatten(out,   &out_flattened, keep_dims_mask);
        out  =  &out_flattened;
        BF_ASSERT(out_flattened.ndim == 3, BF_STATUS_UNSUPPORTED_SHAPE);
    }
  /*  cout << "Input Dimension : " << in_flattened.ndim << " Output Dimension :  " << out_flattened.ndim << endl ;

     //  BF_ASSERT( out->shape[1] == plan->npol()**2,      BF_STATUS_INVALID_SHAPE);
  // cout << out->shape[0] << "  " << out->shape[1] << "  " << out->shape[2] << "  " << out->shape[3] << endl ;
//   cout <<  in->shape[0] << "  " << in->shape[1] << "  " << in->shape[2]  << "  " << in->shape[3] << "  " << in->shape[4] << "  " << in->shape[5] << endl ;


    std::cout << "OUT ndim = " << out->ndim << std::endl;
    std::cout << "   0 = " << out->shape[0] << std::endl;
     std::cout << "   1 = " << out->shape[1] << std::endl;
    std::cout << "   2 = " << out->shape[2] << std::endl;
     std::cout << "   3 = " << out->shape[3] << std::endl;

    std::cout << "IN ndim = " << in->ndim << std::endl;
    std::cout << "   0 = " << in->shape[0] << std::endl;
    std::cout << "   1 = " << in->shape[1] << std::endl;
    std::cout << "   2 = " << in->shape[2] << std::endl;
    std::cout << "   3 = " << in->shape[3] << std::endl;
*/
   
     BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
   // cout<<"Memory accessible from " << space_accessible_from( in->space, BF_SPACE_CUDA)<<endl ;
    BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
   // cout<<"Memory accessible for output " <<space_accessible_from(out->space, BF_SPACE_CUDA) << endl;
    BF_TRY_RETURN(plan->execute(in, out));
}

BFstatus bfaCorrDestroy(BFacorr plan) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    delete plan;
    return BF_STATUS_SUCCESS;
}
