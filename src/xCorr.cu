/* 

Implements the grid-Correlation onto a GPU using CUDA. 

*/
#include <iostream>
#include "bifrost/Corr_romein.h"
#include "assert.hpp"
#include "trace.hpp"
#include "utils.hpp"
#include "cuda.hpp"
#include "cuda/stream.hpp"

#include "Complex.hpp"

#include <thrust/device_vector.h>


#define tile_x 32

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
__global__ void Corr(int npol, int gridsize, int nbatch,
		     cudaTextureObject_t   data_in,
		    // const In* __restrict__  d_in,
                     Out* d_out){

        int bid_x = blockIdx.x, bid_y = blockIdx.y ;
        int blk_x = blockDim.x;
        int grid_x = gridDim.x, grid_y = gridDim.y ;
        int tid_x = threadIdx.x;
	int pol_skip = grid_y*blk_x;

// Making use of shared memory for faster memory accesses by the threads

        extern __shared__  float2 shared[] ;
        float2* xx=shared;      
	// Access pattern is such that coaelescence is achieved both for read and writes to global and shared memory


        int bid=(bid_x*npol*grid_y+bid_y)*blk_x ;

// Reading texture cache as 1D with 1D thread-block indexing and copying it to shared memory
	
#pragma unroll
        for(int i=0;i<(npol/2);++i){
        xx[i*blk_x+tid_x] =tex1Dfetch<float2>(data_in, bid+i*pol_skip+tid_x);}
                __syncthreads();

// Iterate across polarizations to estimate XX*, YY*, XY*		
        #pragma unroll
        for(int i=0;i<npol-1;++i){
                d_out[bid+i*pol_skip+tid_x].x += xx[i/2*blk_x+tid_x].x*xx[i%2*blk_x+tid_x].x + xx[i/2*blk_x+tid_x].y*xx[i%2*blk_x+tid_x].y;   d_out[bid+i*pol_skip+tid_x].y += xx[i/2*blk_x+tid_x].y*xx[i%2*blk_x+tid_x].x - xx[i/2*blk_x+tid_x].x*xx[i%2*blk_x+tid_x].y ;}
        __syncthreads();
//  YX* is the same as XY*; just negate the imaginary part
        d_out[bid+3*pol_skip+tid_x].x=d_out[bid+2*pol_skip+tid_x].x;  d_out[bid+3*pol_skip+tid_x].y=(-1)*d_out[bid+2*pol_skip+tid_x].y;

	}
 

template<typename In, typename Out>
inline void launch_corr_kernel(int npol, bool polmajor, int gridsize, int nbatch,
                               In*  d_in,
                               Out* d_out,
                               cudaStream_t stream=0) {
   
    int grid_count = gridsize*gridsize ;
    int tile_y=1024/tile_x;             
    int block_y=grid_count/(tile_x*tile_y);
    int block_x=nbatch ;
    // Maximum thread blocks per device for GeForce cards on intrepid
    dim3 block(1024,1); /// Flattened one-D to reduce indexing arithmetic
   
//    cout << endl << " batch " << nbatch << " polz " << npol << " bool " << polmajor << endl ;
 
    dim3 grid(block_x, block_y);
    
    //  cout << endl << " batch " << nbatch << " polz " << npol << " bool " << polmajor << endl ;
    //cout << "  Block size is " << block.x << " by " << block.y << " by " << block.z << endl;
    //cout << "  Grid  size is " << grid.x << " by " << grid.y << " by " << grid.z << endl;
   
  

    // Determine how to create the texture object
    // NOTE:  Assumes some type of complex float
    cudaChannelFormatKind channel_format = cudaChannelFormatKindFloat;
    int dx = 32;
    int dy = 32;
    int dz = 0;
    int dw = 0;
    //if( sizeof(In) == sizeof(Complex64) ) {
      //  channel_format = cudaChannelFormatKindUnsigned;
      //  dz = 32;
     //   dw = 32;
  //  }

    // Create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_in;
    resDesc.res.linear.desc.f = channel_format;
    resDesc.res.linear.desc.x = dx;
    resDesc.res.linear.desc.y = dy;
    resDesc.res.linear.desc.z = dz;
    resDesc.res.linear.desc.w = dw;
    resDesc.res.linear.sizeInBytes = 2*grid_count*nbatch*sizeof(In);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t data_in;
    BF_CHECK_CUDA_EXCEPTION(cudaCreateTextureObject(&data_in, &resDesc, &texDesc, NULL),
                            BF_STATUS_INTERNAL_ERROR);





    void* args[] = {&npol,
                    &gridsize, 
                    &nbatch,
		    &data_in,
                  //  &d_in,
                    &d_out};
     size_t loc_size=2*block.x;//tile_x*tile_y;
	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)Corr<In,Out>,
						 grid, block,
						 &args[0], loc_size*sizeof(float2), stream),BF_STATUS_INTERNAL_ERROR);
   

 BF_CHECK_CUDA_EXCEPTION(cudaDestroyTextureObject(data_in),
                            BF_STATUS_INTERNAL_ERROR);



}

class BFcorr_impl {
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
    BFcorr_impl() : _npol(1), _polmajor(true), _stream(g_cuda_stream) {}
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
        
#define LAUNCH_CORR_KERNEL(IterType,OterType) \
        launch_corr_kernel(_npol, _polmajor, _gridsize, nbatch, \
                             (IterType)in->data, (OterType)out->data, \
                             _stream)
        
        switch( in->dtype ) {
            case BF_DTYPE_CI4:
                if( in->big_endian ) {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_CORR_KERNEL(nibble2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_CORR_KERNEL(nibble2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                } else {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_CORR_KERNEL(blenib2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_CORR_KERNEL(blenib2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                }
                break;
            case BF_DTYPE_CI8:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_CORR_KERNEL(char2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_CORR_KERNEL(char2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                };
                break;
            case BF_DTYPE_CI16:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_CORR_KERNEL(short2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_CORR_KERNEL(short2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_CORR_KERNEL(int2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_CORR_KERNEL(int2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_CORR_KERNEL(long2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_CORR_KERNEL(long2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_CORR_KERNEL(float2*, Complex32*);   break;
                    case BF_DTYPE_CF64: LAUNCH_CORR_KERNEL(float2*, Complex64*);   break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_CORR_KERNEL(double2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_CORR_KERNEL(double2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        }
#undef LAUNCH_CORR_KERNEL
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
    }
    void set_stream(cudaStream_t stream) {
        _stream = stream;
    }
};

BFstatus bfCorrCreate(BFcorr* plan_ptr) {
    BF_TRACE();
    BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN_ELSE(*plan_ptr = new BFcorr_impl(),
                       *plan_ptr = 0);
}
BFstatus bfCorrInit(BFcorr       plan,
                      BFsize         gridsize,
                      BFbool         polmajor) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
        
    int npol=4 ;
     
    BF_TRY_RETURN(plan->init(npol, polmajor, gridsize));
}
BFstatus bfCorrSetStream(BFcorr    plan,
                           void const* stream) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}
BFstatus bfCorrExecute(BFcorr          plan,
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
    BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_TRY_RETURN(plan->execute(in, out));
}

BFstatus bfCorrDestroy(BFcorr plan) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    delete plan;
    return BF_STATUS_SUCCESS;
}
