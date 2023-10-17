/*
 * Copyright (c) 2018-2020, The Bifrost Authors. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of The Bifrost Authors nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* 

This module adopts the same work distribution strategy outlined in Romein(2011) to grid 
raw antenna voltages generating a X- and Y- complex uv grid. It replaces the earlier romein gridding
kernel as this module implements a higher level of parallelism with improved performance and hence is an 
upgraded version to the romein_kernel. Note that VGrid is specific to EPIC and cannot be used to grid visibilities
in its current form.

*/
#include <iostream>
#include <bifrost/VGrid.h>
#include "romein_kernels.cuh"

#include "assert.hpp"
#include "trace.hpp"
#include "utils.hpp"
#include "cuda.hpp"
#include "cuda/stream.hpp"
#include "Complex.hpp"

#define MAX_THREADS_PER_BLOCK 128
#define MIN_BLOCKS_PER_MP     4

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
inline Complex<RealType> Complexfcma(Complex<RealType> x, Complex<RealType> y, Complex<RealType> d) {
    RealType real_res;
    RealType imag_res;

    // This function is used to grid voltages, where we want the kernel (x) to be conjugated.
    // So here we do conj(x) * y
    real_res = (x.x *  y.x) + d.x;
    imag_res = (x.x *  y.y) + d.y;

    real_res =  (x.y * y.y) + real_res;
    imag_res = -(x.y * y.x) + imag_res;

    return Complex<RealType>(real_res, imag_res);
}


template<typename InType, typename OutType>
__global__ void 
//__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
VGrid_kernel(      int   		       npol,
		   int                         maxsupport, 
		   int                         gridsize, 
		   const int* __restrict__     x,
		   const int* __restrict__     y,
		   const int* __restrict__     z,
		   const OutType* __restrict__ kernels,
		   const InType* __restrict__  d_in,
		   OutType*                    d_out) {
        
	/// Thread and Block Indexes and Dimensions

    int bid_x = blockIdx.x, bid_y = blockIdx.y, bid_z = blockIdx.z ;
    int blk_x = blockDim.x, blk_y = blockDim.y;
    int grid_y = gridDim.y, grid_z = gridDim.z ;
    int illum_x = threadIdx.x, tid_y = threadIdx.y ;

    int vi_s = (bid_y+bid_x*grid_y)*grid_z*blk_y*npol ;
    int grid_s = (bid_y+bid_x*grid_y)*npol*gridsize*gridsize;

    /// Dynamic allocation of shared memory

    extern __shared__ float shared[];
    
    float* xdata = shared;
    float* ydata = xdata + blk_y * npol;
    
    /// Store X- and Y- Locations of Antennas within a thread-block in shared memory

    #pragma unroll
    for(int kk=0;kk<npol;kk++)
    {
          xdata[tid_y*npol + kk] = x[vi_s + npol *(bid_z*blk_y+tid_y)+kk];	
	  ydata[tid_y*npol + kk] = y[vi_s + npol *(bid_z*blk_y+tid_y)+kk];
    }  
    __syncthreads();

       int myU = illum_x% maxsupport; int myV = illum_x / maxsupport;
       float grid_point_u = myU; float grid_point_v = myV;
       OutType sum = OutType(0.0, 0.0);
  
       #pragma unroll
       for(int pol = 0; pol < npol;pol++) 
       {
                 
	    float xl = xdata[tid_y*npol+pol]; 
	    float yl = ydata[tid_y*npol+pol];
            // Determine convolution point. This is basically just an
            // optimised way to calculate.
            float myConvU = 0.0; 
	    float myConvV = 0.0;
            if( maxsupport > 1 ) 
	    {
                myConvU = (int(xl) - myU) % maxsupport; myConvV = (int(yl) - myV) % maxsupport;    
                if (myConvU < 0) myConvU += maxsupport; if (myConvV < 0) myConvV += maxsupport;
            } 
            // Determine grid point. Because of the above we know here that
           float myGridU = xl + myConvU; float myGridV = yl + myConvV;
           
       	   // If grid-point changes for a given illumination pattern 

          if (!(myGridU == grid_point_u && myGridV == grid_point_v)) 
	  { 
               // Round Step for UV grid pos. to avoid int() rounding down always
               grid_point_u = round(grid_point_u);
               grid_point_v = round(grid_point_v);

               // Loop to zero if above the grid size (maybe change to grid_point_X %= gridsize for speed/clarity?) [CT]
               if( round(grid_point_u) >= gridsize )
               {
               grid_point_u = 0.0;
               }

               if( round(grid_point_v) >= gridsize )
               {
               grid_point_v = 0.0;
               }


               if( grid_point_u >= 0 && grid_point_u < gridsize && \
                    grid_point_v >= 0 && grid_point_v < gridsize ) 
	        {
                       // Atomic add to fix autocorrelation problems 
                       atomicAdd(&d_out[grid_s + pol*gridsize*gridsize + gridsize*int(grid_point_v) + int(grid_point_u)].x, sum.x);
                       atomicAdd(&d_out[grid_s + pol*gridsize*gridsize + gridsize*int(grid_point_v) + int(grid_point_u)].y, sum.y);
                }
	        // Switch to new point
                sum = OutType(0.0, 0.0);
                grid_point_u = myGridU; grid_point_v = myGridV;
           }
            //TODO: Re-do the w-kernel/gcf for our data.
            OutType px = kernels[((bid_z*blk_y+tid_y)*npol+vi_s+pol)*maxsupport*maxsupport + int(myConvV) * maxsupport + int(myConvU)];
            // Sum up
            InType temp = d_in[(bid_z*blk_y+tid_y)*npol+vi_s+pol];
            OutType vi_v = OutType(temp.x, temp.y);
            sum=Complexfcma(px, vi_v, sum);     
 
            /// Add to the grid within an illumination pattern

            if( grid_point_u >= 0 && grid_point_u < gridsize && \
                grid_point_v >= 0 && grid_point_v < gridsize ) 
	    {
                // Round Step for UV grid pos. to avoid int() rounding down always 
                grid_point_u = round(grid_point_u);
                grid_point_v = round(grid_point_v);

                // Loop to zero if above the grid size (maybe change to grid_point_X %= gridsize for speed/clarity?) [CT]
                if( round(grid_point_u) >= gridsize )
                {
                grid_point_u = 0.0;
                }

                if( round(grid_point_v) >= gridsize )
                {
                grid_point_v = 0.0;
                }

                // Atomic Add to fix autocorrelation problems 
                atomicAdd(&d_out[grid_s + pol*gridsize*gridsize + gridsize*int(grid_point_v) + int(grid_point_u)].x, sum.x);
                atomicAdd(&d_out[grid_s + pol*gridsize*gridsize + gridsize*int(grid_point_v) + int(grid_point_u)].y, sum.y);

           }
       }/// End of polarization loop
    __syncthreads();
 }

template<typename InType, typename OutType>
inline void launch_VGrid_kernel(int      nbaseline,
                                 int      npol,
                                 bool     polmajor,
                                 int      maxsupport, 
                                 int      gridsize, 
                                 int      nbatch,
                                 int      nchan,
				 float*     xpos,
                                 float*     ypos,
                                 float*     zpos,
                                 OutType* kernels,
                                 InType*  d_in,
                                 OutType* d_out,
                                 cudaStream_t stream=0) {
    
    cudaDeviceProp dev;
    cudaError_t error;
    error = cudaGetDeviceProperties(&dev, 0);
    if(error != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(error));
    int block_x=maxsupport*maxsupport ;
    int block_y ;
    if(block_x==1)block_y=std::min(nbaseline,dev.maxThreadsPerBlock/2);
    else block_y = std::min((int)nbaseline/block_x,dev.maxThreadsPerBlock/2);
    int grid_z=nbaseline/block_y ;
    dim3 block(block_x,block_y);
    if(polmajor)npol=1;
    dim3 grid(nbatch, nchan, grid_z);
    
    void* args[] = {&npol,
                    &maxsupport,
                    &gridsize, 
                    &xpos,
                    &ypos,
                    &zpos,
                    &kernels,
                    &d_in,
                    &d_out};
    size_t loc_size = 2 * block.y * npol * sizeof(float);
    BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)VGrid_kernel<InType,OutType>,
						 grid, block,&args[0], loc_size, stream),BF_STATUS_INTERNAL_ERROR);
    
}

class BFVGrid_impl {
    typedef int    IType;
    typedef double FType;
public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
    typedef float  DType;
private:
    IType        _nbaseline;
    IType        _npol;
    bool         _polmajor;
    IType        _maxsupport;
    IType        _gridsize;
    IType        _nxyz = 0;
    float*         _x = NULL;
    float*         _y = NULL;
    float*         _z = NULL;    
    IType        _nkernels = 0;
    BFdtype      _tkernels = BF_DTYPE_INT_TYPE;
    void*        _kernels = NULL;
    cudaStream_t _stream;
public:
    BFVGrid_impl() : _nbaseline(1), _npol(1), _polmajor(true), \
                      _maxsupport(1), _stream(g_cuda_stream) {}
    inline IType nbaseline()  const { return _nbaseline;  }
    inline IType npol()       const { return _npol;       }
    inline bool polmajor()    const { return _polmajor;   }
    inline IType maxsupport() const { return _maxsupport; }
    inline IType gridsize()   const { return _gridsize;   }
    inline IType nxyz()       const { return _nxyz;       }
    inline IType nkernels()   const { return _nkernels;   }
    inline IType tkernels()   const { return _tkernels;   }
    void init(IType nbaseline,
              IType npol,
              bool  polmajor,
              IType maxsupport, 
              IType gridsize) {
        BF_TRACE();
        _nbaseline  = nbaseline;
        _npol       = npol;
        _polmajor   = polmajor;
        _maxsupport = maxsupport;
        _gridsize   = gridsize;
    }
    void set_positions(BFarray const* positions) { 
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        //BF_ASSERT_EXCEPTION(positions->dtype == BF_DTYPE_I32, BF_STATUS_UNSUPPORTED_DTYPE);
        
        int npositions = positions->shape[1];
        int stride = positions->shape[1];
	for(int i=2; i<positions->ndim-2; ++i) {
            npositions *= positions->shape[i];
	    stride *= positions->shape[i];
	}
	stride *= positions->shape[positions->ndim-2];
	stride *= positions->shape[positions->ndim-1];
	_nxyz = npositions;
        _x = (float *) positions->data;
        _y = _x + stride;
        _z = _y + stride;
    }
    void set_kernels(BFarray const* kernels) {
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        BF_ASSERT_EXCEPTION((kernels->dtype == BF_DTYPE_CF32) \
                                              || (kernels->dtype == BF_DTYPE_CF64), BF_STATUS_UNSUPPORTED_DTYPE);
        
        int nkernels = kernels->shape[0];
        for(int i=1; i<kernels->ndim-4; ++i) {
            nkernels *= kernels->shape[i];
        }
        
        _nkernels = nkernels;
        _tkernels = kernels->dtype;
        _kernels = (void*) kernels->data;
    }
    void execute(BFarray const* in, BFarray const* out, int nbatch, int nchan) {
        BF_TRACE();
        BF_TRACE_STREAM(_stream);
        BF_ASSERT_EXCEPTION(_x != NULL, BF_STATUS_INVALID_STATE);
	BF_ASSERT_EXCEPTION(_y != NULL, BF_STATUS_INVALID_STATE);
	BF_ASSERT_EXCEPTION(_z != NULL, BF_STATUS_INVALID_STATE);
        BF_ASSERT_EXCEPTION(_kernels != NULL, BF_STATUS_INVALID_STATE);
        BF_ASSERT_EXCEPTION((out->dtype == BF_DTYPE_CF32) \
                                          || (out->dtype == BF_DTYPE_CF64), BF_STATUS_UNSUPPORTED_DTYPE);
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
        
               
#define LAUNCH_VGRID_KERNEL(IterType,OterType) \
        launch_VGrid_kernel(_nbaseline, _npol, _polmajor, _maxsupport, _gridsize, nbatch, nchan, \
                             _x, _y, _z, (OterType)_kernels,		\
                             (IterType)in->data, (OterType)out->data, \
                             _stream)
        
        switch( in->dtype ) {
            case BF_DTYPE_CI4:
                if( in->big_endian ) {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_VGRID_KERNEL(nibble2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_VGRID_KERNEL(nibble2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                } else {
                    switch( out->dtype ) {
                        case BF_DTYPE_CF32: LAUNCH_VGRID_KERNEL(blenib2*, Complex32*);  break;
                        case BF_DTYPE_CF64: LAUNCH_VGRID_KERNEL(blenib2*, Complex64*);  break;
                        default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                    };
                }
                break;
            case BF_DTYPE_CI8:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_VGRID_KERNEL(char2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_VGRID_KERNEL(char2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                };
                break;
            case BF_DTYPE_CI16:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_VGRID_KERNEL(short2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_VGRID_KERNEL(short2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_VGRID_KERNEL(int2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_VGRID_KERNEL(int2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CI64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_VGRID_KERNEL(long2*, Complex32*); break;
                    case BF_DTYPE_CF64: LAUNCH_VGRID_KERNEL(long2*, Complex64*); break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF32:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_VGRID_KERNEL(float2*, Complex32*);   break;
                    case BF_DTYPE_CF64: LAUNCH_VGRID_KERNEL(float2*, Complex64*);   break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            case BF_DTYPE_CF64:
                switch( out->dtype ) {
                    case BF_DTYPE_CF32: LAUNCH_VGRID_KERNEL(double2*, Complex32*);  break;
                    case BF_DTYPE_CF64: LAUNCH_VGRID_KERNEL(double2*, Complex64*);  break;
                    default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
                }
                break;
            default: BF_ASSERT_EXCEPTION(false, BF_STATUS_UNSUPPORTED_DTYPE);
        }
#undef LAUNCH_VGRID_KERNEL
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
    }
    void set_stream(cudaStream_t stream) {
        _stream = stream;
    }
};

BFstatus bfVGridCreate(BFvgrid* plan_ptr) {
    BF_TRACE();
    BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN_ELSE(*plan_ptr = new BFVGrid_impl(),
                       *plan_ptr = 0);
}

BFstatus bfVGridInit(BFvgrid       plan,
                      BFarray const* positions,
                      BFarray const* kernels,
                      BFsize         gridsize,
                      BFbool         polmajor) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(positions,                                BF_STATUS_INVALID_POINTER);
    BF_ASSERT(positions->ndim >= 4,                     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(positions->shape[0] == 3, BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(space_accessible_from(positions->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_ASSERT(kernels,                                BF_STATUS_INVALID_POINTER);
    BF_ASSERT(kernels->ndim >= 5,                     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(kernels->shape[kernels->ndim-2] \
              == kernels->shape[kernels->ndim-1],     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(space_accessible_from(kernels->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    
    // Discover the dimensions of the positions/kernels.
    int npositions, nbaseline, npol, nkernels, maxsupport;
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
    nkernels = kernels->shape[0];
    for(int i=1; i<kernels->ndim-4; ++i) {
        nkernels *= kernels->shape[i];
    }
    maxsupport = kernels->shape[kernels->ndim-1];
    
    // Validate
    BF_ASSERT(npositions == nkernels, BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(kernels->shape[kernels->ndim-4] \
              == positions->shape[positions->ndim-2], BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(kernels->shape[kernels->ndim-3] \
              == positions->shape[positions->ndim-1], BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(kernels->shape[kernels->ndim-2] \
              == kernels->shape[kernels->ndim-1], BF_STATUS_INVALID_SHAPE);
    
    BF_TRY(plan->init(nbaseline, npol, polmajor, maxsupport, gridsize));
    BF_TRY(plan->set_positions(positions));
    BF_TRY_RETURN(plan->set_kernels(kernels));
}
BFstatus bfVGridSetStream(BFvgrid    plan,
                           void const* stream) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
    BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}
BFstatus bfVGridSetPositions(BFvgrid       plan,
                              BFarray const* positions) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(positions,            BF_STATUS_INVALID_POINTER);
    BF_ASSERT(positions->ndim >= 4, BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(positions->shape[0] == 3,                                     BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(space_accessible_from(positions->space,   BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    if( plan->polmajor() ) {
        BF_ASSERT(positions->shape[positions->ndim-2] == plan->npol(),      BF_STATUS_INVALID_SHAPE  );
        BF_ASSERT(positions->shape[positions->ndim-1] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE  );
    } else {
        BF_ASSERT(positions->shape[positions->ndim-2] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE  );
        BF_ASSERT(positions->shape[positions->ndim-1] == plan->npol(),      BF_STATUS_INVALID_SHAPE  );
    }
    
    BF_TRY_RETURN(plan->set_positions(positions));
}
BFstatus bfVGridSetKernels(BFvgrid       plan, 
                            BFarray const* kernels) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(kernels,            BF_STATUS_INVALID_POINTER);
    BF_ASSERT(kernels->ndim >= 5, BF_STATUS_INVALID_SHAPE  );
    if( plan->polmajor() ) {
        BF_ASSERT(kernels->shape[kernels->ndim-4] == plan->npol(),      BF_STATUS_INVALID_SHAPE  );
        BF_ASSERT(kernels->shape[kernels->ndim-3] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE  );
    } else {
        BF_ASSERT(kernels->shape[kernels->ndim-4] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE  );
        BF_ASSERT(kernels->shape[kernels->ndim-3] == plan->npol(),      BF_STATUS_INVALID_SHAPE  );
    }
    BF_ASSERT(kernels->shape[kernels->ndim-2] == plan->maxsupport(), BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(kernels->shape[kernels->ndim-1] == plan->maxsupport(), BF_STATUS_INVALID_SHAPE  );
    BF_ASSERT(space_accessible_from(kernels->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    
    BF_TRY_RETURN(plan->set_kernels(kernels));
}
BFstatus bfVGridExecute(BFvgrid          plan,
                         BFarray const* in,
                         BFarray const* out) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
    BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
    BF_ASSERT( in->ndim >= 3,          BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->ndim == in->ndim+1, BF_STATUS_INVALID_SHAPE);
    
    int nbatch = in->shape[0];
    int nchan = in->shape[1];

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
  
    BF_ASSERT( in->shape[0] == plan->nxyz(),     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT( in->shape[0] == plan->nkernels(), BF_STATUS_INVALID_SHAPE);
    if( plan->polmajor() ) {
        BF_ASSERT( in->shape[1] == plan->npol(),      BF_STATUS_INVALID_SHAPE);
        BF_ASSERT( in->shape[2] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE);
    } else {
        BF_ASSERT( in->shape[1] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE);
        BF_ASSERT( in->shape[2] == plan->npol(),      BF_STATUS_INVALID_SHAPE);
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

    BF_ASSERT(out->shape[0] == plan->nxyz(),     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[0] == plan->nkernels(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[1] == plan->npol(),     BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[2] == plan->gridsize(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[3] == plan->gridsize(), BF_STATUS_INVALID_SHAPE);
    
    BF_ASSERT(out->dtype == plan->tkernels(),    BF_STATUS_UNSUPPORTED_DTYPE);
    
    BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
    BF_TRY_RETURN(plan->execute(in, out, nbatch, nchan));
}

BFstatus bfVGridDestroy(BFvgrid plan) {
    BF_TRACE();
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    delete plan;
    return BF_STATUS_SUCCESS;
}
