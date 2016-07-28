/*
 * Copyright (c) 2016, The Bifrost Authors. All rights reserved.
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

// TODO: Polish off the functions in here ready for public release

#include <bifrost/correlate.h>
#include "assert.hpp"
#include "cuda/stream.hpp"
#include "Jones.hpp"
#include "correlate_kernels.cuh"
#include "cuda/stream.hpp"
#include "cuda/device_variable.hpp"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
/*
inline __host__ __device__
BFcomplex64 make_BFcomplex64(float r, float i=0) {
	BFcomplex64 c;
	c.r = r;
	c.i = i;
	return c;
}
*/

#if THRUST_VERSION >= 100800 // WAR for old Thrust version on TK1
#define thrust_cuda_par_on(stream) thrust::cuda::par.on(stream)
#else
#define thrust_cuda_par_on(stream) thrust::cuda::par(stream)
#endif

BFstatus fill_hermitian_cuda(BFsize       nchan,
                             BFsize       ninput,
                             BFspace      space,
                             BFcomplex64* data, // [nchan,ninput,ninput]
                             BFsize       stride) { // in elements
	enum {
		TILE_DIM   = 32,
		BLOCK_ROWS =  8
	};
        printf("Entered fill hermitian\n");
	cuda::scoped_stream stream;
	int ntile_side = (ninput-1) / TILE_DIM + 1;
	int ntile = ntile_side*(ntile_side+1)/2;
	dim3 grid(ntile);
	dim3 block(TILE_DIM, BLOCK_ROWS);
	
	fill_hermitian_kernel
		<TILE_DIM,BLOCK_ROWS>
		<<<grid,block,0,stream>>>
		(ninput,data,stride);
	
	return BF_STATUS_SUCCESS;
}

// TODO: Change the below functions to use this enum
typedef enum {
	BF_VISIBILITIES_MATRIX,     // Suitable for use with BLAS
	BF_VISIBILITIES_TRIANGULAR, // Memory efficient
	BF_VISIBILITIES_STORAGE     // Suitable for storage as UVFITS, MS etc.
} BFvisformat;

BFstatus bfVisibilitiesFillHermitian(BFsize       nchan,
                                     BFsize       ninput,
                                     BFspace      space,
                                     //BFvisformat  format, // Must be BF_VISIBILITIES_MATRIX
                                     BFcomplex64* data,   // [nchan,ninput,ninput]
                                     BFsize       stride) {
	if( space != BF_SPACE_CUDA ) {
		return BF_STATUS_UNSUPPORTED;
	}
	//try {
	BF_TRY(fill_hermitian_cuda(nchan, ninput, space, data, stride),
	       BF_NO_OP);
	//}
	//catch(std::runtime_error const& err) {
	//	std::cout << "HERE" << std::endl;
	//	return BF_STATUS_SUCCESS;
	//}
}

// TODO
BFstatus bfVisibilitiesReorder(BFsize             nchan,
                               BFsize             nstand,
                               BFsize             npol,
                               BFspace            space,
                               BFvisformat        iformat,
                               BFcomplex64 const* iptr,
                               BFsize             istride,
                               BFvisformat        oformat,
                               BFcomplex64*       optr,
                               BFsize             ostride);

// Y = G X G^
BFstatus bfApplyGains(BFsize  nchan,
                      BFsize  nstand,
                      BFsize  npol,
                      BFspace space,
                      BFbool  invert,
                      const BFcomplex64* X,
                      const BFcomplex64* G,
                      const int8_t*      states,
                      BFcomplex64*       Y) {
	BF_ASSERT(npol  == 2, BF_STATUS_UNSUPPORTED);
	BF_ASSERT(space == BF_SPACE_CUDA, BF_STATUS_UNSUPPORTED);
	BF_ASSERT(X,      BF_STATUS_INVALID_POINTER);
	BF_ASSERT(G,      BF_STATUS_INVALID_POINTER);
	//BF_ASSERT(states, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(Y,      BF_STATUS_INVALID_POINTER);
	enum {
		// TODO: Tune these
		BLOCKDIM_X = 256,
		BLOCKDIM_Y = 1
	};
	cuda::scoped_stream stream;
	dim3 grid(std::min((nstand-1)/BLOCKDIM_X+1, BFsize(65535)),
	          std::min((nchan-1) /BLOCKDIM_Y+1, BFsize(65535)));
	dim3 block(BLOCKDIM_X,
	           BLOCKDIM_Y);
	apply_gains_kernel
		//<BLOCKDIM_X,BLOCKDIM_Y>
		<<<grid,block,0,stream>>>
		(nstand, nchan, invert,
		 (float4*)X, (float4*)G, states, (float4*)Y);
	return BF_STATUS_SUCCESS;
}

//template<int N, typename T>
//inline bool shapes_equal(const T a[N], const T b[N], int ndim) {
template<typename T, typename U>
inline bool shapes_equal(T const& a, U const& b) {
	if( a.ndim != b.ndim ) {
                printf("%d != %d\n", a.ndim, b.ndim);
		return false;
	}
	for( int d=0; d<a.ndim; ++d ) {
		if( a.shape[d] != b.shape[d] ) {
			return false;
		}
	}
	return true;
}
/*
template<typename T>
inline bool valid_array(T const& a) {
	return (a.data &&
	        (a.space == BF_SPACE_SYSTEM ||
	         a.space == BF_SPACE_CUDA   ||
	         a.space == BF_SPACE_
}
*/
BFstatus bfSolveGains(BFconstarray V,      // Observed data. [nchan,nstand^,npol^,nstand,npol] cf32
                      BFconstarray M,      // The model. [nchan,nstand^,npol^,nstand,npol] cf32
                      BFarray      G,      // Jones matrices. [nchan,pol^,nstand,npol] cf32
                      BFarray      flags,  // [nchan,nstand] i8
                      BFbool       l1norm,
                      float        l2reg,
                      float        eps,
                      int          maxiter,
                      int*         num_unconverged_ptr) {
	BF_ASSERT(shapes_equal(V, M), BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(    V.ndim == 5, BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(    G.ndim == 4, BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(flags.ndim == 2, BF_STATUS_INVALID_SHAPE);
	BFsize nchan  = V.shape[0];
	BFsize nstand = V.shape[1];
	BFsize npol   = V.shape[2];
	BF_ASSERT(V.shape[3] == nstand, BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(V.shape[4] == npol,   BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(G.shape[0] == nchan,  BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(G.shape[1] == npol,   BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(G.shape[2] == nstand, BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(G.shape[3] == npol,   BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(flags.shape[0] == nchan,  BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(flags.shape[1] == nstand, BF_STATUS_INVALID_SHAPE);
	BF_ASSERT(V.space == M.space &&
	          M.space == G.space &&
	          G.space == flags.space,
	          BF_STATUS_INVALID_SPACE);
	BF_ASSERT(V.space == BF_SPACE_CUDA, BF_STATUS_UNSUPPORTED);
	BF_ASSERT(maxiter > 0, BF_STATUS_INVALID_ARGUMENT);
	
	BF_ASSERT(npol  == 2, BF_STATUS_UNSUPPORTED);
	BF_ASSERT(V.space == BF_SPACE_CUDA, BF_STATUS_UNSUPPORTED);
	BF_ASSERT(V.data, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(M.data, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(G.data, BF_STATUS_INVALID_POINTER);
	BF_ASSERT(flags.data, BF_STATUS_INVALID_POINTER);
	
	int convergence_status = bfSolveGains_old(nchan,
                         nstand,
                         npol, 
                         V.space,
                         (const BFcomplex64*) V.data,
                         (const BFcomplex64*) M.data,
                         (BFcomplex64*) G.data,
                         (int8_t*) flags.data,
                         l1norm,
                         l2reg,
                         eps,
                         maxiter,
                         num_unconverged_ptr);
	return BF_STATUS_SUCCESS;
}

BFstatus bfSolveGains_old(BFsize             nchan,
                      BFsize             nstand,
                      BFsize             npol,
                      BFspace            space,
                      const BFcomplex64* V,      // [nchan,nstand^,npol^,nstand,npol]
                      const BFcomplex64* M,      // [nchan,nstand^,npol^,nstand,npol]
                      BFcomplex64*       G,      // [nchan,pol^,nstand,npol]
                      int8_t*            states, // [nchan,nstand]
                      BFbool             l1norm,
                      float              l2reg,
                      float              eps,
                      int                maxiter,
                      int*               num_unconverged_ptr) {
	BF_ASSERT(npol  == 2, BF_STATUS_UNSUPPORTED);
	BF_ASSERT(space == BF_SPACE_CUDA, BF_STATUS_UNSUPPORTED);
	BF_ASSERT(V,      BF_STATUS_INVALID_POINTER);
	BF_ASSERT(M,      BF_STATUS_INVALID_POINTER);
	BF_ASSERT(G,      BF_STATUS_INVALID_POINTER);
	BF_ASSERT(states, BF_STATUS_INVALID_POINTER);
	enum {
		// TODO: Tune these
		BLOCKDIM_X = 256,
		BLOCKDIM_Y = 1
	};
	// TODO: Need to catch exceptions (e.g., from kernel failure)
	cuda::scoped_stream stream;
	// TODO: This is convenient, but should really allow user to manage
	//         temp space allocations themselves.
	// HACK
	/*thread_local*/ cuda::device_variable<int> num_unconverged(stream);
	thrust::transform(thrust_cuda_par_on(stream),
	                  /*
#if THRUST_VERSION >= 100800 // WAR for old Thrust version on TK1
	                  thrust::cuda::par.on(stream),
#else
	                  thrust::cuda::par(stream),
#endif
	                  */
	                  states, states + nchan*nstand, states, ResetState());
	dim3 grid(std::min((nstand-1)/BLOCKDIM_X+1, BFsize(65535)),
	          std::min((nchan-1) /BLOCKDIM_Y+1, BFsize(65535)));
	dim3 block(BLOCKDIM_X,
	           BLOCKDIM_Y);
	for( int it=0; it<maxiter; ++it ) {
		num_unconverged = 0;
		gains_iteration_kernel
			//<BLOCKDIM_X,BLOCKDIM_Y>
			<<<grid,block,0,stream>>>
			(0, nstand, nstand, nchan,
			 (float4*)V, (float4*)M,
			 (float4*)G, (float4*)G,
			 states, &num_unconverged,
			 (it%2==1),
			 l1norm, l2reg, eps);
		//cudaStreamSynchronize(stream);
		if( num_unconverged == 0 ) {
			if( num_unconverged_ptr ) {
				*num_unconverged_ptr = 0;
			}
			
			// **TODO: Apply phase referencing!
			
			std::cout << "CONVERGED AFTER " << it+1 << " ITERATIONS" << std::endl;
			return BF_STATUS_SUCCESS;
		}
                if ( it%(int(maxiter/100.0)) == 0 ){
		    std::cout << "Iteration " << it << ", num_unconverged = " << num_unconverged << std::endl;
                }
	}
	if( num_unconverged_ptr ) {
		*num_unconverged_ptr = num_unconverged;
	}
        printf("Failed to converge.\n");
	return BF_STATUS_FAILED_TO_CONVERGE;
}

// HACK TESTING
BFstatus bfFoo(BFconstarray a,
               BFarray      b) {
	std::cout << a.data << std::endl;
	std::cout << a.space << std::endl;
	std::cout << a.dtype << std::endl;
	std::cout << "** " << (a.dtype & BF_DTYPE_NBIT_BITS) << std::endl;
	std::cout << (a.dtype & BF_DTYPE_TYPE_BITS) << std::endl;
	std::cout << a.ndim << std::endl;
	std::cout << a.shape[0] << ", " << a.shape[1] << std::endl;
	//std::cout << a. << std::endl;
	return BF_STATUS_SUCCESS;
}
