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

// TODO: Visibility generator (point + gaussian sources)
//         Source flux linearly interpolated Stokes as function of freq (check units)
//         Source position (what basis? far and near field?)
//         Source size (units?)
//         Input positions (note: ideally per-pol not per-stand)
//         Channel frequencies
// TODO: Gridding
//         Use cusparse? Perhaps not (too much redundancy?)
//           Gather approach?
//             Maybe just wait for more info from Levi etc.
//         **Start with just nearest point scatter
/*
BFstatus bfGridNearest(BFsize nbatch,
                       BFsize npoint,
                       BFsize ngridx,
                       BFsize ngridy,
                       const BFcomplex64* data,
                       BFsize data_stride,
                       const int*         coords,
                       BFsize coords_stride,
                       BFcomplex64*       grid
                       BFsize grid_stride) {
	
}
*/
#include <thrust/functional.h>

inline __host__ __device__
int project_triangular(int i, int j) {
	// Note: Assumes i >= j
	return i*(i+1)/2 + j;
}
inline __host__ __device__
void lift_triangular(int b, int* i, int* j) {
	// Note: Returned values obey i >= i
	// Warning: This works up to nant=4607; beyond that, float64 is required
	*i = int((sqrtf(8*b+1)-1)/2);
	*j = b - project_triangular(*i, 0);
}
inline __host__ __device__
int project_square(int i, int j, int n) {
	return i*n + j;
}
inline __host__ __device__
void lift_square(int b, int n, int* i, int* j) {
	*i = b / n;
	*j = b - project_square(*i, 0, n);
}
inline __host__ __device__
int triangular2square(int b, int n) {
	int i, j;
	lift_triangular(b, &i, &j);
	return project_square(i, j, n);
}
inline __host__ __device__
int square2triangular(int b, int n) {
	int i, j;
	lift_square(b, n, &i, &j);
	return project_triangular(i, j);
}

template<int TILE_DIM, int BLOCK_ROWS>
__global__
void fill_hermitian_kernel(int          ninput,
                           BFcomplex64* data,
                           int          iembed) {
	__shared__ BFcomplex64 tile[TILE_DIM][TILE_DIM+1];
	int blk = blockIdx.x;
	int bx, by;
	lift_triangular(blk, &bx, &by);
	int chan = blockIdx.y;
	data += chan*iembed*iembed;
	
	int ix = bx*TILE_DIM + threadIdx.x;
	int iy = by*TILE_DIM + threadIdx.y;
	ix = min(ix, ninput-1);
	//__syncthreads();
#pragma unroll
	for( int r=0; r<TILE_DIM; r+=BLOCK_ROWS ) {
		int iyr = iy+r;
		iyr = min(iyr, ninput-1);
		int offset = ix + iyr*iembed;
		BFcomplex64 val = data[offset];
		val.i = -val.i; // Conjugate
		tile[threadIdx.y+r][threadIdx.x] = val;
	}
	__syncthreads();
	int ox = by*TILE_DIM + threadIdx.x;
	// **TODO: Check the ix>=iy condition here
	if( ox >= ninput || ix >= iy ) return;//continue; // Cull excess threads
	int oy = blockIdx.x*TILE_DIM + threadIdx.y;
#pragma unroll
	for( int r=0; r<TILE_DIM; r+=BLOCK_ROWS ) {
		BFcomplex64 val = tile[threadIdx.x][threadIdx.y+r];
		int oyr = oy+r;
		if( oyr >= ninput ) return;//continue;
		int offset = ox + oyr*iembed;
		data[offset] = val;
	}
}

struct ResetState : public thrust::unary_function<int8_t,int8_t> {
	__host__ __device__
	inline int8_t operator()(int8_t state) const {
		return (state == BF_GAINSTATE_FLAGGED ?
		        BF_GAINSTATE_FLAGGED :
		        BF_GAINSTATE_UNCONVERGED);
	}
};

template<typename T, int WIDTH=32>
inline __device__ T warp_sum(T x) {
#pragma unroll
	for( int k=WIDTH>>1; k>=1; k>>=1 ) {
		x += __shfl_xor(x, k, WIDTH);
	}
	return x;
}

inline __host__ __device__
void print_matrix(JonesMat const& m, const char* name="") {
	//printf("%s: %f %fj, %f %fj; %f %fj, %f %fj\n",
	//       name,
	//       m.x.x.x, m.x.x.y, m.x.y.x, m.x.y.y,
	//       m.y.x.x, m.y.x.y, m.y.y.x, m.y.y.y);
}

// This implements one iteration of an alternating least squares solver
//   (aka StefCal, ADI iteration) for the equation Vij = Gi Mij Gj^.
// TODO: Block dims (x and y) can be tuned
//template<typename T>
//template<int BLOCKDIM_X, int BLOCKDIM_Y>
__global__
void gains_iteration_kernel(int stand0,         // First stand to compute
                            int nstand_compute, // No. stands to compute
                            int nstand,         // No. stands total
                            int nchan,          // No. channels
                             //JonesRowVec const* V,  // Measured vis [chan,input^,input]
                             //JonesRowVec const* M,  // Model vis    [chan,input^,input]
                             //JonesRowVec const* G0, // Input gains  [chan,pol,input]
                             //JonesRowVec*       G,  // Output gains [chan,pol,input]
                            float4 const* __restrict__ V,  // Measured vis [chan,input^,input]
                            float4 const* __restrict__ M,  // Model vis    [chan,input^,input]
                            // **TODO: Is restrict OK here even if they do actually overlap?
                            //           Note that in-place operation is non-deterministic regardless of this!
                            float4 const* __restrict__ G0, // Input gains  [chan,pol,input]
                            float4*       __restrict__ G,  // Output gains [chan,pol,input]
                            int8_t*       __restrict__ states, // [chan,stand]
                            int*          __restrict__ num_unconverged,
                            bool          odd_iteration,
                            bool  l1norm,
                            float l2reg,
                            float eps) {
	typedef float real_type;
	int i0 = threadIdx.x + blockIdx.x*blockDim.x;
	int c0 = threadIdx.y + blockIdx.y*blockDim.y;
	int thread_num_unconverged = 0;
	for( int c=c0; c<nchan; c+=blockDim.y*gridDim.y ) {
	for( int i=stand0+i0; i<stand0+nstand_compute; i+=blockDim.x*gridDim.x ) {
		JonesMat Gi(G0[i+nstand*(0 + 2*c)],
		            G0[i+nstand*(1 + 2*c)]);
		int8_t istate = states[i + nstand*c];
		if( istate != BF_GAINSTATE_UNCONVERGED ) {
			if( G != G0 ) {
				// Just copy the input value
				G[i+nstand*(0 + 2*c)] = Gi.x;
				G[i+nstand*(1 + 2*c)] = Gi.y;
			}
			continue;
		}
		JonesMat VZi(0);
		JonesMat ZZi(0);
		for( int j=0; j<nstand; ++j ) {
			if( j == i ) {
				continue; // Skip autocorrelations
			}
			int8_t jstate = states[j + nstand*c];
			if( jstate == BF_GAINSTATE_FLAGGED ) {
				continue; // Skip flagged stands
			}
			JonesMat Vij(V[i+nstand*(0 + 2*(j + nstand*c))],
			             V[i+nstand*(1 + 2*(j + nstand*c))]);
			JonesMat Mij(M[i+nstand*(0 + 2*(j + nstand*c))],
			             M[i+nstand*(1 + 2*(j + nstand*c))]);
			
			// TODO: Try this (make it optional), which may reduce global loads by almost 50%
			//JonesMat Mij(0);
			//for( int r=0; r<rank; ++r ) {
			// // TODO: Does each rank component need to be a JonesMat or just a vector?
			//	Mij.mad(M[i+nstand*(r + rank*c)],
			//	        M[j+nstand*(r + rank*c)]);
			//}
			
			JonesMat Gj(G0[j+nstand*(0 + 2*c)],
			            G0[j+nstand*(1 + 2*c)]);
			JonesMat Zij  = Mij*Gj.H();
			JonesMat WZij = Zij;
			if( l1norm ) {
				// Iteratively-reweighted least squares
				// Compute residual Rij = Vij - Gi * Mij * Gj.H()
				JonesMat Rij = Vij;
				Rij.msub(Gi, Zij);
				// Note: Avoids small denominators
				WZij.x.x *= rsqrtf(max(Rij.x.x.mag2(), eps*eps));
				WZij.x.y *= rsqrtf(max(Rij.x.y.mag2(), eps*eps));
				WZij.y.x *= rsqrtf(max(Rij.y.x.mag2(), eps*eps));
				WZij.y.y *= rsqrtf(max(Rij.y.y.mag2(), eps*eps));
			}
			VZi.mad(Vij, WZij.H());
			ZZi.mad(Zij, WZij.H());
		}
		if( l2reg > 0 ) {
			// Ridge regression
			ZZi.x.x.x += l2reg*l2reg;
			ZZi.y.y.x += l2reg*l2reg;
		}
		if( ZZi.is_singular(eps) ) {
			states[i + nstand*c] = BF_GAINSTATE_FLAGGED;
			if( G != G0 ) {
				// Just copy the input value
				G[i+nstand*(0 + 2*c)] = Gi.x;
				G[i+nstand*(1 + 2*c)] = Gi.y;
			}
			continue;
		}
		JonesMat Gi_new = VZi / ZZi;
		if( odd_iteration ) {
			real_type rel_res2 = (Gi_new - Gi).mag2() / Gi.mag2();
			if( rel_res2 <= eps*eps ) {
				states[i + nstand*c] = BF_GAINSTATE_CONVERGED;
				istate = BF_GAINSTATE_CONVERGED;
			}
			else {
				// Average odd iterations to ensure convergence
				Gi_new *= 0.5f;
				Gi_new += 0.5f*Gi;
			}
		}
		G[i+nstand*(0 + 2*c)] = Gi_new.x;
		G[i+nstand*(1 + 2*c)] = Gi_new.y;
		thread_num_unconverged += (istate == BF_GAINSTATE_UNCONVERGED);
	} // i
	} // c
	int warp_num_unconverged = warp_sum(thread_num_unconverged);
	int tid = threadIdx.x + blockDim.x*threadIdx.y;
	if( tid % warpSize == 0 ) {
		atomicAdd(num_unconverged, warp_num_unconverged);
	}
}

// TODO: Add stand and chan strides
//template<int BLOCKDIM_X, int BLOCKDIM_Y>
__global__
void apply_gains_kernel(int  nstand,         // No. stands
                        int  nchan,          // No. channels
                        bool invert,         // Invert gains before applying
                        float4 const* __restrict__ X,      // Input visibilities [chan,input^,input]
                        float4 const* __restrict__ G,      // Gain matrices [chan,pol,input]
                        int8_t const* __restrict__ states, // Gain flags [chan,stand]
                        float4*       __restrict__ Y) {    // Output visibilities [chan,input^,input]
	typedef float real_type;
	int i0 = threadIdx.x + blockIdx.x*blockDim.x;
	int j0 = threadIdx.y + blockIdx.y*blockDim.y;
	int c0 = threadIdx.z + blockIdx.z*blockDim.z;
	for( int c=c0; c<nchan; c+=blockDim.z*gridDim.z ) {
	for( int j=j0; j<nstand; j+=blockDim.y*gridDim.y ) {
	for( int i=i0; i<nstand; i+=blockDim.x*gridDim.x ) {
		JonesMat Gi(G[i+nstand*(0 + 2*c)],
		            G[i+nstand*(1 + 2*c)]);
		JonesMat Gj(G[j+nstand*(0 + 2*c)],
		            G[j+nstand*(1 + 2*c)]);
		JonesMat Xij(X[i+nstand*(0 + 2*(j + nstand*c))],
		             X[i+nstand*(1 + 2*(j + nstand*c))]);
		if( invert ) {
			Gi = Gi.inverse();
			Gj = Gj.inverse();
		}
		if( states ) {
			int8_t Si(states[i+nstand*c]);
			int8_t Sj(states[j+nstand*c]);
			if( Si == BF_GAINSTATE_FLAGGED ) {
				Gi = JonesMat(0);
			}
			if( Sj == BF_GAINSTATE_FLAGGED ) {
				Gj = JonesMat(0);
			}
		}
		JonesMat Yij = Gi*Xij*Gj.H();
		Y[i+nstand*(0 + 2*(j + nstand*c))] = Yij.x;
		Y[i+nstand*(1 + 2*(j + nstand*c))] = Yij.y;
	}
	}
	}
}
