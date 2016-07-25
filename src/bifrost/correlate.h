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

#ifndef BF_CORRELATE_H_INCLUDE_GUARD_
#define BF_CORRELATE_H_INCLUDE_GUARD_

#include <bifrost/common.h>
#include <bifrost/memory.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct __attribute__((aligned(8))) {
	// Note: Using unions in here prevents vectorized ld/st (for some reason)
	float r, i;
} BFcomplex64;

enum {
	BF_GAINSTATE_CONVERGED   = 0,
	BF_GAINSTATE_FLAGGED     = 1,
	BF_GAINSTATE_UNCONVERGED = 2
};// BFgainstate;

// TODO: Add chan stride
BFstatus bfVisibilitiesFillHermitian(BFsize       nchan,
                                     BFsize       ninput,
                                     BFspace      space,
                                     //BFvisformat  format, // Must be BF_VISIBILITIES_MATRIX
                                     BFcomplex64* data,   // [nchan,ninput,ninput]
                                     BFsize       stride);

BFstatus bfApplyGains(BFsize  nchan,
                      BFsize  nstand,
                      BFsize  npol,
                      BFspace space,
                      BFbool  invert,
                      const BFcomplex64* X,
                      const BFcomplex64* G,
                      const int8_t*      states,
                      BFcomplex64*       Y);
	/*
int    bfTypeNbit(unsigned dtype)      { return dtype & BF_DTYPE_NBITS; }
BFbool bfTypeIsFloat(unsigned dtype)   { return dtype & BF_DTYPE_F_BIT; }
BFbool bfTypeIsComplex(unsigned dtype) { return dtype & BF_DTYPE_C_BIT; }
	*/

// HACK TESTING
BFstatus bfFoo(BFconstarray a,
               BFarray      b);

BFstatus bfSolveGains(BFconstarray V,      // [nchan,nstand^,npol^,nstand,npol] cf32
                      BFconstarray M,      // [nchan,nstand^,npol^,nstand,npol] cf32
                      BFarray      G,      // [nchan,pol^,nstand,npol] cf32
                      BFarray      flags,  // [nchan,nstand] i8
                      BFbool       l1norm,
                      float        l2reg,
                      float        eps,
                      int          maxiter,
                      int*         num_unconverged_ptr);

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
                      int*               num_unconverged_ptr);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_CORRELATE_H_INCLUDE_GUARD_
