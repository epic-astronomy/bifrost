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

/*! \file common.h
 *  \brief Common definitions used throughout the library
 */

#ifndef BF_COMMON_H_INCLUDE_GUARD_
#define BF_COMMON_H_INCLUDE_GUARD_
#define BF_MAX_DIM 100

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


enum {
    BF_DTYPE_NBIT_BITS      = 0x00FF,
    BF_DTYPE_TYPE_BITS      = 0x0F00,
    BF_DTYPE_INT_TYPE       = 0x0000,
    BF_DTYPE_UINT_TYPE      = 0x0100,
    BF_DTYPE_FLOAT_TYPE     = 0x0200,
    BF_DTYPE_STRING_TYPE    = 0x0300,
    BF_DTYPE_COMPLEX_BIT    = 0x1000,

    BF_DTYPE_I1    =  1 | BF_DTYPE_INT_TYPE,
    BF_DTYPE_I2    =  2 | BF_DTYPE_INT_TYPE,
    BF_DTYPE_I4    =  4 | BF_DTYPE_INT_TYPE,
    BF_DTYPE_I8    =  8 | BF_DTYPE_INT_TYPE,
    BF_DTYPE_I16   = 16 | BF_DTYPE_INT_TYPE,
    BF_DTYPE_I32   = 32 | BF_DTYPE_INT_TYPE,
    BF_DTYPE_I64   = 64 | BF_DTYPE_INT_TYPE,

    BF_DTYPE_U1    =   1 | BF_DTYPE_UINT_TYPE,
    BF_DTYPE_U2    =   2 | BF_DTYPE_UINT_TYPE,
    BF_DTYPE_U4    =   4 | BF_DTYPE_UINT_TYPE,
    BF_DTYPE_U8    =   8 | BF_DTYPE_UINT_TYPE,
    BF_DTYPE_U16   =  16 | BF_DTYPE_UINT_TYPE,
    BF_DTYPE_U32   =  32 | BF_DTYPE_UINT_TYPE,
    BF_DTYPE_U64   =  64 | BF_DTYPE_UINT_TYPE,

    BF_DTYPE_F16   =  16 | BF_DTYPE_FLOAT_TYPE,
    BF_DTYPE_F32   =  32 | BF_DTYPE_FLOAT_TYPE,
    BF_DTYPE_F64   =  64 | BF_DTYPE_FLOAT_TYPE,
    BF_DTYPE_F128  = 128 | BF_DTYPE_FLOAT_TYPE,

    BF_DTYPE_CI1   =   1 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,
    BF_DTYPE_CI2   =   2 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,
    BF_DTYPE_CI4   =   4 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,
    BF_DTYPE_CI8   =   8 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,
    BF_DTYPE_CI16  =  16 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,
    BF_DTYPE_CI32  =  32 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,
    BF_DTYPE_CI64  =  64 | BF_DTYPE_INT_TYPE | BF_DTYPE_COMPLEX_BIT,

    BF_DTYPE_CF16  =  16 | BF_DTYPE_FLOAT_TYPE | BF_DTYPE_COMPLEX_BIT,
    BF_DTYPE_CF32  =  32 | BF_DTYPE_FLOAT_TYPE | BF_DTYPE_COMPLEX_BIT,
    BF_DTYPE_CF64  =  64 | BF_DTYPE_FLOAT_TYPE | BF_DTYPE_COMPLEX_BIT,
    BF_DTYPE_CF128 = 128 | BF_DTYPE_FLOAT_TYPE | BF_DTYPE_COMPLEX_BIT
};

typedef int                BFstatus;
typedef int                BFbool;
typedef int                BFenum;
typedef float              BFcomplex[2];
typedef float              BFreal;
typedef uint64_t           BFsize; // TODO: Check this on TK1 (32 bit)
//typedef unsigned long      BFsize;
//typedef size_t             BFsize;
//typedef unsigned long long BFoffset;
typedef uint64_t BFoffset;
//typedef unsigned char      BFoffset; // HACK TESTING correct offset wrapping
typedef   signed long long BFdelta;
enum {
	BF_SPACE_AUTO         = 0,
	BF_SPACE_SYSTEM       = 1, // aligned_alloc
	BF_SPACE_CUDA         = 2, // cudaMalloc
	BF_SPACE_CUDA_HOST    = 3, // cudaHostAlloc
	BF_SPACE_CUDA_MANAGED = 4  // cudaMallocManaged
};

typedef BFenum BFspace;
/// Defines a single atom of data to be passed to a function.
typedef struct BFarray_ {
    /*! The data pointer can point towards any type of data, 
     *  so long as there is a corresponding definition in dtype. 
     *  This data should be an ndim array, which every element of
     *  type dtype.
     */
    void* data;
    /*! Where this data is located in memory.
     *  Used to ensure that operations called are localized within
     *  that space, such as a CUDA funciton operating on device
     *  memory.
     */
    BFspace space;
    unsigned dtype;
    int ndim;
    BFsize shape[BF_MAX_DIM];
    BFsize strides[BF_MAX_DIM];
} BFarray;

/// Defines a single atom of data to be passed to a function.
typedef struct BFconstarray_ {
    /*! The data pointer can point towards any type of data, 
     *  so long as there is a corresponding definition in dtype. 
     *  This data should be an ndim array, which every element of
     *  type dtype.
     */
    const void* data;
    /*! Where this data is located in memory.
     *  Used to ensure that operations called are localized within
     *  that space, such as a CUDA funciton operating on device
     *  memory.
     */
    BFspace space;
    unsigned dtype;
    int ndim;
    BFsize shape[BF_MAX_DIM];
    BFsize strides[BF_MAX_DIM];
} BFconstarray;

enum {
	BF_STATUS_SUCCESS            = 0,
	BF_STATUS_END_OF_DATA        = 1,
	BF_STATUS_INVALID_POINTER    = 2,
	BF_STATUS_INVALID_HANDLE     = 3,
	BF_STATUS_INVALID_ARGUMENT   = 4,
	BF_STATUS_INVALID_STATE      = 5,
	BF_STATUS_MEM_ALLOC_FAILED   = 6,
	BF_STATUS_MEM_OP_FAILED      = 7,
	BF_STATUS_UNSUPPORTED        = 8,
	BF_STATUS_FAILED_TO_CONVERGE = 9,
	BF_STATUS_INTERNAL_ERROR     = 10,
        BF_STATUS_INVALID_SHAPE      = 11,
        BF_STATUS_INVALID_SPACE      = 12
};

// Utility
const char* bfGetStatusString(BFstatus status);
BFbool      bfGetDebugEnabled();
BFbool      bfGetCudaEnabled();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BF_COMMON_H_INCLUDE_GUARD_
