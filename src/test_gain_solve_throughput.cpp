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

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <bifrost/correlate.h>
#define NCHAN 1
#define NSTAND 256
#define NPOL 2

int main() {
    BFspace space = BF_SPACE_CUDA;
    //BFcomplex64 test_vis[NCHAN][NSTAND][NPOL][NSTAND][NPOL] = {0};
    //BFcomplex64 test_jones[NCHAN][NPOL][NSTAND][NPOL] = {0};
    //char flags_values[NCHAN][NSTAND] = {0};
    BFcomplex64 *test_vis;
    BFcomplex64 *test_model;
    BFcomplex64 *test_jones;
    char *flag_values;
    cudaMalloc((void**)&test_vis, NCHAN*NSTAND*NPOL*NSTAND*NPOL*sizeof(BFcomplex64));
    cudaMemset((void**)&test_vis, 1, NCHAN*NSTAND*NPOL*NSTAND*NPOL*sizeof(BFcomplex64));
    cudaMalloc((void**)&test_model, NCHAN*NSTAND*NPOL*NSTAND*NPOL*sizeof(BFcomplex64));
    cudaMemset((void**)&test_model, 2, NCHAN*NSTAND*NPOL*NSTAND*NPOL*sizeof(BFcomplex64));
    cudaMalloc((void**)&test_jones, NCHAN*NPOL*NSTAND*NPOL*sizeof(BFcomplex64));
    cudaMemset((void**)&test_jones, 1, NCHAN*NPOL*NSTAND*NPOL*sizeof(BFcomplex64));
    cudaMalloc((void**)&flag_values, NCHAN*NSTAND*sizeof(char));
    cudaMemset((void**)&flag_values, 0, NCHAN*NSTAND*sizeof(char));

    BFconstarray V, M;
    V.data = (void*)test_vis;
    M.data = (void*)test_model;
    V.ndim = 5;
    M.ndim = 5;
    V.space = space;
    M.space = space;
    V.shape[0] = NCHAN;
    V.shape[1] = NSTAND;
    V.shape[2] = NPOL;
    V.shape[3] = NSTAND;
    V.shape[4] = NPOL;
    M.shape[0] = NCHAN;
    M.shape[1] = NSTAND;
    M.shape[2] = NPOL;
    M.shape[3] = NSTAND;
    M.shape[4] = NPOL;


    BFarray G;
    G.ndim = 4;
    G.space = space;
    G.data = (void*)test_jones;
    G.shape[0] = NCHAN;
    G.shape[1] = NPOL;
    G.shape[2] = NSTAND;
    G.shape[3] = NPOL;

    BFarray flags;
    flags.ndim = 2;
    flags.space = space;
    flags.data = (void*)flag_values;
    flags.shape[0] = NCHAN;
    flags.shape[1] = NSTAND;
    BFbool l1norm = 1;
    float l2reg = 1;
    float eps = 1;
    int maxiter = 200000;
    int num_unconverged_ptr;
    bfSolveGains(V, M, G, flags, l1norm, l2reg, eps, maxiter, &num_unconverged_ptr);
    //bfVisibilitiesFillHermitian(nchan, ninput, space, data, stride);
    return 0;
}
