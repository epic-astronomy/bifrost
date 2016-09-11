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


#pragma once

namespace cuda {

template<typename T>
class device_variable {
public:
	typedef T*       pointer;
	typedef T const* const_pointer;
	typedef T        value_type;
private:
	pointer      _devptr;
	pointer      _hstptr;
	cudaStream_t _stream;
	void allocate() {
		cudaMalloc(    (void**)&_devptr, sizeof(value_type));
		cudaMallocHost((void**)&_hstptr, sizeof(value_type));
	}
	void free() {
		cudaFreeHost(_hstptr);
		cudaFree(    _devptr);
	}
	device_variable(device_variable const& );
	device_variable& operator=(device_variable const& );
public:
	device_variable(cudaStream_t stream)
		: _devptr(0), _hstptr(0),_stream(stream) { this->allocate(); }
	device_variable(value_type   val,
	                cudaStream_t stream)
		: _devptr(0), _hstptr(0), _stream(stream) { this->allocate(); *this = val; }
	~device_variable() { this->free(); }
	// TODO: This could be dangerous
	pointer       operator&()       { return _devptr; }
	const_pointer operator&() const { return _devptr; }
	operator value_type const&() const {
		cudaMemcpyAsync(_hstptr, _devptr, sizeof(value_type),
		                cudaMemcpyDeviceToHost, _stream);
		cudaStreamSynchronize(_stream);
		return *_hstptr;
	}
	device_variable& operator=(value_type const& val) {
		//if( val == value_type(0) ) { // TODO: This could be dangerous
		//	cudaMemsetAsync(_devptr, 0, sizeof(value_type), _stream);
		//} else {
		*_hstptr = val;
		cudaMemcpyAsync(_devptr, _hstptr, sizeof(value_type),
		                cudaMemcpyHostToDevice, _stream);
		cudaStreamSynchronize(_stream);
		//}
		return *this;
	}
};

} // namespace cuda
