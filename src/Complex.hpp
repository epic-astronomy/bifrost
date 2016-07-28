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

#include <cmath>
//#include <iostream>

using std::atan2;

// Storage-only formats
struct __attribute__((aligned(2))) Complex8 {
	signed char x, y;
};
struct __attribute__((aligned(4))) Complex16 {
	signed short x, y;
};

template<typename Real>
struct Complex;

typedef Complex<float> Complex32;
//typedef Complex<float> Complex;

//inline __host__ __device__
//Complex& mad(Complex& c, Complex a, Complex b) { return c.mad(a, b); }
//inline __host__ __device__
//void msub(Complex& c, Complex a, Complex b) { return c.msub(a, b); }
template<typename R> inline __host__ __device__ Complex<R> operator+(Complex<R> const& a, Complex<R> const& b) { Complex<R> c = a; return c += b; }
template<typename R> inline __host__ __device__ Complex<R> operator-(Complex<R> const& a, Complex<R> const& b) { Complex<R> c = a; return c -= b; }
template<typename R> inline __host__ __device__ Complex<R> operator*(Complex<R> const& a, Complex<R> const& b) { Complex<R> c = a; return c *= b; }
template<typename R> inline __host__ __device__ Complex<R> operator/(Complex<R> const& a, Complex<R> const& b) { Complex<R> c = a; return c /= b; }
template<typename R> inline __host__ __device__ Complex<R> operator*(Complex<R> const& a, R const& b) { Complex<R> c = a; return c *= b; }
template<typename R> inline __host__ __device__ Complex<R> operator*(R const& a, Complex<R> const& b) { Complex<R> c = b; return c *= a; }
template<typename R> inline __host__ __device__ Complex<R> operator/(Complex<R> const& a, R const& b) { Complex<R> c = a; return c /= b; }
//template<typename R> inline std::ostream& operator<<(std::ostream& stream, Complex<R> const& c) {
//	stream << c.x << "," << c.y;
//	return stream;
//}

template<>
struct __attribute__((aligned(8))) Complex<float> {
	// Note: Using unions here may prevent vectorized load/store
	typedef float real_type;
	real_type x, y;
	inline __host__ __device__ Complex() {}//: x(0), y(0) {}
	inline __host__ __device__ Complex(real_type x_, real_type y_=0) : x(x_), y(y_) {}
#ifdef __CUDACC__
	// Note: Use float2 to ensure vectorized load/store
	inline __host__ __device__ Complex(float2 c) : x(c.x), y(c.y) {}
	inline __host__ __device__ operator float2() const { return make_float2(x,y); }
#endif
	inline __host__ __device__ Complex& operator+=(Complex c) { x += c.x; y += c.y; return *this; }
	inline __host__ __device__ Complex& operator-=(Complex c) { x -= c.x; y -= c.y; return *this; }
	inline __host__ __device__ Complex& operator*=(Complex c) {
		Complex tmp;
		tmp.x  = x*c.x;
		tmp.x -= y*c.y;
		tmp.y  = y*c.x;
		tmp.y += x*c.y;
		return *this = tmp;
	}
	inline __host__ __device__ Complex& operator/=(Complex c) {
		return *this *= c.conj() / c.mag2();
	}
	inline __host__ __device__ Complex& operator*=(real_type s) { x *= s; y *= s; return *this; }
	inline __host__ __device__ Complex& operator/=(real_type s) { return *this *= 1/s; }
	inline __host__ __device__ Complex  operator+() const { return Complex(+x,+y); }
	inline __host__ __device__ Complex  operator-() const { return Complex(-x,-y); }
	inline __host__ __device__ Complex conj()  const { return Complex(x, -y); }
	inline __host__ __device__ real_type    phase() const { return atan2(y, x); }
	inline __host__ __device__ real_type    mag2()  const { real_type a = x*x; a += y*y; return a; }
	inline __host__ __device__ real_type    mag()   const { return sqrt(this->mag2()); }
	inline __host__ __device__ real_type    abs()   const { return this->mag(); }
	inline __host__ __device__ Complex& mad(Complex a, Complex b) {
		x += a.x*b.x;
		x -= a.y*b.y;
		y += a.y*b.x;
		y += a.x*b.y;
		return *this;
	}
	inline __host__ __device__ Complex& msub(Complex a, Complex b) {
		x -= a.x*b.x;
		x += a.y*b.y;
		y -= a.y*b.x;
		y -= a.x*b.y;
		return *this;
	}
	inline __host__ __device__ bool operator==(Complex const& c) const { return (x==c.x) && (y==c.y); }
	inline __host__ __device__ bool operator!=(Complex const& c) const { return !(*this == c); }
	inline __host__ __device__ bool isreal_type(real_type tol=1e-6) const {
		return y/x <= tol;
	}
};
