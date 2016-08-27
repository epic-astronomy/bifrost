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

// TODO: Generalise this to float or double via templates

#pragma once

#include "Complex.hpp"

#include <limits>

//using std::max;
//using std::fabs;
/*
// Storage-only formats
struct __attribute__((aligned(4))) JonesVec8 {
	signed char x, y, z, w;
};
struct __attribute__((aligned(8))) JonesVec16 {
	signed short x, y, z, w;
};
*/
struct JonesMat;
struct JonesHMat;

template<typename R, bool IsCol>
struct __attribute__((aligned(16))) JonesVec_ {};

template<typename R, bool IsCol>
struct __attribute__((aligned(16))) JonesVecBase {
	//typedef T Real;
	typedef R                           real_type;
	typedef Complex<real_type>          complex_type;
	typedef JonesVec_<real_type, IsCol> this_type;
	typedef JonesVec_<real_type,!IsCol> thisT_type;
	//typedef Complex<T> complex_type;
	complex_type x, y;
	inline __host__ __device__ JonesVecBase() {}//: x(), y() {}
	inline __host__ __device__ JonesVecBase(complex_type x_, complex_type y_) : x(x_), y(y_) {}
	//inline __host__ __device__ this_type& operator+=(this_type v)    { x += v.x; y += v.y; return *this; }
	//inline __host__ __device__ this_type& operator-=(this_type v)    { x -= v.x; y -= v.y; return *this; }
	//inline __host__ __device__ this_type& operator*=(complex_type c) { x *= c; y *= c; return *this; }
	//inline __host__ __device__ this_type& operator/=(complex_type c) { x /= c; y /= c; return *this; }
	inline __host__ __device__ this_type  operator-() const { return this_type(-x,-y); }
	inline __host__ __device__ this_type  operator+() const { return this_type(+x,+y); }
	inline __host__ __device__ thisT_type transpose() const { return thisT_type(x,y); }
	inline __host__ __device__ this_type  conj()      const { return this_type(x.conj(),y.conj()); }
	inline __host__ __device__ thisT_type adjoint()   const { return this->transpose().conj(); }
	inline __host__ __device__ thisT_type T()         const { return this->transpose(); }
	inline __host__ __device__ thisT_type H()         const { return this->adjoint(); }
	inline __host__ __device__ real_type  mag2()      const { real_type a = x.mag2(); a += y.mag2(); return a; }
	//inline __host__ __device__ JonesHMat         autocorrelation() const;
};
template<typename T>
struct __attribute__((aligned(16))) JonesVec_<T,true> : public JonesVecBase<T,true> {
	typedef T                  real_type;
	typedef Complex<real_type> complex_type;
	inline __host__ __device__ JonesVec_() : JonesVecBase<T,true>() {}
	inline __host__ __device__ JonesVec_(complex_type x_, complex_type y_) : JonesVecBase<T,true>(x_, y_) {}
#ifdef __CUDACC__
	// Note: Use float4 to ensure vectorized load/store
	inline __host__ __device__ JonesVec_(float4 v) : JonesVecBase<T,true>(complex_type(v.x,v.y), complex_type(v.z,v.w)) {}
	inline __host__ __device__ operator float4() const { return make_float4(this->x.x,this->x.y,this->y.x,this->y.y); }
#endif
	//inline __host__ __device__ explicit JonesVec_(JonesVec8  v) : x(Complex(v.x,v.y)), y(Complex(v.z,v.w)) {}
	//inline __host__ __device__ explicit JonesVec_(JonesVec16 v) : x(Complex(v.x,v.y)), y(Complex(v.z,v.w)) {}
	inline __host__ __device__ JonesVec_& operator+=(JonesVec_ v)    { this->x += v.x; this->y += v.y; return *this; }
	inline __host__ __device__ JonesVec_& operator-=(JonesVec_ v)    { this->x -= v.x; this->y -= v.y; return *this; }
	inline __host__ __device__ JonesVec_& operator*=(complex_type c) { this->x *= c; this->y *= c; return *this; }
	inline __host__ __device__ JonesVec_& operator/=(complex_type c) { this->x /= c; this->y /= c; return *this; }
	inline __host__ __device__ JonesVec_& operator*=(JonesMat m);
	inline __host__ __device__ JonesVec_& operator/=(JonesMat m);
	inline __host__ __device__ JonesVec_& mad( JonesMat m, JonesVec_ v);
	inline __host__ __device__ JonesVec_& msub(JonesMat m, JonesVec_ v);
};
template<typename T>
struct __attribute__((aligned(16))) JonesVec_<T,false> : public JonesVecBase<T,false> {
	typedef T                  real_type;
	typedef Complex<real_type> complex_type;
	inline __host__ __device__ JonesVec_() : JonesVecBase<T,false>() {}
	inline __host__ __device__ JonesVec_(complex_type x_, complex_type y_) : JonesVecBase<T,false>(x_, y_) {}
#ifdef __CUDACC__
	// Note: Use float4 to ensure vectorized load/store
	inline __host__ __device__ JonesVec_(float4 v) : JonesVecBase<T,false>(complex_type(v.x,v.y), complex_type(v.z,v.w)) {}
	inline __host__ __device__ operator float4() const { return make_float4(this->x.x,this->x.y,this->y.x,this->y.y); }
#endif
	//inline __host__ __device__ explicit JonesVec_(JonesVec8  v) : x(Complex(v.x,v.y)), y(Complex(v.z,v.w)) {}
	//inline __host__ __device__ explicit JonesVec_(JonesVec16 v) : x(Complex(v.x,v.y)), y(Complex(v.z,v.w)) {}
	inline __host__ __device__ JonesVec_& operator+=(JonesVec_ v)    { this->x += v.x; this->y += v.y; return *this; }
	inline __host__ __device__ JonesVec_& operator-=(JonesVec_ v)    { this->x -= v.x; this->y -= v.y; return *this; }
	inline __host__ __device__ JonesVec_& operator*=(complex_type c) { this->x *= c; this->y *= c; return *this; }
	inline __host__ __device__ JonesVec_& operator/=(complex_type c) { this->x /= c; this->y /= c; return *this; }
	inline __host__ __device__ JonesVec_& operator*=(JonesMat m);
	inline __host__ __device__ JonesVec_& operator/=(JonesMat m);
	inline __host__ __device__ JonesVec_& mad( JonesVec_ v, JonesMat m);
	inline __host__ __device__ JonesVec_& msub(JonesVec_ v, JonesMat m);
};
// TODO: Should allow template type somehow (without C++11?)
typedef JonesVec_<float,true>  JonesColVec;
typedef JonesVec_<float,false> JonesRowVec;
typedef JonesColVec            JonesVec;

struct __attribute__((aligned(32))) JonesMat {
	typedef float              real_type;
	typedef Complex<real_type> complex_type;
	JonesRowVec x, y;
	inline __host__ __device__ JonesMat() {}
	//: x(0,0),
	//  y(0,0) {}
	inline __host__ __device__ explicit JonesMat(complex_type c)
		: x(c,0),
		  y(0,c) {}
	inline __host__ __device__ JonesMat(complex_type xx, complex_type yy)
		: x(xx, 0),
		  y(0, yy) {}
	inline __host__ __device__ JonesMat(complex_type xx, complex_type xy,
	                                    complex_type yx, complex_type yy)
		: x(xx,xy),
		  y(yx,yy) {}
	inline __host__ __device__ JonesMat(JonesRowVec x_,
	                                    JonesRowVec y_)
		: x(x_),
		  y(y_) {}
	inline __host__ __device__ JonesMat  operator-() const { return JonesMat(-x, -y); }
	inline __host__ __device__ JonesMat  operator+() const { return JonesMat(+x, +y); }
	inline __host__ __device__ JonesMat& operator+=(JonesMat m) { x += m.x; y += m.y; return *this; }
	inline __host__ __device__ JonesMat& operator-=(JonesMat m) { x -= m.x; y -= m.y; return *this; }
	inline __host__ __device__ JonesMat& operator*=(complex_type c) { x *= c; y *= c; return *this; }
	inline __host__ __device__ JonesMat& operator/=(complex_type c) { x /= c; y /= c; return *this; }
	// Pre-multiply by m
	inline __host__ __device__ JonesMat& operator*=(JonesMat m) { JonesMat tmp(0); tmp.mad(m, *this); return *this = tmp; }
	// Pre-multiply by m.inverse()
	inline __host__ __device__ JonesMat& operator/=(JonesMat m) { return *this *= m.inverse(); }
	// Matrix multiply-add
	inline __host__ __device__ JonesMat& mad(JonesMat a, JonesMat b) {
		x.mad(a.x, b);
		y.mad(a.y, b);
		return *this;
	}
	inline __host__ __device__ JonesMat& msub(JonesMat a, JonesMat b) {
		x.msub(a.x, b);
		y.msub(a.y, b);
		return *this;
	}
	// Add outer product of two vectors
	inline __host__ __device__ JonesMat& mad(JonesColVec a, JonesRowVec b) {
		x.x.mad(a.x, b.x);
		x.y.mad(a.x, b.y);
		y.x.mad(a.y, b.x);
		y.y.mad(a.y, b.y);
		return *this;
	}
	inline __host__ __device__ JonesMat& msub(JonesColVec a, JonesRowVec b) {
		x.x.msub(a.x, b.x);
		x.y.msub(a.x, b.y);
		y.x.msub(a.y, b.x);
		y.y.msub(a.y, b.y);
		return *this;
	}
	inline __host__ __device__ JonesMat inverse() const {
		complex_type d = this->det();
		JonesMat result( y.y, -x.y,
		                -y.x,  x.x);
		result /= d;
		return result;
	}
	inline __host__ __device__ JonesMat conj()      const { return JonesMat(x.conj(), y.conj()); }
	inline __host__ __device__ JonesMat transpose() const { return JonesMat(x.x, y.x, x.y, y.y); }
	inline __host__ __device__ JonesMat adjoint()   const { return this->transpose().conj(); }
	inline __host__ __device__ JonesMat T()         const { return this->transpose(); }
	inline __host__ __device__ JonesMat H()         const { return this->adjoint(); }
	inline __host__ __device__ real_type    mag2()  const { real_type a = x.mag2(); a += y.mag2(); return a; }
	inline __host__ __device__ complex_type det()   const { return (x.x*y.y).msub(x.y,y.x); }
	inline __host__ __device__ bool is_singular(real_type eps=10*std::numeric_limits<real_type>::epsilon()) const {
		real_type fnorm = this->mag2();
		real_type inv_condition = fnorm*fnorm/this->det().mag2();
		// TODO: Does eps need to be squared?
		return inv_condition <= eps;
	}
};
inline __host__ __device__
JonesMat operator+(JonesMat const& a, JonesMat const& b) { JonesMat c = a; return c += b; }
inline __host__ __device__
JonesMat operator-(JonesMat const& a, JonesMat const& b) { JonesMat c = a; return c -= b; }
inline __host__ __device__
JonesMat operator*(JonesMat const& a, JonesMat const& b) { JonesMat c = b; return c *= a; }
inline __host__ __device__
JonesMat operator/(JonesMat const& a, JonesMat const& b) { return a * b.inverse(); }
inline __host__ __device__
JonesMat operator*(JonesMat const& a, JonesMat::complex_type const& b) { JonesMat c = a; return c *= b; }
inline __host__ __device__
JonesMat operator*(JonesMat::complex_type const& b, JonesMat const& a) { return a*b; }

struct __attribute__((aligned(16))) JonesHMat {
	typedef float              real_type;
	typedef Complex<real_type> complex_type;
	real_type    xx, yy;
	complex_type xy;
	inline __host__ __device__ JonesHMat() : xx(0), yy(0), xy(0) {}
	inline __host__ __device__ JonesHMat(real_type xx_, real_type yy_, complex_type xy_=0) : xx(xx_), yy(yy_), xy(xy_) {}
	inline __host__ __device__ JonesHMat& operator+=(JonesHMat m) { xx += m.xx; yy += m.yy; xy += m.xy; return *this; }
	inline __host__ __device__ JonesHMat& operator-=(JonesHMat m) { xx -= m.xx; yy -= m.yy; xy -= m.xy; return *this; }
	inline __host__ __device__ JonesHMat& operator*=(real_type r) { xx *= r; yy *= r; xy *= r; return *this; }
	inline __host__ __device__ JonesHMat& operator/=(real_type r) { xx /= r; yy /= r; xy /= r; return *this; }
	// Add outer product of vector with itself (autocorrelation)
	inline __host__ __device__ JonesHMat& mad(JonesColVec a) {
		xx += a.x.mag2();
		yy += a.y.mag2();
		xy.mad(a.x, a.y);
		return *this;
	}
	inline __host__ __device__ JonesHMat& msub(JonesColVec a) {
		xx -= a.x.mag2();
		yy -= a.y.mag2();
		xy.msub(a.x, a.y);
		return *this;
	}
	inline __host__ __device__ JonesHMat conj()      const { return JonesHMat(xx, yy, xy.conj()); }
	inline __host__ __device__ JonesHMat transpose() const { return this->conj(); }
	inline __host__ __device__ JonesHMat adjoint()   const { return *this; }
	inline __host__ __device__ JonesHMat T()         const { return this->transpose(); }
	inline __host__ __device__ JonesHMat H()         const { return this->adjoint(); }
	inline __host__ __device__ operator JonesMat() const { return JonesMat(xx, xy, xy.conj(), yy); }
};
inline __host__ __device__
JonesHMat operator+(JonesHMat const& a, JonesHMat const& b) { JonesHMat c = a; return c += b; }
inline __host__ __device__
JonesHMat operator-(JonesHMat const& a, JonesHMat const& b) { JonesHMat c = a; return c -= b; }

//template<bool IsCol>
//inline __host__ __device__ JonesHMat JonesVecBase<IsCol>::autocorrelation() const { return JonesHMat(x.mag2(), y.mag2(), x*y.conj()); }

template<typename R>
inline __host__ __device__ JonesVec_<R,true>& JonesVec_<R,true>::operator*=(JonesMat m) {
	JonesVec_ tmp(0, 0);
	tmp.mad(m, *this);
	return *this = tmp;
}
template<typename R>
inline __host__ __device__ JonesVec_<R,true>& JonesVec_<R,true>::operator/=(JonesMat m) {
	JonesVec_ tmp(0, 0);
	tmp.mad(m.inverse(), *this);
	return *this = tmp;
}
template<typename R>
inline __host__ __device__ JonesVec_<R,true>& JonesVec_<R,true>::mad(JonesMat m, JonesVec_ v) {
	this->x.mad(m.x.x, v.x);
	this->x.mad(m.x.y, v.y);
	this->y.mad(m.y.x, v.x);
	this->y.mad(m.y.y, v.y);
	return *this;
}
template<typename R>
inline __host__ __device__ JonesVec_<R,true>& JonesVec_<R,true>::msub(JonesMat m, JonesVec_ v) {
	this->x.msub(m.x.x, v.x);
	this->x.msub(m.x.y, v.y);
	this->y.msub(m.y.x, v.x);
	this->y.msub(m.y.y, v.y);
	return *this;
}

template<typename R>
inline __host__ __device__ JonesVec_<R,false>& JonesVec_<R,false>::operator*=(JonesMat m) {
	JonesVec_ tmp(0, 0);
	tmp.mad(*this, m);
	return *this = tmp;
}
template<typename R>
inline __host__ __device__ JonesVec_<R,false>& JonesVec_<R,false>::operator/=(JonesMat m) {
	JonesVec_ tmp(0, 0);
	tmp.mad(*this, m.inverse());
	return *this = tmp;
}
template<typename R>
inline __host__ __device__ JonesVec_<R,false>& JonesVec_<R,false>::mad(JonesVec_ v, JonesMat m) {
	this->x.mad(v.x, m.x.x);
	this->x.mad(v.y, m.y.x);
	this->y.mad(v.x, m.x.y);
	this->y.mad(v.y, m.y.y);
	return *this;
}
template<typename R>
inline __host__ __device__ JonesVec_<R,false>& JonesVec_<R,false>::msub(JonesVec_ v, JonesMat m) {
	this->x.msub(v.x, m.x.x);
	this->x.msub(v.y, m.y.x);
	this->y.msub(v.x, m.x.y);
	this->y.msub(v.y, m.y.y);
	return *this;
}

template<typename R, bool IsCol>
inline __host__ __device__ JonesVec_<R,IsCol> operator+(JonesVec_<R,IsCol> const& a, JonesVec_<R,IsCol> const& b) { JonesVec_<R,IsCol> c = a; return c += b; }
template<typename R, bool IsCol>
inline __host__ __device__ JonesVec_<R,IsCol> operator-(JonesVec_<R,IsCol> const& a, JonesVec_<R,IsCol> const& b) { JonesVec_<R,IsCol> c = a; return c -= b; }
template<typename R>
inline __host__ __device__ JonesVec_<R,true> operator*(JonesMat const& m, JonesVec_<R,true> const& v) { JonesVec_<R,true> c = v; return c *= m; }
template<typename R>
inline __host__ __device__ JonesVec_<R,false> operator*(JonesVec_<R,false> const& v, JonesMat const& m) { JonesVec_<R,false> c = v; return c *= m; }
/*
inline std::ostream& operator<<(std::ostream& stream, JonesVec const& v) {
	stream << "J(" << v.x << "; " << v.y << ")";
	return stream;
}
inline std::ostream& operator<<(std::ostream& stream, JonesMat const& m) {
	stream << "M("
	       << m.x.x << "; " << m.x.y << " | "
	       << m.y.x << "; " << m.y.y << ")";
	return stream;
}
inline std::ostream& operator<<(std::ostream& stream, JonesHMat const& h) {
	stream << "H("
	       << h.xx  << "; " << h.xy << " | "
	       << "---" << "; " << h.yy << ")";
	return stream;
}
*/
