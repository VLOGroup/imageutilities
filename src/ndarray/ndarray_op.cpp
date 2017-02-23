#include "ndarray_op.h"
#include <cuda.h>
#include <device_functions.h>
#include "error_cuda.h"

//! copy
void copy_raw_data(void * dest, void * src, size_t n_bytes){
	cudaMemcpy(dest, src, n_bytes, cudaMemcpyDefault);
	cuda_check_error();
}

//! copy data(a,b)
template<typename type, int dims>
ndarray_ref<type, dims>& copy_data(ndarray_ref<type, dims> & dest, const ndarray_ref<type, dims> & src){
	if (dest.same_shape(src)){
		copy_raw_data(dest.ptr(), src.ptr(), src.size_bytes());
		return dest;
	} else{ //different shapes
		if (dest.size() == src.size()){
			//std::cout << "dest = " << dest << "\n";
			//std::cout << "src = " << src << "\n";
			if (src.device_allowed() && dest.device_allowed()){
				return device_op::copy_data(dest, src);
			} else{
				return host_op::copy_data(dest, src);
			};
		} else{
			throw_error("copy between different sizes not implemented") << "\n from: " << src << "\n to: " << dest << "\n";
		};
	};
}

//! a << b
template<typename type1, typename type2, int dims> __NOINLINE__ //NDARRAY_EXPORT
ndarray_ref<type1, dims>& operator << (ndarray_ref<type1, dims> & dest, const ndarray_ref<type2, dims> & src){
	if (typeid(type1) == typeid(type2)){
		return copy_data(dest, src.template recast<type1>());
	};
	if (src.device_allowed() && dest.device_allowed()){
		return device_op:: operator << (dest, src);
	};
	if (src.host_allowed() && dest.host_allowed()){
		return host_op:: operator << (dest, src);
	};
	throw_error("direct assignment not possible\n") << "src=" << src << "\n dest=" << dest << "\n";
}

//! a += b
template<typename type, int dims> __NOINLINE__
ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
	if (a.device_allowed() && b.device_allowed()){
		return device_op::operator +=(a, b);
	} else{
		return host_op::operator +=(a, b);
	};
}

//! a -= b
template<typename type, int dims> __NOINLINE__
ndarray_ref<type, dims> & operator -= (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
	if (a.device_allowed() && b.device_allowed()){
		return device_op::operator -=(a, b);
	} else{
		return host_op::operator -=(a, b);
	};
}

//! a *= b
template<typename type, int dims> __NOINLINE__
ndarray_ref<type, dims> & operator *= (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
	if (a.device_allowed() && b.device_allowed()){
		return device_op::operator *=(a, b);
	} else{
		return host_op::operator *=(a, b);
	};
}

//! a << val
template<typename type, int dims> __NOINLINE__
ndarray_ref<type, dims> & operator << (ndarray_ref<type, dims> & a, const type val){
	if (a.device_allowed()){
		return device_op::operator <<(a, val);
	} else{
		return host_op::operator <<(a, val);
	};
}

//! a += val
template<typename type, int dims> __NOINLINE__
ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> & a, type val){
	if (a.device_allowed()){
		return device_op::operator +=(a, val);
	} else{
		return host_op::operator +=(a, val);
	};
}

//! a *= val
template<typename type, int dims> __NOINLINE__
ndarray_ref<type, dims> & operator *= (ndarray_ref<type, dims> & a, type val){
	if (a.device_allowed()){
		return device_op::operator *=(a, val);
	} else{
		return host_op::operator *=(a, val);
	};
}

//! a = b*w1 + c*w2
template<typename type, int dims> __NOINLINE__
ndarray_ref<type, dims> & madd2(ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b, const ndarray_ref<type, dims> & c, type w1, type w2){
	if (a.device_allowed() && b.device_allowed() && c.device_allowed()){
		device_op::madd2(a, b, c, w1, w2);
	} else{
		host_op::madd2(a, b, c, w1, w2);
	};
	return a;
}


//___tests/instantiation___________

template<typename type, int dims>
void ttest(){
	ndarray_ref<type, dims> a, b;
	a *= type(1);
	a += type(1);
	a << type(1);
	a += a;
	a -= a;
	a *= a;
	copy_data(a, b);
	madd2(a, a, a, type(1), type(1));
}

template<typename type>
void tdtest(){
	ttest<type, 1>();
	ttest<type, 2>();
	ttest<type, 3>();
	ttest<type, 4>();
}

template<int dims>
void dtest(){
	// conversions
	ndarray_ref<float, dims> f;
	ndarray_ref<int, dims> i;
	ndarray_ref<double, dims> d;
	ndarray_ref<unsigned int, dims> ui;
	ndarray_ref<short int, dims> si;
	f << f;
	i << i;
	i << f;
	f << i;
	d << d;
	f << d;
	d << f;
	ui << ui;
	si << si;
}

void test_ops(){
	tdtest<float>();
	tdtest<int>();
	// conversions
	dtest<1>();
	dtest<2>();
	dtest<3>();
	dtest<4>();
}
