#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "ndarray_op.h"
#include "ndarray_op.cuh"

#include "error_cuda.h"

//! copy
void copy_raw_data(void * dest, void * src, size_t n_bytes){
	cudaMemcpy(dest, src, n_bytes, cudaMemcpyDefault);
	cuda_check_error();
};

namespace device_op{


//template<typename type, int dims, typename Func>
//void for_each_device(const ndarray_ref<type, dims> & r, Func func){
//	// launch
//	error_text("Kernel launch not implemented");
//};

template<typename type, int dims>
ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> & a, const type val){
	//check_allowed(a);
	auto func = [=] __device__ (const intn<dims> & ii){ a.kernel()(ii) += val; }; // operation capture
	struct_dims<dims>::for_each(a.shape(), func);
	return a;
}

template<typename type, int dims>
ndarray_ref<type, dims> & operator *= (ndarray_ref<type, dims> & a, const type val){
	//ckeck_allowed(a);
	auto func = [=] __device__ (const intn<dims> & ii){ a.kernel()(ii) *= val; }; // operation capture
	struct_dims<dims>::for_each(a.shape(), func);
	return a;
}

template<typename type, int dims>
	ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
	auto func = [=] __device__ (const intn<dims> & ii){ a.kernel()(ii) += b.kernel()(ii); }; // operation capture
	struct_dims<dims>::for_each(a.shape(), func);
	return a;
}

template<typename type, int dims>
ndarray_ref<type, dims> & operator << (ndarray_ref<type, dims> & a, const type val){
	//ckeck_allowed(a);
	auto func = [=] __device__ (const intn<dims> & ii){ a.kernel()(ii) = val; }; // operation capture
	struct_dims<dims>::for_each(a.shape(), func);
	return a;
}

template<typename type, int dims>
ndarray_ref<type, dims> & copy_data (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
	//ckeck_allowed(a);
	auto func = [=] __device__ (const intn<dims> & ii){ a.kernel()(ii) = b.kernel()(ii); }; // operation capture
	struct_dims<dims>::for_each(a.shape(), func);
	return a;
};

//! converting operator <<
template<typename type1, typename type2, int dims>
ndarray_ref<type1, dims> & operator << (ndarray_ref<type1, dims> & a, const ndarray_ref<type2, dims> & b){
	auto func = [=] __device__ (const intn<dims> & ii){ a.kernel()(ii) = type1(b.kernel()(ii)); }; // operation capture
	struct_dims<dims>::for_each(a.shape(), func);
	return a;
};


/*
template<typename type, int dims>
const ndarray_ref<type, dims> & operator *= (const ndarray_ref<type, dims> & a, const type val){
	auto func = [=] (const intn<dims> & ii){ a(ii) *= val; };
	auto func_device = [=] __device__ (const intn<dims> & ii){ a(ii) *= val; };
	//auto func = [=] __device__ (const intn<dims> & ii){ a(ii) *= val; };
	//for_each(a, func);
	struct_dims<dims>::for_each_on_device(a.shape(), func_device);
	return a;
}
*/
//template const ndarray_ref<float, 2> & operator *= (const ndarray_ref<float, 2> & a, const float val);
/*
template ndarray_ref<float, 2> & operator *= (ndarray_ref<float, 2> & a, const float val);
template ndarray_ref<float, 2> & operator << (ndarray_ref<float, 2> & a, const float val);
template ndarray_ref<float, 2> & copy_data (ndarray_ref<float, 2> & a, const ndarray_ref<float, 2> & b);

template ndarray_ref<float, 3> & operator *= (ndarray_ref<float, 3> & a, const float val);
template ndarray_ref<float, 3> & operator << (ndarray_ref<float, 3> & a, const float val);
template ndarray_ref<float, 3> & copy_data (ndarray_ref<float, 3> & a, const ndarray_ref<float, 3> & b);
*/

//template ndarray_ref<float,3> & operator += (ndarray_ref<float,3> & a, const ndarray_ref<float,3> & b);

};

template<typename type, int dims>
void ttest(){
	ndarray_ref<type, dims> a, b;
	a *= type(1);
	a += type(1);
	a << type(1);
	a += a;
	copy_data(a,b);
};

template<typename type>
void tdtest(){
	ttest<type,1>();
	ttest<type,2>();
	ttest<type,3>();
	ttest<type,4>();
};

template<int dims>
void dtest(){
	// conversions
	ndarray_ref<float, dims> a;
	ndarray_ref<int, dims> b;
	a << b;
	b << a;
	a << a;
	b << b;
};

void test(){
	tdtest<float>();
	tdtest<int>();
	// conversions
	dtest<1>();
	dtest<2>();
	dtest<3>();
	dtest<4>();
};
