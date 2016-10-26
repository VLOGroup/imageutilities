#include <cuda.h>
#include <device_functions.h>
#include "ndarray_op.cuh"
#include "error_cuda.h"

namespace device_op{

	//template<typename type, int dims, typename Func>
	//void for_each_device(const ndarray_ref<type, dims> & r, Func func){
	//	// launch
	//	error_text("Kernel launch not implemented");
	//};

	// a+=val
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> & a, const type val){
		auto func = [=] __device__(const intn<dims> & ii){ a.kernel()(ii) += val; }; // operation capture
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}

	// a*=val
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator *= (ndarray_ref<type, dims> & a, const type val){
		auto func = [=] __device__(const intn<dims> & ii){ a.kernel()(ii) *= val; }; // operation capture
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}

	// a+=b
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
		auto func = [=] __device__(const intn<dims> & ii){ a.kernel()(ii) += b.kernel()(ii); }; // operation capture
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}

	// a-=b
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator -= (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
		auto func = [=] __device__(const intn<dims> & ii){ a.kernel()(ii) -= b.kernel()(ii); }; // operation capture
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}

	// a*=b
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator *= (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
		auto func = [=] __device__(const intn<dims> & ii){ a.kernel()(ii) *= b.kernel()(ii); }; // operation capture
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}

	template<typename type, int dims>
	ndarray_ref<type, dims> & operator << (ndarray_ref<type, dims> & a, const type val){
		//ckeck_allowed(a);
		auto func = [=] __device__(const intn<dims> & ii){ a.kernel()(ii) = val; }; // operation capture
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}

	template<typename type, int dims>
	ndarray_ref<type, dims> & copy_data(ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
		//ckeck_allowed(a);
		auto func = [=] __device__(const intn<dims> & ii){ a.kernel()(ii) = b.kernel()(ii); }; // operation capture
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	};

	//! converting operator <<
	template<typename type1, typename type2, int dims>
	ndarray_ref<type1, dims> & operator << (ndarray_ref<type1, dims> & a, const ndarray_ref<type2, dims> & b){
		auto func = [=] __device__(const intn<dims> & ii){ a.kernel()(ii) = type1(b.kernel()(ii)); }; // operation capture
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	};

	template<typename type, int dims>
	ndarray_ref<type, dims> & madd2(ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b, const ndarray_ref<type, dims> & c, type w1, type w2){
		auto func = [=] __device__(const intn<dims> & ii){ a.kernel()(ii) = b.kernel()(ii)*w1 + c.kernel()(ii)*w2; }; // operation capture
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}


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

	//template NDARRAY_EXPORT ndarray_ref<float, 3>& operator += (ndarray_ref<float, 3> & a, const ndarray_ref<float, 3> & b);
	//template NDARRAY_EXPORT ndarray_ref<int, 4>&  copy_data<int, 4>(ndarray_ref<int, 4>&, const ndarray_ref<int, 4> &);

};

namespace A{

	template<typename type, int dims>
	void ttest(){
		ndarray_ref<type, dims> a, b;
		device_op::operator *= (a, type(1));
		device_op::operator += (a, type(1));
		device_op::operator << (a, type(1));
		device_op::operator += (a, a);
		device_op::operator -= (a, a);
		device_op::operator *= (a, a);
		device_op::copy_data(a, b);
		device_op::madd2(a, a, a, type(1), type(1));
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

		device_op::operator << (f, f);
		device_op::operator << (i, i);
		device_op::operator << (f, i);
		device_op::operator << (i, f);
		device_op::operator << (d, d);
		device_op::operator << (f, d);
		device_op::operator << (d, f);
		device_op::operator << (ui, ui);
		device_op::operator << (si, si);
	}

	void test_ops(){
		tdtest<int>();
		tdtest<float>();
		tdtest<double>();
		tdtest<unsigned int>();
		tdtest<short int>();
		// conversions
		dtest<1>();
		dtest<2>();
		dtest<3>();
		dtest<4>();
	}
};