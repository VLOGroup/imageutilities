#pragma once

#include "ndarray_ref.host.h"
#include "error.h"
#include <typeinfo>

/*
template<typename type, int dims, typename target, typename base> class ndarray_ref_host : public base{
public:
	typedef const intn<dims> & tidx;
public:
	target & operator += (const base & b){ // shorter, no need of rvalue ref version
		host_ops_allowed();
		for_each([&] (tidx ii){ (*this)(ii) += b(ii); });
		return self();
	};
};
 */
/*
#define binary_operator_move(a,b,op,capture)\
		runtime_check(operations_allowed(a) && operations_allowed(b));\
		runtime_check(a.size() == b.size());\
		auto func = capture (const intn<dims> & ii){ op; };\
		struct_dims<dims>::for_each(a.shape(), func);\
		return a;
 */

//________________________________host operations______________________________________________
namespace host_op{

	//! whether host operations allowed on a

	/*
template<typename type, int dims>
bool operations_allowed(const ndarray_ref<type, dims> & a){
	return a.host_allowed();
};
	 */

	template<typename type, int dims> bool check_data(const ndarray_ref<type, dims> & a){
		type * p = a.begin();
		// try accessing some data
		type c = *p;
		tstride<char> offset = 0;
		for(int i=0; i< dims; ++i){
			p += (a.size(i)-1) * a.stride(i);
			c = c + *p;
		};
		return *(char*)&c!=0;
	}

	template<typename type, int dims>
	void inline check_allowed(const ndarray_ref<type, dims> & a){
		runtime_check(a.host_allowed());
	}

	template<int dims> struct struct_dims{
		template<typename Func>
		static void for_each(const shapen<dims> & s, Func func){
			// TODO: reorder by descending strides, optimize 1-2 innermost dimensions
			intn<dims> ii;
			for (ii = 0;;){
				// check not exceeding ranges
				if (ii[dims-1] >= s.size(dims-1)) break;
				// perform operation
				func(ii);
				// increment iterator
				for (int d = 0; d < dims; ++d){
					if (++ii[d] >= s.size(d) && d < dims-1 ){
						ii[d] = 0;
					} else break;
				};
			};
			/*
		for (ii = 0;;){
			// check not exceeding ranges
			if (ii[0] >= s.size(0)) break;
			// perform operation
			func(ii);
			// increment iterator
			for (int d = dims - 1; d >= 0; --d){
				if (++ii[d] >= s.size(d)){
					ii[d] = 0;
				} else break;
			};
		};
			 */
		}
	};

	//! operator a+=b
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
		check_allowed(a);
		check_allowed(b);
		runtime_check(a.size() == b.size());
		auto func = [&] (const intn<dims> & ii){ a(ii) += b(ii); };
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}

	//! operator a*=b (component-wise)
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator *= (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
		check_allowed(a);
		check_allowed(b);
		runtime_check(a.size() == b.size());
		auto func = [&] (const intn<dims> & ii){ a(ii) *= b(ii); };
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}

	//! operator a = val
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator << (ndarray_ref<type, dims> & a, type val){
		check_allowed(a);
		auto func = [&] (const intn<dims> & ii){ a(ii) = val; };
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}

	//! operator a+=val
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> & a, type val){
		check_allowed(a);
		auto func = [&] (const intn<dims> & ii){ a(ii) += val; };
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}

	//! operator a*=val
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator *= (ndarray_ref<type, dims> & a, type val){
		check_allowed(a);
		auto func = [&] (const intn<dims> & ii){ a(ii) *= val; };
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}



	//! copy_data(a,b)
	template<typename type, int dims>
	ndarray_ref<type, dims> & copy_data(ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
		check_allowed(a);
		check_allowed(b);
		runtime_check(a.size() == b.size());
		auto func = [&] (const intn<dims> & ii){ a(ii) = b(ii); };
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
		//binary_operator_move(a,b,a(ii)+=b(ii),[&]);
	}

	//! copy_data:  a << b
	template<typename type1, typename type2, int dims>
	ndarray_ref<type1, dims> & operator << (ndarray_ref<type1, dims> & a, const ndarray_ref<type2, dims> & b){
		check_allowed(a);
		check_allowed(b);
		runtime_check(a.size() == b.size());
		auto func = [&] (const intn<dims> & ii){ a(ii) = type1(b(ii)); };
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}

	//!madd2
	template<typename type, int dims>
	ndarray_ref<type, dims> & madd2(ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b, const ndarray_ref<type, dims> & c, type w1, type w2){
		auto func = [=] __device__ (const intn<dims> & ii){ a.kernel()(ii) = b.kernel()(ii)*w1 + c.kernel()(ii)*w2; }; // operation capture
		struct_dims<dims>::for_each(a.shape(), func);
		return a;
	}

};

//________________________________device operations______________________________________________
namespace device_op{

	//! whether device operations allowed on a
	template<typename type, int dims>
	void ckeck_allowed(const ndarray_ref<type, dims> & a){
		runtime_check(a.device_allowed());
	}


	/*
template<typename type, int dims>
bool operations_allowed(const ndarray_ref<type, dims> & a){
	return a.device_allowed();
};
	 */

	//! only forward declarations, implementation in ndarray_op.cu

	template<int dims> struct struct_dims{
	public:
		template<typename Func>
		static inline void for_each(const shapen<dims> & r, Func func); // implementation through a kernel launch in tarry_op.cu
		//
		//		template<typename pack, int N, typename Func>
		//		static inline void transform(Func func, const ndarray_ref<pack,dims> & inputs[N], ndarray_ref<pack,dims> & dest); // implementation through a kernel launch in tarry_op.cu
	};


	//! operator a+=b
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b);

	//! operator a*=b
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator *= (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b);

	//! operator << val
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator << (ndarray_ref<type, dims> & a, const type val);

	//! operator += val
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> & a, const type val);

	//! operator *=
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator *= (ndarray_ref<type, dims> & a, const type val);

	//! copy_data(a,b)
	template<typename type, int dims>
	ndarray_ref<type, dims> & copy_data (ndarray_ref<type, dims> & dest, const ndarray_ref<type, dims> & src);

	//! copy_data a << b
	template<typename type1, typename type2, int dims>
	ndarray_ref<type1, dims> & operator << (ndarray_ref<type1, dims> & dest, const ndarray_ref<type2, dims> & src);


	template<typename type, int dims> bool check_data(const ndarray_ref<type, dims> & a){
		// will have to launch a kernel for that
		device_op::operator +=(const_cast<ndarray_ref<type, dims> &>(a), type(0));
		return true;
	}

	template<typename type, int dims>
	ndarray_ref<type, dims> & madd2(ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b, const ndarray_ref<type, dims> & c, type w1, type w2);


};

//__________________________auto host/device___________________________________________________

template<typename type, int dims> void check_all(const ndarray_ref<type, dims> & a){
	// check alignment
	for(int i=0; i<dims; ++i){
		runtime_check(a.template stride<char>(i).value() % sizeof(type) == 0) << "ndarray_ref<" << typeid(type).name() << "," << dims << ">:" << "a.size=" << a.size() << " a.strides_b=" << a.stride_bytes() << "\n";
	};
	runtime_check(a.ptr() != 0) << a;
	runtime_check(a.host_allowed() || a.device_allowed()) << a;
	if(a.host_allowed()){
#if DEBUG_LVL > 0
		runtime_check(is_ptr_host_accessible(a.ptr())) << a;
		host_op::check_data(a);
#endif
	};
	if(a.device_allowed()){
#if DEBUG_LVL > 0
		runtime_check(is_ptr_device_accessible(a.ptr())) <<a;
		device_op::check_data(a);
#endif
	};
}

void copy_raw_data(void * dest, void * src, size_t n_bytes);

// copy data
template<typename type, int dims> ndarray_ref<type, dims>& copy_data(ndarray_ref<type, dims> & dest, const ndarray_ref<type, dims> & src){
	if (dest.same_shape(src)){
		copy_raw_data(dest.ptr(), src.ptr(), src.size_bytes());
		return dest;
	} else{ //different shapes
		if(dest.size() == src.size()){
			//std::cout << "dest = " << dest << "\n";
			//std::cout << "src = " << src << "\n";
			if(src.device_allowed() && dest.device_allowed()){
				return device_op::copy_data(dest,src);
			}else{
				return host_op::copy_data(dest,src);
			};
		}else{
			throw_error("copy between different sizes not implemented");
		};
	};
}
//! copy_data:  a << b
template<typename type1, typename type2, int dims> ndarray_ref<type1, dims>& operator << (ndarray_ref<type1, dims> & dest, const ndarray_ref<type2, dims> & src){
	if(typeid(type1) == typeid(type2)){
		return copy_data(dest, src.template recast<type1>() );
	};
	if(src.device_allowed() && dest.device_allowed()){
		return device_op:: operator << (dest,src);
	};
	if(src.host_allowed() && dest.host_allowed()){
		return host_op:: operator << (dest,src);
	};
	throw_error("direct assignment not possible\n")<<"src=" <<src <<"\n dest=" << dest << "\n";
}

template<typename type1, typename type2, int dims>
ndarray_ref<type1, dims> & operator << (ndarray_ref<type1, dims> && a, const ndarray_ref<type2, dims> & b){
	return a << b;
}

/*
template<typename type, int dims>
ndarray_ref<type, dims> & operator << (const ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
	return const_cast<ndarray_ref<type, dims> & >(a) << b;
}*/


template<typename type, int dims> ndarray_ref<type, dims> & copy_data(ndarray_ref<type, dims> && dest, const ndarray_ref<type, dims> & src){
	return copy_data(dest,src);
}


// a += b
template<typename type, int dims>
ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
	if(a.device_allowed() && b.device_allowed()){
		return device_op::operator +=(a,b);
	}else{
		return host_op::operator +=(a,b);
	};
}
// a += b
template<typename type, int dims>
ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> && a, const ndarray_ref<type, dims> & b){
	return a+=b;
}

// a *= b
template<typename type, int dims>
ndarray_ref<type, dims> & operator *= (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b){
	if(a.device_allowed() && b.device_allowed()){
		return device_op::operator *=(a,b);
	}else{
		return host_op::operator *=(a,b);
	};
}
// a *= b
template<typename type, int dims>
ndarray_ref<type, dims> & operator *= (ndarray_ref<type, dims> && a, const ndarray_ref<type, dims> & b){
	return a*=b;
}

// a << val
template<typename type, int dims>
ndarray_ref<type, dims> & operator << (ndarray_ref<type, dims> & a, const type val){
	if(a.device_allowed()){
		return device_op::operator <<(a,val);
	}else{
		return host_op::operator <<(a,val);
	};
}

template<typename type, int dims>
ndarray_ref<type, dims> & operator << (ndarray_ref<type, dims> && a, const type val){
	return operator << (a, val);
}
/*
template<typename type, int dims>
ndarray_ref<type, dims> & operator << (const ndarray_ref<type, dims> & a, const type val){
	return operator << (const_cast<ndarray_ref<type, dims> &>(a), val);
}
 */

// a += val
template<typename type, int dims>
ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> & a, type val){
	if(a.device_allowed()){
		return device_op::operator +=(a,val);
	}else{
		return host_op::operator +=(a,val);
	};
}

template<typename type, int dims>
ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> && a, type val){
	return a+=val;
}

// a *= val
template<typename type, int dims>
ndarray_ref<type, dims> & operator *= (ndarray_ref<type, dims> & a, type val){
	if(a.device_allowed()){
		return device_op::operator *=(a,val);
	}else{
		return host_op::operator *=(a,val);
	};
}

template<typename type, int dims>
ndarray_ref<type, dims> & operator *= (ndarray_ref<type, dims> && a, type val){
	return a*=val;
}

// a = b*w1 + c*w2
template<typename type, int dims>
ndarray_ref<type, dims> & madd2(ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b, const ndarray_ref<type, dims> & c, type w1, type w2){
	if(a.device_allowed() && b.device_allowed() && c.device_allowed()){
		device_op::madd2(a,b,c,w1,w2);
	}else{
		host_op::madd2(a,b,c,w1,w2);
	};
	return a;
}

template<typename type, int dims>
ndarray_ref<type, dims> & madd2(ndarray_ref<type, dims> && a, const ndarray_ref<type, dims> & b, const ndarray_ref<type, dims> & c, type w1, type w2){
	return madd2(a,b,c,w1,w2);
}



/*
template<typename type, int dims>
type sum(ndarray_ref<type, dims> & a){
	if(a.device_allowed()){
		return device_op::operator *=(a,val);
	}else{
		return host_op::operator *=(a,val);
	};
};
 */

/*
 //! perform a computation for each entry of an array, automatically determining host /device
template<typename type, int dims, typename Func>
void for_each(const ndarray_ref<type, dims> & r, Func func){
	if (is_on_device(r.begin())){
		struct_dims<dims>::for_each_on_device(r.shape(), func);
	}else{
		struct_dims<dims>::for_each_on_host(r.shape(), func);
	};
}*/

/*

template <int i, int n, typename F> void for_unroll(F & f){
	bool brk = false;
	if(i<n){
		brk = f(i);
	};
	if(!brk){
		for_unroll< (i<n ? i+1: n), n, F>(f);
	};
};

namespace device_op{
	template<typename type, int dims, typename OP,
	typename std::enable_if< sizeof(type) < 16 && 16 % sizeof(type) == 0,int>::type = 0 >
	void binary_op(ndarray_ref<type,dims> a, ndarray_ref<type,dims> b, ndarray_ref<type,dims> d, OP op){
		// vectorize
		//todo: do not vectorize if the base address or one of the strides is not divisible -- will result in missaligned access
		typedef small_array<type, 16 % sizeof(type)> pack;
		binary_op(a.recast<pack>(), b.recast<pack>(), d.recast.type(), op);
	}

	template<typename type, int dims, typename OP,
	typename std::enable_if< !(sizeof(type) < 16 && 16 % sizeof(type) == 0), int>::type = 0 > void binary_op(ndarray_ref<type,dims> a, ndarray_ref<type,dims> b, ndarray_ref<type,dims> d, OP op){
		// already vectorized
		struct_dims<dims>::for_each_v<Func>(a.shape(), op);
	}
};

//
template<typename type, int dims, typename OP> void binary_op(const ndarray_ref<type,dims> & a, const ndarray_ref<type,dims> & b, const ndarray_ref<type,dims> & d, OP op){
	runtime_check(a.size() == b.size());
	runtime_check(a.size() == d.size());
	// reorder
	int i = a.stride_bytes().min_idx();// fastest dim according to input 1 <- output array
	if(i!=0){
		a = a.swap_dims(0,i);
		b = b.swap_dims(0,i);
		d = d.swap_dims(0,i);
	};
	i = b.stride_bytes().min_idx();// fastest dim according to input 2
	bool dim1_used = false;
	if(i > 1 && b.stride_bytes(i) == sizeof(type)){ // contiguous
		a = a.swap_dims(1,i);
		b = b.swap_dims(1,i);
		d = d.swap_dims(1,i);
		dim1_used = true;
	}else{// input2 is ok, chose dim1 for output
		i = d.stride_bytes().min_idx();// fastest dim according to output
		if(i>1){
			a = a.swap_dims(1,i);
			b = b.swap_dims(1,i);
			d = d.swap_dims(1,i);
			dim1_used = true;
		};
	};
	// fix dim0 and dim1 if dim1_used, sort the reminder
	intn<dims> st = a.stride_bytes();
	st[0] = 0; if(dim1_used) st[1] = 0;
	intn<dims> o = st.sort_idx(); // ascending order sorting (stable)
	a = a.permute_dims(o);
	b = b.permute_dims(o);
	d = d.permute_dims(o);
	//
	// flatten dimensions continuous for the whole tuple
	for_unroll<>([&]-> bool (constexpr int i){
		if(a.dims_continuous(i) && b.dims_continuous(i) && d.dims_continuous(i)){
			binary_op<type,dims-1,OP>(a.compress_dim<i>(), b.compress_dim<i>(), c.compress_dim<i>(), op);
			return true;
		};
		return false;
	});
	// dispatch host / device
	if(a.device_allowed() && b.device_allowed()){
		runtime_check(d.device_allowed());
		device_op::binary_op(a,b,dest,op);
	}else{
		runtime_check(a.host_allowed());
		runtime_check(b.host_allowed());
		runtime_check(d.host_allowed());
		host_op::binary_op(a,b,dest,op);
	};
}

// add(a,b, dest)
template<typename type, int dims>
ndarray_ref<type, dims> void add(const ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b, ndarray_ref<type, dims> & dest){
	struct op{
		__host__ __device__ type operator()(type a, type b){
			return a+b;
		}
	};
	binary_op(a,b,dest,op);
}
 */

