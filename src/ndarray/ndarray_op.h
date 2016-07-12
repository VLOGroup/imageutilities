#ifndef ndarray_op_h
#define ndarray_op_h

#include "ndarray_ref.h"
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
		//binary_operator_move(a,b,a(ii)+=b(ii),[&]);
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
}

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
	};


	//! operator a+=b
	template<typename type, int dims>
	ndarray_ref<type, dims> & operator += (ndarray_ref<type, dims> & a, const ndarray_ref<type, dims> & b);

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

}

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
			slperror("copy between different sizes not implemented");
		};
	};
}
//! copy_data:  a << b
template<typename type1, typename type2, int dims> ndarray_ref<type1, dims>& operator << (ndarray_ref<type1, dims> & dest, const ndarray_ref<type2, dims> & src){
	if(src.device_allowed() && dest.device_allowed()){
		return device_op:: operator << (dest,src);
	};
	if(src.host_allowed() && dest.host_allowed()){
		return host_op:: operator << (dest,src);
	};
	slperror("direct assignment not possible")<<"src=" <<src <<"\n dest=" << dest << "\n";
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



#endif
