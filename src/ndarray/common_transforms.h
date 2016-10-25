#pragma once

#include "transform.h"

// this is interface and public part declaring common transforms


namespace nd{

	//------ arithmetics -------
	template<typename type> struct add_a{
		type w;
		__HOSTDEVICE__ void operator()(type & a)const{
			a += w;
		}
	};

	template<typename type, int dims>
	void add(ndarray_ref<type,dims> & a, const type w){
		transform(add_a<type>(w), a);
	}

	template<typename type> struct add_ab{
		__HOSTDEVICE__ void operator()(type & a, const type & b)const{
			a += b;
		}
	};

	template<typename type, int dims>
	void add(ndarray_ref<type,dims> & a, const ndarray_ref<type,dims> & b){
		transform(add_ab<type>(), a, b);
	}

	template<typename type> struct add_abc{
		__HOSTDEVICE__ void operator()(type & a, const type & b, const type & c)const{
			a = b + c;
		}
	};

	template<typename type, int dims>
	void add(ndarray_ref<type,dims> & a, const ndarray_ref<type,dims> & b, const ndarray_ref<type,dims> & c){
		transform(add_abc<type>(), a, b, c);
	}

	template<typename type> struct madd_abc{
		type w;
		void __HOSTDEVICE__ operator ()(type & a, const type & b, const type & c)const{
			a = b*w + c;
		}
	};

	template<typename type, int dims>
	void madd(ndarray_ref<type,dims> & a, const ndarray_ref<type,dims> & b, const ndarray_ref<type,dims> & c, type w){
		transform(madd_abc<type>{w}, a, b, c);
	}

	template<typename type> struct madd2_abc{
		type w1, w2;
		void __HOSTDEVICE__ operator ()(type & a, const type & b, const type & c)const{
			a = b*w1 + c*w2;
		}
	};

	template<typename type, int dims>
	void madd2(ndarray_ref<type,dims> & a, const ndarray_ref<type,dims> & b, const ndarray_ref<type,dims> & c, type w1, type w2){
		transform(madd2_abc<type>{w1,w2}, a, b, c);
	}

	//------ conversions-------------

	template<typename T1, typename T2> struct convert_ab{
		void __HOSTDEVICE__ operator()(T1 & a, const T2 & b)const{
			a = T1(b);
		}
	};

	template<typename T1, typename T2, int dims>
	void convert(ndarray_ref<T1, dims> & a, const ndarray_ref<T2, dims>& b){
		transform(convert_ab<T1,T2>(), a, b);
	}

	// --- functions recieving full arrays (threads mapping optimization possible)
	template<typename type, int dims> struct grad_ab{
		void __HOSTDEVICE__ operator()(const intn<dims+1> & ii, kernel::ndarray_ref<type, dims+1> & a, const kernel::ndarray_ref<type,dims> & b) const {
			// how ii is bound to threads is determined by the kernel
			int dim = ii[ dims ]; // last index component -> gradient direction, prefer unrolled
			if( ii[dim] < a.size(dim)-1 ){ // index is in range for gradient
				intn<dims> i0 = ii.template erase<dims>();
				type v1 = b[i0];
				++i0[dim];
				a[ii] = b[i0] - v1;
			}else{
				a[ii] = 0;
			};
		}
	};

	//! gradient: for input (b) of size [W x H] the output (a) is of size [W x H x 2], and similar for higher dimensions
	/*
	 * dims - output dimensions
	 */
	template<typename type, int dims>
	void grad(ndarray_ref<type, dims+1> & a, const ndarray_ref<type,dims> & b){
		transform(grad_ab<type, dims>(), a, b);
	}
};

// operators
namespace nd{
	template<typename type, int dims> ndarray_ref<type,dims> & operator += (ndarray_ref<type,dims> & a, const ndarray_ref<type,dims> & b){
		transform(madd_abc<type>{1}, a, b, a); // a = b + a
		return a;
	}
	template<typename type, int dims> ndarray_ref<type,dims> & operator -= (ndarray_ref<type,dims> & a, const ndarray_ref<type,dims> & b){
		transform(madd_abc<type>{1}, a, b, a); // a =-b + a
		return a;
	}
};


void test_transform();
