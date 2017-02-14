#pragma once

#include "transform.h"

// this is interface and public part declaring common transforms


namespace nd{

	//------ arithmetics -------
	template<typename type> struct add_2r{
		HOSTDEVICE void operator()(type & a, const type & b, const type & c)const{
			a = b + c;
		}
	};

	template<typename type, int dims> void
	add(ndarray_ref<type,dims> & a, const ndarray_ref<type,dims> & b, const ndarray_ref<type,dims> & c){
		transform(add_2r<type>(), a, b, c);
	}

	template<typename type>
	HOSTDEVICE void add_f(type & a, const type & b){
		a += b;
	}

	template<typename type> struct add_2{
		HOSTDEVICE void operator()(type & a, const type & b)const{
			a += b;
		}
	};

	template<typename type> struct scalar_ops{
		static HOSTDEVICE void add(type & a, const type & b){
			a += b;
		}
	};

	//HOSTDEVICE void foo(int, int);

	template<typename type, int dims>
	void add(ndarray_ref<type,dims> & a, const ndarray_ref<type,dims> & b){
		//transform(add_2<type>(), a, b);
		auto ff = make_functor1<decltype(add_f<type>), add_f >();
		//auto ff = make_functor<decltype(foo), foo >();
		transform(ff, a, b);
		//transform([]__device__(type & a, const type & b)-> void { a+=b;}, a, b);
		//transform(scalar_ops<type>::add, a, b);
	}

	// functors
	template<typename type> struct add_const_f{
		type w;
		void HOSTDEVICE operator ()(type & a)const{
			a += w;
		}
	};

	template<typename type, int dims> void add(ndarray_ref<type,dims> & a, type w){
		transform(add_const_f<type>{w}, a);
	}

	template<typename type> struct madd2_f{
		type w1, w2;
		void HOSTDEVICE operator ()(type & a, const type & b, const type & c)const{
			a = b*w1 + c*w2;
		}
	};

	template<typename type, int dims>
	void madd2(ndarray_ref<type,dims> & a, const ndarray_ref<type,dims> & b, const ndarray_ref<type,dims> & c, type w1, type w2){
		transform(madd2_f<type>{w1,w2}, a, b, c);
	}


	// --- functions recieving full arrays (threads mapping optimization possible)


	template<typename type, int dims>
	void HOSTDEVICE grad_f(const intn<dims+1> & ii, kernel::ndarray_ref<type, dims+1> & a, const kernel::ndarray_ref<type,dims> & b){
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

	//! gradient: for input (b) of size [W x H] the output (a) is of size [W x H x 2], and similar for higher dimensions
	/*
	 * dims - output dimensions
	 */
	template<typename type, int dims>
	void grad(ndarray_ref<type, dims+1> & a, const ndarray_ref<type,dims> & b){
		//test_transform(a, b);
		transform(grad_f, a, b);
	}

	//------ conversions-------------

	template<typename T1, typename T2>
	void HOSTDEVICE convert_f(T1 & a, const T2 & b){
		a = T1(b);
	}

	template<typename T1, typename T2, int dims>
	void convert(ndarray_ref<T1, dims> & a, const ndarray_ref<T2, dims>& b){
		transform(convert_f, a, b);
	}

	/*
	// functions recieving pointers
	template<typename type, int dims> struct grad{
		tstride<char> di[dims];
		intn<dims> size;
		void HOSTDEVICE operator () (const intn<dims> & ii, type * a, const type * b){ // b is zero stride extended
			for(int dim=0; dim < dims-1; ++dim){
				if(ii[dim] >= size[dim]-1 ) return;
			};
			*a = *( b + di[ ii[dims-1] ] ) - *b;
		}
	}

	//! gradient: for input (b) of size [W x H] the output (a) is of size [W x H x 2], and similar for higher dimensions
	template<typename type, int dims> void
	grad(const ndarray_ref<type, dims> & a, const ndarray_ref<type,dims - 1> & b){
		for(int dim = 0; dim < dims-1; ++dim){
			runtime_check(a.size(dim) == b.size(dim));
		};
		runtime_check(a.size(dims-1) == dims);
		transform(grad, a, b);
	}
	 */
};

void test_transform();
