#pragma once

#include "ndarray_ref.host.h"
#include "ndarray_mem.h"
#include "ndarray_op.h"
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>


inline void eat_delim(std::fstream & ff){
	while (!ff.eof() && (ff.peek() == ',' || ff.peek() == '\n' || ff.peek() == '\r' || ff.peek()==' '))ff.ignore();
};

//! file i/o
template<typename type, int dims> void dlm_read(ndarray<type,dims> & A, std::fstream & ff, ndarray_flags fl){
	//C_total.create<memory::GPU_managed>(K, X, Y);
	//
	//streampos pos = ff.tellg();
	//ff.seekg(pos);
	//for(loop = 0; loop<2; ++loop){
	intn<dims> sz;
	for(int d=0; d<dims; ++d){
		ff >> sz[d];
		eat_delim(ff);
	};
	//std::cout<< "reading size=" << sz << "\n";
	intn<dims> st;
	for(int d=0; d<dims; ++d){
		ff >> st[d];
		eat_delim(ff);
	};
	//std::cout << "reading shape: size " << sz << "stride_bytes: " << st << "\n";
	int access;
	ff >> access;
	if(!fl.access_valid()){
		fl.set_access(access);
	};
	ndarray_ref<type,dims> shape(0,sz,st,fl.access());
	std::cout << "loading array with shape: " << shape << "\n";
	A.clear();
	if(fl.host_allowed() && !fl.device_allowed()){
		A.template create<memory::CPU>(shape);
	}else if(!fl.host_allowed() && fl.device_allowed()){
		A.template create<memory::GPU>(shape);
	}else{
		A.template create<memory::GPU_managed>(shape);
	};
	//std::cout << "created " << A << "\n";
	intn<2> sz2(A.size()[0], A.size().prod() / A.size()[0]); // reshape as 2D, keep dim 0
	//auto A1 = A.reshape(sz2);
	//std::cout<< "A1=" << A1 << "\n";
	for(int i=0; i < sz2[0]; ++i){
		for(int j=0; j< sz2[1]; ++j){
			float val;
			ff >> val;
			//if(i< 4 && j< 4)std::cout << val << ">> A1(" << i <<"," << j << ")\n";
			//A1(i,j) = val;
			intn<dims> ii = A.size().integer_to_index(i + j*sz2[0]);
			A(ii) = val;
			eat_delim(ff);
		};
	};
}

//! file i/o
template<typename type, int dims> void dlm_write(const ndarray_ref<type,dims> & A, std::fstream & ff){
	intn<dims> sz = A.size();
	for(int d=0; d<dims; ++d){
		ff << sz[d];
		if(d<dims-1)ff << ", ";
	};
	ff << "\n";
	intn<dims> st = A.stride_bytes();
	for(int d=0; d<dims; ++d){
		ff << st[d];
		if(d<dims-1)ff << ", ";
	};
	ff << "\n";
	ff << A.access()<<"\n";
	intn<2> sz2(A.size()[0], A.size().prod() / A.size()[0]); // reshape as 2D, keep dim 0
	// array or CPU proxy
	ndarray_ref<type,dims> pA = A;
	ndarray<type,dims> cA;
	if(!A.host_allowed()){
		cA.template create<memory::CPU>(A);
		cA << A;
		pA = cA;
	};
	//auto A1 = pA.reshape(sz2);
	//std::cout << "A1=" << A1 <<"\n";
	for(int i=0; i < sz2[0]; ++i){
		for(int j=0; j< sz2[1]; ++j){
			intn<dims> ii = A.size().integer_to_index(i + j*sz2[0]);
			ff << pA(ii);
			if(j< sz2[1]-1) ff << ", ";
		};
		ff << "\n";
	};
}

//! file i/o
template<typename type, int dims> void dlm_read(ndarray<type,dims> & A, std::string fname, ndarray_flags fl = ndarray_flags::no_access){
	std::fstream ff;
	ff.open(fname,std::fstream::in);
	if(!ff.is_open()){
		throw_error() << "file" << fname << " not found ";
	};
	dlm_read(A,ff,fl);
	ff.close();
}

//! file i/o
template<typename type, int dims> void dlm_write(const ndarray_ref<type,dims> & A, std::string fname){
	std::fstream ff;
	ff.open(fname,std::fstream::out);
	if(!ff.is_open()){
		throw_error() << "file" << fname << " not opened";
	};
	dlm_write(A,ff);
	ff.close();
}
