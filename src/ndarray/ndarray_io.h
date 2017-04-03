#pragma once

#include "ndarray_ref.host.h"
#include "ndarray_mem.h"
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>


void eat_delim(std::fstream & ff){
	while (!ff.eof() && (ff.peek() == ',' || ff.peek() == '\n' || ff.peek() == '\r' || ff.peek()==' '))ff.ignore();
};

//! file i/o
template<typename type, int dims> void operator >> (std::fstream & ff, ndarray<type,dims> & A){
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
	std::cout<< "reading size=" << sz << "\n";
	A.template create<memory::GPU_managed>(sz);
	intn<2> sz2(A.size()[0], A.size().prod() / A.size()[0]); // reshape as 2D, keep dim 0
	auto A1 = A.reshape(sz2);
	std::cout<< "A1=" << A1 << "\n";
	for(int i=0; i < A1.size()[0]; ++i){
		for(int j=0; j< A1.size()[1]; ++j){
			float val;
			ff >> val;
			//if(i< 4 && j< 4)std::cout << val << ">> A1(" << i <<"," << j << ")\n";
			A1(i,j) = val;
			eat_delim(ff);
		};
	};
}

//! file i/o
template<typename type, int dims> void operator << (std::fstream & ff, const ndarray<type,dims> & A){
	intn<dims> sz = A.size();
	for(int d=0; d<dims; ++d){
		ff << sz[d];
		if(d<dims-1)ff << ", ";
	};
	ff << "\n";
	intn<2> sz2(A.size()[0], A.size().prod() / A.size()[0]); // reshape as 2D, keep dim 0
	auto A1 = A.reshape(sz2);
	for(int i=0; i < A1.size()[0]; ++i){
		for(int j=0; j< A1.size()[1]; ++j){
			ff << A1(i,j);
			if(j< A1.size()[1]-1) ff << ", ";
		};
		ff << "\n";
	};
}

//! file i/o
template<typename type, int dims> void dlm_read(const ndarray<type,dims> & A, std::string fname){
	std::fstream ff;
	ff.open(fname,std::fstream::in);
	if(!ff.is_open()){
		throw_error() << "file" << fname << " not found ";
	};
	ff >> A;
	ff.close();
}

//! file i/o
template<typename type, int dims> void dlm_write(const ndarray_ref<type,dims> & A, std::string fname){
	std::fstream ff;
	ff.open(fname,std::fstream::out);
	if(!ff.is_open()){
		throw_error() << "file" << fname << " not opened";
	};
	ff << A;
	ff.close();
}
