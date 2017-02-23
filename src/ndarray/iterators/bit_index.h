#pragma once

#include "intn.h"
#include <bitset>
#include <iostream>


 inline int fls(int x){
	 return x ? sizeof(x) * 8 - __builtin_clz(x) : 0;
 }

struct bitindex;
std::ostream & operator << (std::ostream & ss, const bitindex & b);

struct bitindex{
public:
	unsigned int ii;
	unsigned int szc; // size complement
	unsigned int mask; // starting positions mask
public:
	bitindex(const bitindex & b) = default;
	template<int dims>
	bitindex(intn<dims> size){
		ii = 0;
		szc = 0;
		int bits = 0;
		intn<dims> bb;
		intn<dims> cc;
		mask = 0;
//		intn<dims> mask;
		for(int d=0; d<dims; ++d){
			if(size[d] == 0)size[d] = 1;
			bb[d] = bits;
			unsigned int b = fls(size[d] - 1);
			//mask[d] = ((1 << b) -1) << bits;
			//asize[d] = size[d] << bits;
			unsigned int co = (1 << b) - size[d];
			//unsigned int co = ~size[d] && ;
			cc[d] = co;
			szc = szc | (co << bits);
			mask = mask | (1 << bits);
			bits += b;
			std::cout << b << " ";
		};
		std::cout << bb << "\n";
		std::cout << cc << "\n";
		std::cout << "mask=" << std::bitset<32>(mask)<<"\n";
		std::cout << " szc=" << std::bitset<32>(szc)<<"\n";
	}

	template<int dims>
	unsigned int build_increment(intn<dims> di){ // assume di < size -- normalized
		//unsigned int r = 0;
		bitindex r = *this;
		r.ii = 0;
		unsigned int m = mask;
		for(int d = dims-1; d>=0; --d){
			unsigned int b = fls(m) - 1;
			unsigned int a = (di[d] << b);
			m ^= (1 << b); // clear most significant bit
			std::cout << "m =" << std::bitset<32>(m) << "\n";
			r.ii |= a;
		};
		//std::cout << "di=" << std::bitset<32>(r) << "\n";
		std::cout << "di=\n" << r << "\n";
		r.ii += szc;
		std::cout << "dic=\n" << r << "\n";
		return r.ii;
	};

	bitindex(int i){
	}

	bitindex & operator += (int i){
		ii += i; // add with complement
		ii -= szc; // subtract complement back
		return *this;
	}
};

std::ostream & operator << (std::ostream & ss, const bitindex & b){
	std::bitset<32> ii(b.ii);
	std::bitset<32> mask(b.mask);
	for(int i=31; i>= 0; --i){ // printing bits left to right
		ss<<ii[i];
		if(mask[i])ss << " ";
	};
	/*
	int a = b.ii;
	for(int m = ii.mask; m!=0;){ // decode
		unsigned int b = fls(m);
		int x =
	};
	*/
	ss << "\n";
	/*
	std::bitset<32> szc(b.szc);
	for(int i=31; i>= 0; --i){ // printing bits left to right
		ss<<szc[i];
		if(mask[i])ss << " ";
	};
	*/
	return ss;
};

/*
template<int dims, int packing> struct layout{
	int v[dims];
};

template<>
struct layout<1,0>{
	unsigned int i0;
};

template<>
struct layout<2,0>{
	unsigned int i0;
	unsigned int i1;
};

template<>
struct layout<4,1>{
	unsigned short i0;
	unsigned short i1;
	unsigned short i2;
	unsigned short i3;
	static const int offset[4] = {offsetof(layout2, i0),offsetof(layout2, i1),offsetof(layout2, i2),offsetof(layout2, i3)};
};

template<typename pack_layout>
struct packed_index_template{
	union{
		unsigned long long ii;
		pack_layout pack;
	};
};

struct packed_index{
	unsigned long long ii;
	unsigned long long sz;
	unsigned long long di;
	int adv;
	const int type;
	template<int dims>
	packed_index(intn<dims> size){
		// determine layout
	}

	template<int dims>
	int multiply_add(intn<dims> strides){

	}

	void advance(int _adv){
		switch(type){
			0: static_cast<layout<1,0> & >(ii) += static_cast<layout<1,0> & >(di);
		}
	}
};
*/

struct layout{
	const int    b_dims[4] = { 1, 2, 3, 4};
	const int  b1_start[4] = { 0, 0, 0, 0};
	const int  b2_start[4] = {32,32,16,16};
	const int  b3_start[4] = {32,64,32,32};
	const int  b4_start[4] = {32,64,32,48};
};

struct packed_index{
	unsigned long long ii;
	unsigned long long sz;
	unsigned long long di;
	int adv;
	const int type;
	template<int dims>
	packed_index(intn<dims> size){
		// determine layout
	}

	template<int dims>
	int multiply_add(intn<dims> strides){

	}

	template<int type>
	void advance_t(int _adv){
		constexpr int dims =
		int i0 = extract<1>(ii,0);
		i += _adv;
		if(i >= extract<1>(ii,0) )
			extract<1>(ii,0) = i;
		break;
	}

	void advance(int _adv){
		switch(type){// dispatch
			case 0: advance_t<0>(_adv); break;
			case 1: advance_t<1>(_adv); break;
			case 2: advance_t<2>(_adv); break;
			case 3: advance_t<3>(_adv); break;
		};
};


inline void test_bit_index(){
	intn<3> sz(50,60,70);
	bitindex ii(sz);
	unsigned int di = ii.build_increment(intn<3>(47,25,0));

	for(int i=0; i< 10; ++i){
		ii += di;
		std::cout << ii << "\n";
	};
};
