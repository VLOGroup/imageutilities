#pragma once
#include <vector_types.h>


template<> struct type_expand<float2>{
	typedef float s_type;
	static const int n = 2;
};

template<> struct type_expand<float3>{
	typedef float s_type;
	static const int n = 3;
};

template<> struct type_expand<float4>{
	typedef float s_type;
	static const int n = 4;
};

template<> struct type_expand<int2>{
	typedef int s_type;
	static const int n = 2;
};

template<> struct type_expand<int3>{
	typedef int s_type;
	static const int n = 3;
};

template<> struct type_expand<int4>{
	typedef int s_type;
	static const int n = 4;
};

template<> struct type_expand<uchar2>{
	typedef unsigned char s_type;
	static const int n = 2;
};

template<> struct type_expand<uchar3>{
	typedef unsigned char s_type;
	static const int n = 3;
};

template<> struct type_expand<uchar4>{
	typedef unsigned char s_type;
	static const int n = 4;
};
