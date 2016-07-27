#ifndef IUHELPERMATH_H
#define IUHELPERMATH_H

#include <helper_math.h>

////////////////////////////////////////////////////////////////////////////////
// other host / device math functions
////////////////////////////////////////////////////////////////////////////////
template<typename Type>
inline __host__ __device__ Type sqr(Type a) {return a*a;}



////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ uchar2 operator+(uchar2 a, uchar2 b)
{
    return make_uchar2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ uchar3 operator+(uchar3 a, uchar3 b)
{
    return make_uchar3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ uchar4 operator+(uchar4 a, uchar4 b)
{
    return make_uchar4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}


////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ uchar2 operator*(uchar2 a, uchar2 b)
{
    return make_uchar2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ uchar3 operator*(uchar3 a, uchar3 b)
{
    return make_uchar3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ uchar4 operator*(uchar4 a, uchar4 b)
{
    return make_uchar4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}


////////////////////////////////////////////////////////////////////////////////
// print various types
////////////////////////////////////////////////////////////////////////////////
inline __host__ std::ostream& operator<< (std::ostream & out, float2 const& v)
{
    out << "[" << v.x << ", " << v.y << "]";
    return out;
}

inline __host__ std::ostream& operator<< (std::ostream & out, int2 const& v)
{
    out << "[" << v.x << ", " << v.y << "]";
    return out;
}

inline __host__ std::ostream& operator<< (std::ostream & out, uint2 const& v)
{
    out << "[" << v.x << ", " << v.y << "]";
    return out;
}

inline __host__ std::ostream& operator<< (std::ostream & out, float3 const& v)
{
    out << "[" << v.x << ", " << v.y << ", " << v.z << "]";
    return out;
}

inline __host__ std::ostream& operator<< (std::ostream & out, int3 const& v)
{
    out << "[" << v.x << ", " << v.y << ", " << v.z << "]";
    return out;
}

inline __host__ std::ostream& operator<< (std::ostream & out, uint3 const& v)
{
    out << "[" << v.x << ", " << v.y << ", " << v.z << "]";
    return out;
}

inline __host__ std::ostream& operator<< (std::ostream & out, float4 const& v)
{
    out << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
    return out;
}

inline __host__ std::ostream& operator<< (std::ostream & out, int4 const& v)
{
    out << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
    return out;
}

inline __host__ std::ostream& operator<< (std::ostream & out, uint4 const& v)
{
    out << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
    return out;
}

#endif
