#ifndef THRUST_KERNELS_CUH
#define THRUST_KERNELS_CUH

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>

#include "helper_math.h"

namespace iuprivate {
namespace math {

//// special operators for combo-floats
//__host__ __device__ float2 operator*(const float2& v1, const float2& v2) { return make_float2(v1.x*v2.x,v1.y*v2.y);}
//__host__ __device__ float2 operator+(const float2& v1, const float2& v2) { return make_float2(v1.x+v2.x,v1.y+v2.y);}

template <typename ValueType, typename IndexType>
struct minmax_transform_tuple :
        public thrust::unary_function< thrust::tuple<ValueType,IndexType>,
        thrust::tuple<bool,ValueType,ValueType> >
{
    typedef typename thrust::tuple<ValueType, IndexType> InputTuple;
    typedef typename thrust::tuple<bool,ValueType,ValueType> OutputTuple;
    IndexType n, N;
    minmax_transform_tuple(IndexType n, IndexType N) : n(n), N(N) {}
    __host__ __device__
    OutputTuple operator()(const InputTuple& t) const
    {
        bool is_valid = (thrust::get<1>(t) % N) < n;
        return OutputTuple(is_valid, thrust::get<0>(t), thrust::get<0>(t));
    }
};

template <typename ValueType, typename IndexType>
struct minmax_reduce_tuple :
        public thrust::binary_function< thrust::tuple<bool,ValueType,ValueType>,
        thrust::tuple<bool,ValueType,ValueType>,
        thrust::tuple<bool,ValueType,ValueType> >
{
    typedef typename thrust::tuple<bool,ValueType,ValueType> Tuple;
    __host__ __device__
    Tuple operator()(const Tuple& t0, const Tuple& t1) const
    {
        if(thrust::get<0>(t0) && thrust::get<0>(t1)) // both valid
            return Tuple(true,
                         thrust::min(thrust::get<1>(t0), thrust::get<1>(t1)),
                         thrust::max(thrust::get<2>(t0), thrust::get<2>(t1)));
        else if (thrust::get<0>(t0))
            return t0;
        else if (thrust::get<0>(t1))
            return t1;
        else
            return t1; // if neither is valid then it doesn't matter what we return
    }
};

template <typename ValueType, typename IndexType>
struct sum_transform_tuple :
        public thrust::unary_function< thrust::tuple<ValueType, IndexType>,
        thrust::tuple<bool,ValueType,ValueType> >
{
    typedef typename thrust::tuple<ValueType, IndexType> InputTuple;
    typedef typename thrust::tuple<bool,ValueType> OutputTuple;
    IndexType n, N;
    sum_transform_tuple(IndexType n, IndexType N) : n(n), N(N) {}
    __host__ __device__
    OutputTuple operator()(const InputTuple& t) const
    {
        bool is_valid = (thrust::get<1>(t) % N) < n;
        return OutputTuple(is_valid, thrust::get<0>(t));
    }
};

template <typename ValueType, typename IndexType>
struct sum_reduce_tuple :
        public thrust::binary_function< thrust::tuple<bool,ValueType,ValueType>,
        thrust::tuple<bool,ValueType,ValueType>,
        thrust::tuple<bool,ValueType,ValueType> >
{
    typedef typename thrust::tuple<bool,ValueType> Tuple;
    __host__ __device__
    Tuple operator()(const Tuple& t0, const Tuple& t1) const
    {
        if(thrust::get<0>(t0) && thrust::get<0>(t1)) // both valid
            return Tuple(true,thrust::get<1>(t0)+thrust::get<1>(t1));
        else if (thrust::get<0>(t0))
            return t0;
        else
            return t1;

    }
};

template <typename ValueType, typename IndexType>
struct diffsqr_transform_tuple :
        public thrust::unary_function< thrust::tuple<ValueType,ValueType,IndexType>,
        thrust::tuple<bool,ValueType> >
{
    typedef typename thrust::tuple<ValueType,ValueType,IndexType> InputTuple;
    typedef typename thrust::tuple<bool,ValueType> OutputTuple;
    IndexType n, N;
    diffsqr_transform_tuple(IndexType n, IndexType N) : n(n), N(N) {}
    __host__ __device__
    OutputTuple operator()(const InputTuple& t) const
    {
        bool is_valid = (thrust::get<2>(t) % N) < n;
        return OutputTuple(is_valid, (thrust::get<1>(t)-thrust::get<0>(t))*(thrust::get<1>(t)-thrust::get<0>(t)));
    }
};

template <typename ValueType>
struct weightedsum_transform_tuple :
        public thrust::unary_function< thrust::tuple<ValueType,ValueType>,ValueType>
{
    typedef typename thrust::tuple<ValueType,ValueType> InputTuple;
    ValueType w1,w2;
    weightedsum_transform_tuple(ValueType _w1, ValueType _w2) : w1(_w1), w2(_w2) {}
    __host__ __device__
    ValueType operator()(const InputTuple& t) const
    {
        return thrust::get<0>(t)*w1+thrust::get<1>(t)*w2;
    }
};

template <typename ValueType, typename IndexType>
struct diffabs_transform_tuple :
        public thrust::unary_function< thrust::tuple<ValueType,ValueType,IndexType>,
        thrust::tuple<bool,ValueType> >
{
    typedef typename thrust::tuple<ValueType,ValueType,IndexType> InputTuple;
    typedef typename thrust::tuple<bool,ValueType> OutputTuple;
    IndexType n, N;
    diffabs_transform_tuple(IndexType n, IndexType N) : n(n), N(N) {}
    __host__ __device__
    OutputTuple operator()(const InputTuple& t) const
    {
        bool is_valid = (thrust::get<2>(t) % N) < n;
        return OutputTuple(is_valid, abs(thrust::get<1>(t)-thrust::get<0>(t)));
    }
};

} // namespace math
} // namespace iuprivate

#endif // THRUST_KERNELS_CUH
