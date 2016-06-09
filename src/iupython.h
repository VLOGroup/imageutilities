#pragma once

#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <exception>
#include <string>
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include <eigen3/Eigen/Dense>
#include <iu/iucore.h>



namespace bp = boost::python;
using std::string;

class Exc
{
public:
    Exc(const std::string msg) { what_ = msg; }
    ~Exc() {}

    const char* what() const { return what_.c_str(); }

private:
    std::string what_;
};


/**
 * @brief getPyArrayFromPyObject get a PyArray from a generic PyObject. The memory referenced by the numpy array must be c-contiguous
 * @param obj a PyObject* wrapped in boost::python
 * @param kind datatype kind. <b>ool, <i>nt (signed) <u>int, <f>loat, <c>omplex... See numpy C-API
 * @param type <i>int, <f>loat, <d>ouble... See numpy C-API
 * @param writeable bool to indicate writable
 * @return a PyArrayObject pointer
 */
PyArrayObject* getPyArrayFromPyObject(bp::object& obj, char kind = 'f', char type = 'f', bool writeable = true)
{
    if (!PyArray_Check(obj.ptr()))
        throw Exc("Argument is not a numpy array");


    PyArrayObject* pyarr = (PyArrayObject*)obj.ptr();    // safe since we know it's a Pyarrayobject


    if (!PyArray_IS_C_CONTIGUOUS(pyarr))
        throw Exc("Numpy array is not c-contiguous (maybe you are using transpose() or reshape() somewhere?)");

    if (writeable && !PyArray_ISWRITEABLE(pyarr))
        throw Exc("Numpy array is not writeable");

    PyArray_Descr* dtype = PyArray_DTYPE(pyarr);
    if (kind != 'x')
    {
        if (dtype->kind != kind)
            throw Exc("Numpy array wrong datatype kind '"+string(1,dtype->kind)+"'. Expected kind '"+string(1,kind)+"'");
    }
    if (type != 'x')
    {
        if (dtype->type != type)
            throw Exc("Numpy array wrong datatype type '"+string(1,dtype->type)+"'. Expected type '"+string(1,type)+"'");
    }

    return pyarr;
}


template<typename PixelType, class Allocator, IuPixelType _pixel_type>
void imageCpu_from_PyArray(bp::object& py_arr,
                           iu::ImageCpu<PixelType, Allocator, _pixel_type> &img)
{
    PyArrayObject* py_img = getPyArrayFromPyObject(py_arr, 'f', 'f', false);
    int ndim = PyArray_NDIM(py_img);
    if (ndim != 2)
        throw Exc("Image must be a 2d numpy array");
    npy_intp* dims = PyArray_DIMS(py_img);

    float* data = static_cast<float*>(PyArray_DATA(py_img));          // get data pointer

    img = iu::ImageCpu<PixelType, Allocator, _pixel_type>(data, dims[1], dims[0], dims[1]*sizeof(float), true);  // wrap it in imagecpu
}


/**
 * @brief Matrix3f_from_PyArray Get an Eigen::Matrix3f from a numpy array
 * @param py_arr numpy array (must contain floating point data, double will be cast to float)
 * @param m Matrix3f to fill
 */
void Matrix3f_from_PyArray(bp::object& py_arr, Eigen::Ref<Eigen::Matrix3f> m)
{
    // don't care if float or double
    PyArrayObject* arr = getPyArrayFromPyObject(py_arr, 'f', 'x', false);
    int ndim = PyArray_NDIM(arr);
    if (ndim != 2)
        throw Exc("Expected a 2d numpy array");
    npy_intp* dims = PyArray_DIMS(arr);
    if (dims[0] < 3 || dims[1] < 3)
        throw Exc("numpy array must be at least 3x3");

    void* data;
    PyArray_Descr* dtype = PyArray_DTYPE(arr);
    if (dtype->type == 'f')
    {
        data = static_cast<float*>(PyArray_DATA(arr));
        m(0,0) = static_cast<float*>(data)[0]; m(0,1) = static_cast<float*>(data)[1]; m(0,2) = static_cast<float*>(data)[2];
        m(1,0) = static_cast<float*>(data)[3]; m(1,1) = static_cast<float*>(data)[4]; m(1,2) = static_cast<float*>(data)[5];
        m(2,0) = static_cast<float*>(data)[6]; m(2,1) = static_cast<float*>(data)[7]; m(2,2) = static_cast<float*>(data)[8];
    }
    else if (dtype->type == 'd')
    {
        data = static_cast<double*>(PyArray_DATA(arr));
        m(0,0) = static_cast<double*>(data)[0]; m(0,1) = static_cast<double*>(data)[1]; m(0,2) = static_cast<double*>(data)[2];
        m(1,0) = static_cast<double*>(data)[3]; m(1,1) = static_cast<double*>(data)[4]; m(1,2) = static_cast<double*>(data)[5];
        m(2,0) = static_cast<double*>(data)[6]; m(2,1) = static_cast<double*>(data)[7]; m(2,2) = static_cast<double*>(data)[8];
    }
}

/**
 * @brief Matrix4f_from_PyArray Get an Eigen::Matrix4f from a numpy array
 * @param py_arr numpy array (must contain floating point data, double will be cast to float)
 * @param m Matrix4f to fill
 */
void Matrix4f_from_PyArray(bp::object& py_arr, Eigen::Ref<Eigen::Matrix4f> m)
{
    PyArrayObject* arr = getPyArrayFromPyObject(py_arr, 'f', 'x', false);
    int ndim = PyArray_NDIM(arr);
    if (ndim != 2)
        throw Exc("Expected a 2d numpy array");
    npy_intp* dims = PyArray_DIMS(arr);
    if (dims[0] < 4 || dims[1] < 4)
        throw Exc("numpy array must be at least 4x4");

    void* data;
    PyArray_Descr* dtype = PyArray_DTYPE(arr);
    if (dtype->type == 'f')
    {
        data = static_cast<float*>(PyArray_DATA(arr));
        m(0,0) = static_cast<float*>(data)[0]; m(0,1) = static_cast<float*>(data)[1]; m(0,2) = static_cast<float*>(data)[2]; m(0,3) = static_cast<float*>(data)[3];
        m(1,0) = static_cast<float*>(data)[4]; m(1,1) = static_cast<float*>(data)[5]; m(1,2) = static_cast<float*>(data)[6]; m(1,3) = static_cast<float*>(data)[7];
        m(2,0) = static_cast<float*>(data)[8]; m(2,1) = static_cast<float*>(data)[9]; m(2,2) = static_cast<float*>(data)[10]; m(2,3) = static_cast<float*>(data)[11];
        m(3,0) = static_cast<float*>(data)[12]; m(3,1) = static_cast<float*>(data)[13]; m(3,2) = static_cast<float*>(data)[14]; m(3,3) = static_cast<float*>(data)[15];
    }
    else if (dtype->type == 'd')
    {
        data = static_cast<double*>(PyArray_DATA(arr));
        m(0,0) = static_cast<double*>(data)[0]; m(0,1) = static_cast<double*>(data)[1]; m(0,2) = static_cast<double*>(data)[2]; m(0,3) = static_cast<double*>(data)[3];
        m(1,0) = static_cast<double*>(data)[4]; m(1,1) = static_cast<double*>(data)[5]; m(1,2) = static_cast<double*>(data)[6]; m(1,3) = static_cast<double*>(data)[7];
        m(2,0) = static_cast<double*>(data)[8]; m(2,1) = static_cast<double*>(data)[9]; m(2,2) = static_cast<double*>(data)[10]; m(2,3) = static_cast<double*>(data)[11];
        m(3,0) = static_cast<double*>(data)[12]; m(3,1) = static_cast<double*>(data)[13]; m(3,2) = static_cast<double*>(data)[14]; m(3,3) = static_cast<double*>(data)[15];
    }
}

/**
 * @brief ExcTranslator Custom exception translator to map c++ exceptions -> python.
 * Call "bp::register_exception_translator<Exc>(&ExcTranslator);" in your BOOST_PYTHON_MODULE
 * @param err
 */
static void ExcTranslator(const Exc& err)     //custom excpetion translator for boost.python
{
    PyErr_SetString(PyExc_RuntimeError, err.what());
}
