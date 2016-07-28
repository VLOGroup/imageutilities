#pragma once

#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <exception>
#include <string>
#include <climits>
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include <eigen3/Eigen/Dense>
#include "iucore.h"
#include "iucore/copy.h"
#include "iucore/image_allocator_cpu.h"


#include "iuio.h"


namespace bp = boost::python;
using std::string;

namespace iu {

/// python interface
namespace python {

/** \defgroup IuPython iupython
 * \brief Interface to python. Contains functions to convert between ImageUtilities classes and
 * python objects through boost.python.
 * \{
 */



/**
  * @brief Python exception class
  */
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
 * Custom exception translator to map c++ exceptions -> python.
 * Call "bp::register_exception_translator<Exc>(&iu::python::ExcTranslator);" in your BOOST_PYTHON_MODULE
 * @param err
 */
static void ExcTranslator(const Exc& err)
{
    PyErr_SetString(PyExc_RuntimeError, err.what());
}

/**
 * @brief get a PyArray from a generic PyObject. The memory referenced by the numpy array must be c-contiguous
 * @param obj a PyObject* wrapped in boost::python::object
 * @param kind datatype kind. \<b\>ool, \<i\>nt (signed) \<u\>int, \<f\>loat, \<c\>omplex... See numpy C-API. Default 'x' = disable check
 * @param type \<b\>yte, \<i\>int, \<f\>loat, \<d\>ouble... See numpy C-API. Default 'x' = disable check
 * @param writeable bool to indicate writable. Default true
 * @return a PyArrayObject pointer
 */
PyArrayObject* getPyArrayFromPyObject(const bp::object& obj, char kind = 'x', char type = 'x', bool writeable = true)
{
    // from: https://github.com/numpy/numpy/blob/master/numpy/core/include/numpy/ndarraytypes.h
//    enum NPY_TYPECHAR {
//        NPY_BOOLLTR = '?',
//        NPY_BYTELTR = 'b',
//        NPY_UBYTELTR = 'B',
//        NPY_SHORTLTR = 'h',
//        NPY_USHORTLTR = 'H',
//        NPY_INTLTR = 'i',
//        NPY_UINTLTR = 'I',
//        NPY_LONGLTR = 'l',
//        NPY_ULONGLTR = 'L',
//        NPY_LONGLONGLTR = 'q',
//        NPY_ULONGLONGLTR = 'Q',
//        NPY_HALFLTR = 'e',
//        NPY_FLOATLTR = 'f',
//        NPY_DOUBLELTR = 'd',
//        NPY_LONGDOUBLELTR = 'g',
//        NPY_CFLOATLTR = 'F',
//        NPY_CDOUBLELTR = 'D',
//        NPY_CLONGDOUBLELTR = 'G',
//        NPY_OBJECTLTR = 'O',
//        NPY_STRINGLTR = 'S',
//        NPY_STRINGLTR2 = 'a',
//        NPY_UNICODELTR = 'U',
//        NPY_VOIDLTR = 'V',
//        NPY_DATETIMELTR = 'M',
//        NPY_TIMEDELTALTR = 'm',
//        NPY_CHARLTR = 'c',

//        /*
//             * No Descriptor, just a define -- this let's
//             * Python users specify an array of integers
//             * large enough to hold a pointer on the
//             * platform
//             */
//        NPY_INTPLTR = 'p',
//        NPY_UINTPLTR = 'P',

//        /*
//             * These are for dtype 'kinds', not dtype 'typecodes'
//             * as the above are for.
//             */
//        NPY_GENBOOLLTR ='b',
//        NPY_SIGNEDLTR = 'i',
//        NPY_UNSIGNEDLTR = 'u',
//        NPY_FLOATINGLTR = 'f',
//        NPY_COMPLEXLTR = 'c'
//   };


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


//*
// * @brief imageGpu_from_PyArray iu::imageGpu from a boost::python::object holding a PyArray
// * @param py_arr boost::python::object representing a numpy array
// * @param img The ImageGpu. It should be empty (i.e. size 0), a new ImageGpu with the right size
// * will be created. If the pixeltype of the ImageGpu does not match the datatype of
// * the numpy array an exception is thrown.
// */
//template<typename PixelType, class Allocator>
//void imageGpu_from_PyArray(const bp::object& py_arr,
//                           iu::ImageGpu<PixelType, Allocator> &img)
//{
//    if (img.data())
//        throw Exc("imageGpu_from_PyArray(): Expected emtpy image (will be created with right size)! ");

//    iu::ImageCpu<PixelType, iuprivate::ImageAllocatorCpu<PixelType> > h_img;
//    imageCpu_from_PyArray(py_arr, h_img);

//    img = iu::ImageGpu<PixelType, Allocator>(h_img.size());  // allocate new gpu image

//    iuprivate::copy(&h_img, &img);
//}




/**
 * @brief PyArray from an ImageCpu
 * @param img An ImageCpu
 * @return A PyObject* representing a numpy array that can be returned directly to python. The PyObject* contains
 * a deep copy of the ImageCpu data and can be manipulated in python independent from the ImageCpu
 */
template<typename PixelType, class Allocator>
PyObject* PyArray_from_ImageCpu(iu::ImageCpu<PixelType, Allocator> &img)
{
    npy_intp dims[2] = { img.height(), img.width() };
    PyObject* res = NULL;

    if (dynamic_cast<iu::ImageCpu_32f_C1*>(&img))
        res = PyArray_SimpleNew(2, dims, NPY_FLOAT32);        // new numpy array
    else if (dynamic_cast<iu::ImageCpu_8u_C1*>(&img))
        res = PyArray_SimpleNew(2, dims, NPY_UINT8);        // new numpy array
    else
        throw Exc("PyArray_from_ImageCpu(): conversion for this image type not implemented");


    PixelType* data = static_cast<PixelType*>(PyArray_DATA((PyArrayObject*)res));    // data pointer
    iu::ImageCpu<PixelType, Allocator> h_pyRef(data, dims[1], dims[0], dims[1]*sizeof(PixelType), true);  // wrapped in imagecpu

    iu::copy(&img, &h_pyRef);        // copy img to numpy array
    return res;
}


/**
 * @brief PyArray from an ImageGpu
 * @param img An ImageGpu
 * @return A PyObject* representing a numpy array that can be returned directly to python.
 */
template<typename PixelType, class Allocator>
PyObject* PyArray_from_ImageGpu(iu::ImageGpu<PixelType, Allocator> &img)
{
    npy_intp dims[2] = { img.height(), img.width() };
    PyObject* res = NULL;

    if (dynamic_cast<iu::ImageGpu_32f_C1*>(&img))
        res = PyArray_SimpleNew(2, dims, NPY_FLOAT32);        // new numpy array
    else if (dynamic_cast<iu::ImageGpu_8u_C1*>(&img))
        res = PyArray_SimpleNew(2, dims, NPY_UINT8);        // new numpy array
    else
        throw Exc("PyArray_from_ImageGpu(): conversion for this image type not implemented");


    PixelType* data = static_cast<PixelType*>(PyArray_DATA((PyArrayObject*)res));    // data pointer
    iu::ImageCpu<PixelType, iuprivate::ImageAllocatorCpu<PixelType> > h_pyRef(data, dims[1], dims[0], dims[1]*sizeof(PixelType), true);  // wrapped in imagecpu

    iuprivate::copy(&img, &h_pyRef);
    return res;
}



/**
 * @brief Get an Eigen::Matrix3f from a numpy array
 * @param py_arr numpy array (must contain floating point data, double will be cast to float)
 * @param m Matrix3f to fill
 */
void Matrix3f_from_PyArray(const bp::object& py_arr, Eigen::Ref<Eigen::Matrix3f> m)
{
    // don't care if float or double
    PyArrayObject* arr = getPyArrayFromPyObject(py_arr, 'f', 'x', false);
    int ndim = PyArray_NDIM(arr);
    if (ndim != 2)
        throw Exc("Expected a 2d numpy array");
    npy_intp* dims = PyArray_DIMS(arr);
    if (dims[0] != 3 || dims[1] != 3)
        throw Exc("numpy array must be 3x3");

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
 * @brief Get an Eigen::Matrix4f from a numpy array
 * @param py_arr numpy array (must contain floating point data, double will be cast to float)
 * @param m Matrix4f to fill
 */
void Matrix4f_from_PyArray(const bp::object& py_arr, Eigen::Ref<Eigen::Matrix4f> m)
{
    PyArrayObject* arr = getPyArrayFromPyObject(py_arr, 'f', 'x', false);
    int ndim = PyArray_NDIM(arr);
    if (ndim != 2)
        throw Exc("Expected a 2d numpy array");
    npy_intp* dims = PyArray_DIMS(arr);
    if (dims[0] != 4 || dims[1] != 4)
        throw Exc("numpy array must be 4x4");

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

/** \} */ // end of iupython

} // namespace python

} // namepsace iu


namespace iu {

/** \ingroup IuPython
 *  \{ */

template<typename PixelType, class Allocator>
ImageCpu<PixelType, Allocator>::ImageCpu(boost::python::object &py_arr) : data_(0), pitch_(0), ext_data_pointer_(true)
{
    PyArrayObject* py_img = NULL;

    py_img = python::getPyArrayFromPyObject(py_arr);  // don't care what datatype, just to get dimensions
    int ndim = PyArray_NDIM(py_img);
    if (ndim != 2)
        throw python::Exc("imageCpu_from_PyArray(): Image must be a 2d numpy array");
    npy_intp* dims = PyArray_DIMS(py_img);
    size_ = IuSize(dims[1], dims[0]);

    npy_intp* strides = PyArray_STRIDES(py_img);
//    printf("strides %ld %ld\n", strides[0], strides[1]);

//    PyArray_Descr* dtype = PyArray_DTYPE(py_img);
//    if (dtype->kind == 'u')
//    {
//        printf("unsigned ");
//        if (dtype->type == 'B')
//        {
//            printf("byte\n");
//        }
//        else if (dtype->type == 'I')
//        {
//            printf("int32\n");
//        }
//        else
//        {
//            printf("unknown (%c)\n", dtype->type);
//        }
//    }
//    if (dtype->kind == 'i')
//    {
//        printf("signed ");
//        if (dtype->type == 'b')
//        {
//            printf("byte\n");
//        }
//        else if (dtype->type == 'i')
//        {
//            printf("int32\n");
//        }
//        else
//        {
//            printf("unknown (%c)\n", dtype->type);
//        }
//    }
//    else if (dtype->kind == 'f')
//    {
//        printf("floating point ");
//        if (dtype->type == 'f')
//        {
//            printf("float\n");
//        }
//        else if (dtype->type == 'd')
//        {
//            printf("double\n");
//        }
//        else
//        {
//            printf("unknown (%c)\n", dtype->type);
//        }
//    }

    if (strides[1] != sizeof(PixelType))
    {
        char msg[250];
        sprintf(msg, "ImageCpu from PyArray datatype mismatch.\n"
                     "Tried to construct Image with %ld-bit type, PyArray has "
                     "%ld-bit type\n", sizeof(PixelType)*CHAR_BIT, strides[1]*CHAR_BIT);
        throw python::Exc(msg);
    }

    pitch_ = strides[0];
    data_ = reinterpret_cast<PixelType*>(PyArray_DATA(py_img));
}

/** \} */ // end of ingroup copy


} // namespace iu

