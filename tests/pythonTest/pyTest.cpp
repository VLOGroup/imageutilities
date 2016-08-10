#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <exception>
#include <string>

#include <iostream>
#include "../config.h"
#include "../../src/iupython.h"

#include "test.cuh"

namespace bp = boost::python;

using std::cout;
using std::endl;


// a simple class that computes something on an image
class MyClass
{
public:
    MyClass() {
        input_ = NULL;
        result_ = NULL;
    }
    ~MyClass() {
        delete input_; input_ = NULL;
        delete result_; result_ = NULL;
    }

    void set_image(iu::ImageCpu_32f_C1& im) {

        if (!input_)
            input_ = new iu::ImageGpu_32f_C1(im.size());
        if (input_->size() != im.size())
        {
            delete input_; delete result_;
            input_ = new iu::ImageGpu_32f_C1(im.size());
            result_ = new iu::ImageGpu_32f_C1(im.size());
        }
        iu::copy(&im, input_);

        if (!result_)
            result_ = new iu::ImageGpu_32f_C1(im.size());

    }

    void compute(float parameter) {
        cout << "C++ class method compute cuda" << endl;
        cuda_function(*input_, *result_, parameter);
    }

    iu::ImageGpu_32f_C1* get_result() { return result_; }

private:
    iu::ImageGpu_32f_C1* input_;
    iu::ImageGpu_32f_C1* result_;
};


// glue code for python
// we can't call the set_image method directly from python since it accepts
// a ImageCPU. We expose this function as the class set_image method instead, note
// that the first argument MUST be the self/this object as usual in a class method call
void myclass_set_image(bp::object& self, bp::object& py_img)
{
    MyClass& my_class = bp::extract<MyClass&>(self);  // get MyClass instance

    iu::ImageCpu_32f_C1 img(py_img);  // wrap numpy array

    my_class.set_image(img);  // call class method
}

PyObject* myclass_get_result(bp::object& self)
{
    MyClass& my_class = bp::extract<MyClass&>(self);  // get MyClass instance

    return iu::python::PyArray_from_ImageGpu(*my_class.get_result());
}


//==============================================================================

// function that can be called from python with a numpy array as argument
void test1(bp::object& py_obj)
{
    // construct imagecpu from numpy array (no deep copy, just wrap data pointer -> very fast!)
    iu::ImageCpu_32f_C1 im(py_obj);
    cout << "Normalize image c++" << endl;

    // normalize image to [0,1]. Since the ImageCpu just wraps the data pointer, this change
    // is transparent to python
    for (int y=0; y<im.height(); y++)
        for (int x=0; x<im.width(); x++)
            *im.data(x,y) = fmax(0.f, fmin(1.f, *im.data(x,y) / 255.0f));
}

// function that can be called with a numpy array as argument and returns a numpy array
PyObject* test2(bp::object& py_obj)
{
    iu::ImageCpu_32f_C1 im(py_obj);  // construct imagecpu from numpy array
    iu::ImageCpu_32f_C1 result(im.size()); // output image
    cout << "Flip image c++" << endl;

    // flip image upside down
    for (int y=0; y<im.height(); y++)
        for (int x=0; x<im.width(); x++)
            *result.data(x,im.height()-y-1) = *im.data(x,y);

    // return a numpy array from the imagecpu. This allocates a new numpy array and makes a
    // deep copy, since we need to make sure the memory returned to python stays valid
    // after destruction of the local variable result.
    // i.e. numpy array can be manipulated in python independently from result
    return iu::python::PyArray_from_ImageCpu(result);
}


void test3(bp::object& py_obj1, bp::object& py_obj2)
{
    Eigen::Matrix4f m1, m2;

    // get eigen matrices from numpy array
    iu::python::Matrix4f_from_PyArray(py_obj1, Eigen::Ref<Eigen::Matrix4f>(m1));
    iu::python::Matrix4f_from_PyArray(py_obj2, Eigen::Ref<Eigen::Matrix4f>(m2));

    cout << "c++ product of m1*m2" << endl << m1*m2 << endl;
}


//==============================================================================
// create python module
//==============================================================================

BOOST_PYTHON_MODULE(libpyTest)   // name must (!) be the same as the resulting *.so file
                                 // get python ImportError about missing init function otherwise
                                 // probably best to sort it out in cmake...
{
    import_array();                   // initialize numpy c-api
    bp::register_exception_translator<iu::python::Exc>(&iu::python::ExcTranslator);

    // expose a function to python
    bp::def("test1", test1);
    bp::def("test2", test2);
    bp::def("test3", test3);


    // expose a class
    bp::class_<MyClass>("MyClass", bp::init<>())
            .def("set_image", myclass_set_image)   // don't expose class method directly, see comments above
            .def("get_result", myclass_get_result)
            .def("compute", &MyClass::compute);   // here we directly expose the class method
                                                // the float parameter is automatically converted by boost.python

}
