#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <exception>
#include <string>
//#include <boost/python.hpp>
//#include <numpy/ndarrayobject.h>

#include <iostream>
#include "../config.h"
#include "../../src/iucore.h"
#include "../../src/iuio.h"
#include "../../src/iupython.h"

namespace bp = boost::python;

using std::cout;
using std::endl;



void test1(bp::object& py_obj)
{
    iu::ImageCpu_8u_C1 im(py_obj);
    //iu::python::imageCpu_from_PyArray(py_obj, im);

    cout << im.size() << endl;

    for (int y=0; y<5; y++)
    {
        for (int x=0; x<10; x++)
        {
            printf("%d ", *im.data(x,y));
        }
        printf("\n");
    }

    iu::imsave(&im, "test.png", true);
}

BOOST_PYTHON_MODULE(libpyTest)   // name must (!) be the same as the resulting *.so file
                                 // get python ImportError about missing init function otherwise
                                 // probably best to sort it out in cmake...
{
    import_array();                   // initialize numpy c-api
    bp::register_exception_translator<iu::python::Exc>(&iu::python::ExcTranslator);

    // expose a function to python
    bp::def("test1", test1);

//    bp::class_<SegmentationLib, boost::noncopyable>("SegmentationLib", bp::init<>())
//            .def("solve", &SegmentationLib::solve, solve_overloads())
//            .def("get_u", get_u)             // don't use the class methods directly, call it internally in this function
//            .def("set_weights", set_weights)
//            .def("get_current_pd_gap", &SegmentationLib::get_current_pd_gap);




}
