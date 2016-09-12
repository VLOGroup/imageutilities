ImageUtilities - Bridge the gap between CPU and GPU
===================================================

Installation
-------------

- Set up environment variable `COMPUTE_CAPABILITY` with the CC of your CUDA-enabled GPU
- Set up environment variable `CUDA_SDK_ROOT_DIR` to point to the NVidia CUDA examples
- Set up environment vaiable `IMAGEUTILITIES_ROOT` to point to the path of this directory

- To use the matlab wrapper, set up environment vaiable `MATLAB_ROOT` to point to matlab root directory

To build simply perform the following steps:

~~~
$ cd build
$ cmake ..
$ make
$ make install
~~~

Documentation
-------------
A pre-built documentation is available [here](https://vlogroup.github.io/imageutilities/).
To build the documentation on your own (requires doxygen), additionally do
~~~
$ make apidoc
~~~

Usage
------

In your cmake-based project include the following lines
~~~
set(ImageUtilities_DIR $ENV{IMAGEUTILITIES_ROOT})
find_package(ImageUtilities REQUIRED COMPONENTS iucore)
include_directories(${IMAGEUTILITIES_INCLUDE_DIR})
~~~

and link your application with
~~~
target_link_libraries(your_application
  your_libraries
  ${IMAGEUTILITIES_LIBRARIES}
)
~~~

Example
--------

Image Utilities take away the hassle of memory management when dealing with CUDA
code. The follwing code snippet shows a simple example of image manipulation
using CUDA:

~~~{.c}
// read two images from files
iu::ImageGpu_32f_C1 *I1 = iu::imread_cu32f_C1(DATA_PATH("army_1.png"));
iu::ImageGpu_32f_C1 *I2 = iu::imread_cu32f_C1(DATA_PATH("army_2.png"));
// allocate memory for the output
iu::ImageGpu_32f_C1 result(I1->size());

// Add 0.5 to the first image and save the result
iu::math::addC(*I1,0.5,result);
iu::imsave(&result,RESULTS_PATH("army_1_bright.png"));

// Subtract one image from the other and save the result
iu::math::addWeighted(*I1,1,*I2,-1,result);
iu::imsave(&result,RESULTS_PATH("army_1_minus_2.png"));
~~~

They also make it easy to use images in CUDA kernels by providing additional
information about the image that can be easily passed to kernels. Host code:

~~~{.c}
iu::ImageGpu_32f_C1 img(320,240);

dim3 dimBlock(16, 16);
dim3 dimGrid(iu::divUp(img.width(), dimBlock.x), img.height(), dimBlock.y);

test_kernel <<< dimGrid, dimBlock >>> (img);
~~~

and the device code:

~~~{.c}
__global__ void test_kernel(iu::ImageGpu_32f_C1::KernelData img)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x < img.width_ && y < img.height_)
    {
        img(x,y) = x+y;
    }
}
~~~
