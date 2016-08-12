#include <iostream>
#include "../config.h"
#include "../../src/iucore.h"
#include "test.cuh"

int main()
{

    iu::Size<2> sz1(397, 211);
    iu::Size<2> sz2;
    sz2.width=397; sz2.height=211;
    iu::ImageGpu_32f_C1 d_I(sz1);
    cudaMemset2D(d_I.data(), d_I.pitch(), 0, d_I.pitch(), d_I.height());
    cuda::test(d_I);

    iu::ImageCpu_32f_C1 h_I(sz2);
    for (unsigned int y = 0; y < h_I.height(); y++)
    {
        for (unsigned int x = 0; x < h_I.width(); x++)
        {
			*h_I.data(x, y) = static_cast<float>(x + y);
        }
    }

    iu::ImageCpu_32f_C1 h_I2(sz2);
    iu::copy(&d_I, &h_I2);

    double sum=0;
    for (unsigned int y = 0; y < h_I2.height(); y++)
    {
        for (unsigned int x = 0; x < h_I2.width(); x++)
        {
            sum += *h_I.data(x,y) - *h_I2.data(x,y);
        }
    }

    printf("diff: %f\n", sum);

    if (sum != 0)
        return EXIT_FAILURE;
    else
        return EXIT_SUCCESS;
}
