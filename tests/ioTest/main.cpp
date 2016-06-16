#include <iostream>
#include "../config.h"
#include "../../src/iucore.h"
#include "../../src/iuio.h"

int main()
{
<<<<<<< HEAD:src/tests/ioTest/main.cpp
    std::cout << "Read image to device directly" << std::endl;
    std::cout << DATA_PATH("army_1.png") << std::endl;
    //	iu::ImageCpu_32f_C1 I1(320,240);
    //	iu::ImageGpu_32f_C1 I2(320,240);
    //	iu::copy(&I1,&I2);
    iu::ImageCpu_32f_C1 *I1 = iu::imread_32f_C1(DATA_PATH("army_1.png"));
    iu::ImageGpu_32f_C1 *I2 = iu::imread_cu32f_C1(DATA_PATH("army_2.png"));
    iu::ImageGpu_32f_C1 I3;
    if(I1->sameType(*I2))
        std::cout << "yes ";
    else
        std::cout << "no  ";

    if(I2->sameType(I3))
        std::cout << "yes ";
    else
        std::cout << "no  ";

    iu::imsave(I1,RESULTS_PATH("army_1.png"));
    iu::imsave(I2,RESULTS_PATH("army_2.png"));
    std::cout << "DONE :)" << std::endl;
=======
	std::cout << "Read image to device directly" << std::endl;
	std::cout << DATA_PATH("army_1.png") << std::endl;
//	iu::ImageCpu_32f_C1 I1(320,240);
//	iu::ImageGpu_32f_C1 I2(320,240);
//	iu::copy(&I1,&I2);
	iu::ImageCpu_32f_C1 *I1 = iu::imread_32f_C1(DATA_PATH("army_1.png"));
	iu::ImageGpu_32f_C1 *I2 = iu::imread_cu32f_C1(DATA_PATH("army_2.png"));
		
  iu::imsave(I1,RESULTS_PATH("army_1.png"));
  iu::imsave(I2,RESULTS_PATH("army_2.png"));
  std::cout << "DONE :)" << std::endl;

  delete I1;
  delete I2;

  return EXIT_SUCCESS;
>>>>>>> f466a37e355fc8768780a0470baeb4c44b15bd28:tests/ioTest/main.cpp
}
