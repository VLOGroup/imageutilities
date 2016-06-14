#include <iostream>
#include "../config.h"
#include "iucore.h"
#include "iuio.h"

int main()
{
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
}
