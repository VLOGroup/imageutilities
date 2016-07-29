#include <iostream>
#include "../config.h"
#include "../../src/iucore.h"
#include "../../src/iuio.h"
#include "../../src/iuio/openexrio.h"

int main()
{
    std::cout << "Using OpenEXR io module" << std::endl;

    iu::OpenEXRInputFile in1(DATA_PATH("frame_0000.exr"));
    std::cout << "total number of channels: " << in1.get_channels().size() << std::endl;
    for (unsigned int i=0; i < in1.get_channels().size(); i++)
    {
        iu::OpenEXRInputFile::Channel c = in1.get_channels().at(i);
        std::cout << "  channel " << i << " name: " << c.name_ << std::endl;
        std::cout << "  channel " << i << " type: " << c.type_ << std::endl;
    }

    std::cout << "total number of attributes: " << in1.get_attributes().size() << std::endl;
    for (unsigned int i=0; i < in1.get_attributes().size(); i++)
    {
        std::cout << "  attribute " << i << " name: " << in1.get_attributes().at(i) << std::endl;
    }

    // OpenEXR supports uint, half and float. channel "gray" of frame_000.exr contains
    // a normal uchar [0-255] image stored as uint.
    iu::ImageCpu_32f_C1 data1(in1.get_size());
    iu::ImageCpu_32u_C1 data2(in1.get_size());
    in1.read_channel("depth", data1);
    in1.read_channel("gray", data2);

    iu::ImageCpu_32f_C1 data3(data2.size());
    iu::convert_32u32f_C1(&data2, &data3, 1);  // data3 will contain values [0-255]

    iu::ImageCpu_32f_C1 data4(in1.get_size());
    in1.read_channel_32f("gray", data4);      // convenience method. If the channel contains uint data, it will be converted to float
    iu::ImageCpu_8u_C1 data5(data4.size());
    iu::convert_32f8u_C1(&data4, &data5, 1);  // convert directly to uchar

    iu::imsave(&data1, RESULTS_PATH("exr_depth.png"), true);
    iu::imsave(&data3, RESULTS_PATH("exr_gray.png"), true);
    iu::imsave(&data4, RESULTS_PATH("exr_gray1.png"), true);
    iu::imsave(&data5, RESULTS_PATH("exr_gray2.png"));


    iu::OpenEXROutputFile out1(RESULTS_PATH("exr_test.exr"), data1.size());
    out1.add_channel("depth", data1);
    out1.add_channel("gray1", data2);
    out1.add_channel("gray2", data5);  // convenience method, uchar image will be stored as uint
    out1.write();
		
    std::cout << "DONE :)" << std::endl;

    return EXIT_SUCCESS;
}
