#ifndef OPENEXRIO_H
#define OPENEXRIO_H

#include <string>
#include <vector>
#include <map>
#include <ImfIO.h>
#include <ImfOutputFile.h>
#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImfMatrixAttribute.h>

#include "../iucore.h"

#ifdef IUIO_EIGEN3
    #include <eigen3/Eigen/Dense>
#endif

namespace iu {



class OpenEXRInputFile
{
public:
    /**
     * @brief OpenEXRInputFile constructor. Opens a file for reading.
     * @param filename name of the file.
     */
    OpenEXRInputFile(const std::string& filename);
    ~OpenEXRInputFile();


    struct Channel
    {
        Channel(const std::string& name, const std::string& type) {
            name_ = name;
            type_ = type;
        }
        std::string name_;
        std::string type_;
    };

    IuSize get_size() { return sz_; }
    std::vector<Channel> get_channels() { return channels_; }
    std::vector<std::string> get_attributes();

    void read_channel(const std::string& name, ImageCpu_32u_C1 &img);
    void read_channel(const std::string& name, ImageCpu_32f_C1 &img);
    void read_channel_32f(const std::string& name, ImageCpu_32f_C1 &img);

    void read_channel(const std::string& name, ImageGpu_32u_C1 &img);
    void read_channel(const std::string& name, ImageGpu_32f_C1 &img);
    void read_channel_32f(const std::string& name, ImageGpu_32f_C1 &img);


    #ifdef IUIO_EIGEN3
    void read_attribute(const std::string& name, Eigen::Ref<Eigen::Matrix3f> mat);
    void read_attribute(const std::string& name, Eigen::Ref<Eigen::Matrix4f> mat);
    #endif

private:
    IuSize sz_;
    std::string filename_;
    std::vector<Channel> channels_;
    std::map<Imf::PixelType, std::string> pixeltype_to_string_;
};


class OpenEXROutputFile
{
public:
    /**
     * @brief OpenEXROutputFile constructor. Opens a file for writing.
     * @param filename name of the file.
     * @param size size of the image.
     */
    OpenEXROutputFile(const std::string& filename, IuSize size);
    ~OpenEXROutputFile();


    void add_channel(const std::string& name, iu::ImageCpu_8u_C1& img);
    void add_channel(const std::string& name, iu::ImageCpu_32u_C1& img);
    void add_channel(const std::string& name, iu::ImageCpu_32f_C1& img);
    void add_channel(const std::string& name1, const std::string& name2, iu::ImageCpu_32f_C2& img);
    void add_channel(const std::string& name1, const std::string& name2,
                     const std::string& name3, const std::string& name4, iu::ImageCpu_32f_C4& img);

    void add_channel(const std::string& name, iu::ImageGpu_32f_C1& img);
    void add_channel(const std::string& name1, const std::string& name2, iu::ImageGpu_32f_C2& img);
    void add_channel(const std::string& name1, const std::string& name2,
                     const std::string& name3, const std::string& name4, iu::ImageGpu_32f_C4& img);

    #ifdef IUIO_EIGEN3
    void add_attribute(const std::string& name, Eigen::Ref<Eigen::Matrix3f> mat);
    void add_attribute(const std::string& name, Eigen::Ref<Eigen::Matrix4f> mat);
    #endif

    void write();

private:
    bool check_channel_name(const std::string& name);
    bool check_attachement_name(const std::string& name);

    IuSize sz_;
    std::string filename_;

    Imf::Header header_;
    Imf::FrameBuffer fb_;

    // temporary memory for calling add_channel with gpu images
    // cannot use local ImageCpu* copy in add_channel, because data is accessed
    // only when write() is called, at that point a local copy in add_channel is not
    // available any more...
    std::vector<iu::ImageCpu_32f_C1*> pool_32f_C1_;
    std::vector<iu::ImageCpu_32f_C2*> pool_32f_C2_;
    std::vector<iu::ImageCpu_32f_C4*> pool_32f_C4_;
};

} // namespace iu

#endif // OPENEXRIO_H
