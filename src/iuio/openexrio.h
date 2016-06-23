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


/** \defgroup OpenEXRIO OpenEXR
  * \ingroup IO
  * \brief Read  and write OpenEXR files
  * \{
  */

/**
 * @brief The OpenEXRInputFile class interfaces the OpenEXR library with the Imageutilities. After
 * construction various information (image size, channels, attributes) can be queried which
 * allows to construct the corresponding ImageCpu* variables needed to call the \ref read_channel method.
 * convenience methods to read directly to GPU-images are provided as well.
 */
class OpenEXRInputFile
{
public:
    /**
     * @brief OpenEXRInputFile constructor. Opens a file for reading.
     * @param filename name of the file.
     */
    OpenEXRInputFile(const std::string& filename);
    ~OpenEXRInputFile();

    /**
     * @brief The Channel struct contains the name (string) and datatype (string) of a channel
     */
    struct Channel
    {
        Channel(const std::string& name, const std::string& type) {
            name_ = name;
            type_ = type;
        }
        std::string name_;
        std::string type_;
    };

    /**
     * @brief return size of the OpenEXR image
     * @return Iusize
     */
    IuSize get_size() { return sz_; }

    /**
     * @brief Get a list of all channels in the file
     * @return vector of \ref Channel
     */
    std::vector<Channel> get_channels() { return channels_; }

    /**
     * @brief Get a list of all attributes (attachements) in the file
     * @return vector of attributes
     */
    std::vector<std::string> get_attributes();

    /**
     * @brief Search for a channel \p name and read its contents into \p img. The size of \p img
     *  must match the size returned by \ref get_size.
     * @param name name of the channel. the channel must contain uint32 data.
     * @param img An image whose size matches that returned by \ref get_size
     */
    void read_channel(const std::string& name, ImageCpu_32u_C1 &img);

    /**
     * @brief Search for a channel \p name and read its contents into \p img. The size of \p img
     *  must match the size returned by \ref get_size.
     * @param name name of the channel. the channel must contain float data.
     * @param img An image whose size matches that returned by \ref get_size
     */
    void read_channel(const std::string& name, ImageCpu_32f_C1 &img);

    /**
     * @brief Convenience function to read into a float image regardless of channel data. If
     * the channel contains uint32 data, it will be converted to float.
     * @param name name of the channel
     * @param img An image whose size matches that returned by \ref get_size
     */
    void read_channel_32f(const std::string& name, ImageCpu_32f_C1 &img);

    /**
     * @brief read directly to a GPU-image
     * @param name name of the channel containing uint32 data
     * @param img A GPU-image whose size matches that returned by \ref get_size
     */
    void read_channel(const std::string& name, ImageGpu_32u_C1 &img);

    /**
     * @brief read directly to a GPU-image
     * @param name name of the channel containing float data
     * @param img A GPU-image whose size matches that returned by \ref get_size
     */
    void read_channel(const std::string& name, ImageGpu_32f_C1 &img);

    /**
     * @brief Convenience function to read into a GPU float image regardless of channel data. If
     * the channel contains uint32 data, it will be converted to float.
     * @param name name of the channel
     * @param img A GPU-image whose size matches that returned by \ref get_size
     */
    void read_channel_32f(const std::string& name, ImageGpu_32f_C1 &img);


    #ifdef IUIO_EIGEN3
    /**
     * @brief Read a 3x3 matrix attribute into an Eigen::Matrix3f
     * @param name attribute name (must be a 3x3 matrix)
     * @param mat Ref to Eigen::Matrix3f
     */
    void read_attribute(const std::string& name, Eigen::Ref<Eigen::Matrix3f> mat);

    /**
     * @brief Read a 4x4 matrix attribute into an Eigen::Matrix4f
     * @param name attribute name (must be a 4x4 matrix)
     * @param mat Ref to Eigen::Matrix4f
     */
    void read_attribute(const std::string& name, Eigen::Ref<Eigen::Matrix4f> mat);
    #endif

private:
    IuSize sz_;
    std::string filename_;
    std::vector<Channel> channels_;
    std::map<Imf::PixelType, std::string> pixeltype_to_string_;
};


/**
 * @brief The OpenEXROutputFile class interfaces the OpenEXR library with the Imageutilities. Constructing
 * an \ref OpenEXROutputFile requires the filename and image size. Data is added through various \ref add_channel and
 * \ref add_attribute methods. Note that \ref write has to called explicitly to write the data to disk.
 * \todo what happens if \ref write is called multiple times?
 */
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


    /**
     * @brief add the channel \p name with image data \p img to the OpenEXR file.
     * unsigned char data will be converted to uint32.
     * @param name name of the channel (must not exist already)
     * @param img Image
     */
    void add_channel(const std::string& name, iu::ImageCpu_8u_C1& img);

    /**
     * @brief add the channel \p name with image data \p img to the OpenEXR file.
     * @param name name of the channel (must not exist already)
     * @param img Image
     */
    void add_channel(const std::string& name, iu::ImageCpu_32u_C1& img);

    /**
     * @brief add the channel \p name with image data \p img to the OpenEXR file.
     * @param name name of the channel (must not exist already)
     * @param img Image
     */
    void add_channel(const std::string& name, iu::ImageCpu_32f_C1& img);

    /**
     * @brief Convenience method to add a 2-channel image to the OpenEXR file
     * @param name1 name of first channel
     * @param name2 name of second channel
     * @param img Image
     */
    void add_channel(const std::string& name1, const std::string& name2, iu::ImageCpu_32f_C2& img);

    /**
     * @brief Convenience method to add a 4-channel image to the OpenEXR file
     * @param name1 channel name
     * @param name2 channel name
     * @param name3 channel name
     * @param name4 channel name
     * @param img Image
     */
    void add_channel(const std::string& name1, const std::string& name2,
                     const std::string& name3, const std::string& name4, iu::ImageCpu_32f_C4& img);

    /**
     * @brief GPU version of \ref add_channel
     * @param name channel name
     * @param img Image
     */
    void add_channel(const std::string& name, iu::ImageGpu_32f_C1& img);

    /**
     * @brief GPU version of \ref add_channel
     * @param name1 channel name
     * @param name2 channel name
     * @param img Image
     */
    void add_channel(const std::string& name1, const std::string& name2, iu::ImageGpu_32f_C2& img);

    /**
     * @brief GPU version of \ref add_channel
     * @param name1 channel name
     * @param name2 channel name
     * @param name3 channel name
     * @param name4 channel name
     * @param img Image
     */
    void add_channel(const std::string& name1, const std::string& name2,
                     const std::string& name3, const std::string& name4, iu::ImageGpu_32f_C4& img);

    #ifdef IUIO_EIGEN3
    /**
     * @brief add a 3x3 matrix attribute to the OpenEXR file
     * @param name attribute name
     * @param mat A 3x3 matrix
     */
    void add_attribute(const std::string& name, Eigen::Ref<Eigen::Matrix3f> mat);

    /**
     * @brief add a 4x4 matrix attribute to the OpenEXR file
     * @param name attribute name
     * @param mat A 4x4 matrix
     */
    void add_attribute(const std::string& name, Eigen::Ref<Eigen::Matrix4f> mat);
    #endif

    /**
     * @brief Write to disk. Call this function once all channels/attributes have been added.
     */
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

/** \}  */ // end of OpenEXRIO


} // namespace iu

#endif // OPENEXRIO_H
