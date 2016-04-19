#include "openexrio.h"


using std::string;

namespace iu {

OpenEXRInputFile::OpenEXRInputFile(const std::string &filename)
{
    filename_ = filename;
    Imf::InputFile file(filename.c_str());
    Imath::Box2i dw = file.header().dataWindow();

    int width = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;
    sz_ = IuSize(width, height);

    channels_.clear();
    Imf::ChannelList cl = file.header().channels();
    Imf::ChannelList::Iterator it;
    for (it=cl.begin(); it != cl.end(); it++)
    {
        switch(it.channel().type)
        {
        case Imf_2_1::UINT:
            channels_.push_back(Channel(it.name(), IU_32U_C1));
            break;
        case Imf_2_1::HALF:
            printf("OpenEXRFile: encountered channel of dataype HALF in file %s, corresponding datatype (16 bit float) not implemented in imageutilities!\n", it.name());
            break;
        case Imf_2_1::FLOAT:
            channels_.push_back(Channel(it.name(), IU_32F_C1));
            break;
        }
    }
}


OpenEXRInputFile::~OpenEXRInputFile()
{
}

std::vector<std::string> OpenEXRInputFile::get_attributes()
{
    Imf::InputFile file(filename_.c_str());

    Imf::Header::ConstIterator it;
    std::vector<std::string> result;
    result.clear();

    for(it = file.header().begin(); it != file.header().end(); it++)
        result.push_back(it.name());

    return result;
}

void OpenEXRInputFile::read_channel(const std::string &name, ImageCpu_32u_C1 &img)
{
    Imf::InputFile file(filename_.c_str());
    Imath::Box2i dw = file.header().dataWindow();

    int width = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;

    if (img.width() != width || img.height() != height)
    {
        printf("OpenEXRFile get_channel(): image sizes do not match!\n");
        return;
    }

    if (!file.header().channels().findChannel(name))
    {
        printf("OpenEXRFile get_channel(): couldn't find channel %s in exr file!\n", name.c_str());
        return;
    }

    if (file.header().channels().findChannel(name)->type != Imf::UINT)
    {
        printf("OpenEXRFile get_channel(): channel %s does not contain UINT data!\n", name.c_str());
        return;
    }

    Imf::FrameBuffer fb;

    fb.insert(name, Imf::Slice(Imf::UINT, (char*)img.data(0,0), sizeof(unsigned int), img.pitch()));
    file.setFrameBuffer(fb);
    file.readPixels(dw.min.y, dw.max.y);
}

void OpenEXRInputFile::read_channel(const std::string &name, ImageCpu_32f_C1 &img)
{
    Imf::InputFile file(filename_.c_str());
    Imath::Box2i dw = file.header().dataWindow();

    int width = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;

    if (img.width() != width || img.height() != height)
    {
        printf("OpenEXRFile read_channel(): image sizes do not match!\n");
        return;
    }

    if (!file.header().channels().findChannel(name))
    {
        printf("OpenEXRFile read_channel(): couldn't find channel %s in %s!\n", name.c_str(), filename_.c_str());
        return;
    }

    if (file.header().channels().findChannel(name)->type != Imf::FLOAT)
    {
        printf("OpenEXRFile read_channel(): channel %s does not contain FLOAT data!\n", name.c_str());
        return;
    }

    Imf::FrameBuffer fb;

    fb.insert(name, Imf::Slice(Imf::FLOAT, (char*)img.data(0,0), sizeof(float), img.pitch()));
    file.setFrameBuffer(fb);
    file.readPixels(dw.min.y, dw.max.y);

}

#ifdef IUIO_EIGEN3
void OpenEXRInputFile::read_attribute(const std::string &name, Eigen::Ref<Eigen::Matrix3f> mat)
{
    Imf::InputFile file(filename_.c_str());
    const Imf::M33fAttribute *m33Attr = file.header().findTypedAttribute<Imf::M33fAttribute>(name);

    if (!m33Attr)
    {
        printf("Error: Attribute %s (3x3) not found in %s!\n", name.c_str(), filename_.c_str());
        return;
    }


    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            mat(i,j) = m33Attr->value().x[i][j];
}

void OpenEXRInputFile::read_attribute(const std::string &name, Eigen::Ref<Eigen::Matrix4f> mat)
{
    Imf::InputFile file(filename_.c_str());
    const Imf::M44fAttribute *m44Attr = file.header().findTypedAttribute<Imf::M44fAttribute>(name);

    if (!m44Attr)
    {
        printf("Error: Attribute %s (4x4) not found in %s!\n", name.c_str(), filename_.c_str());
        return;
    }


    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            mat(i,j) = m44Attr->value().x[i][j];

}

OpenEXROutputFile::OpenEXROutputFile(const std::string &filename, IuSize size)
{
    filename_ = filename;
    sz_ = size;
    header_ = Imf::Header(sz_.width, sz_.height);
}

OpenEXROutputFile::~OpenEXROutputFile()
{
}


void OpenEXROutputFile::add_channel(const std::string &name, ImageCpu_8u_C1 &img)
{
    if (!check_channel_name(name))
        return;

    header_.channels().insert(name, Imf::Channel(Imf::UINT));

    iu::ImageCpu_32u_C1 temp(img.size());
    for (int y=0; y < img.height(); y++)
    {
        for (int x=0; x < img.width(); x++)
        {
            *temp.data(x,y) = *img.data(x,y);
        }
    }
    fb_.insert(name, Imf::Slice(Imf::UINT, (char*)temp.data(0,0), sizeof(unsigned int), temp.pitch()));
}



void OpenEXROutputFile::add_channel(const std::string &name, ImageCpu_32u_C1 &img)
{
    if (!check_channel_name(name))
        return;

    header_.channels().insert(name, Imf::Channel(Imf::UINT));
    fb_.insert(name, Imf::Slice(Imf::UINT, (char*)img.data(0,0), sizeof(unsigned int), img.pitch()));
}



void OpenEXROutputFile::add_channel(const std::string &name, ImageCpu_32f_C1 &img)
{
    if (!check_channel_name(name))
        return;

    header_.channels().insert(name, Imf::Channel(Imf::FLOAT));
    fb_.insert(name, Imf::Slice(Imf::FLOAT, (char*)img.data(0,0), sizeof(float), img.pitch()));
}


#ifdef IUIO_EIGEN3
void OpenEXROutputFile::add_attribute(const std::string &name, Eigen::Ref<Eigen::Matrix3f> mat)
{
    if (!check_attachement_name(name))
        return;

    Imath::M33f attr(mat(0,0), mat(0,1), mat(0,2),
                     mat(1,0), mat(1,1), mat(1,2),
                     mat(2,0), mat(2,1), mat(2,2));

    header_.insert(name, Imf::M33fAttribute(attr));
}

void OpenEXROutputFile::add_attribute(const std::string &name, Eigen::Ref<Eigen::Matrix4f> mat)
{
    if (!check_attachement_name(name))
        return;

    Imath::M44f attr(mat(0,0), mat(0,1), mat(0,2), mat(0,3),
                     mat(1,0), mat(1,1), mat(1,2), mat(1,3),
                     mat(2,0), mat(2,1), mat(2,2), mat(2,3),
                     mat(3,0), mat(3,1), mat(3,2), mat(3,3));

    header_.insert(name, Imf::M44fAttribute(attr));

}
#endif

void OpenEXROutputFile::write()
{

    if (fb_.begin() == fb_.end())
    {
        printf("OpenEXR write(): framebuffer of %s is empty, call add_channel() before writing\n", filename_.c_str());
        return;
    }

    Imf::OutputFile outfile_(filename_.c_str(), header_);
    outfile_.setFrameBuffer(fb_);
    outfile_.writePixels(sz_.height);
}

bool OpenEXROutputFile::check_channel_name(const std::string &name)
{
    if (header_.channels().findChannel(name))
    {
        printf("OpenEXR: channel %s already exists in %s!\n", name.c_str(), filename_.c_str());
        return false;
    }
    return true;
}

bool OpenEXROutputFile::check_attachement_name(const std::string &name)
{
    try
    {
        header_[name];
    }
    catch (Iex::ArgExc)
    {
        return true;
    }
    printf("OpenEXR: attachement %s already exists in %s!\n", name.c_str(), filename_.c_str());
    return false;
}

#endif


} // namespace iu
