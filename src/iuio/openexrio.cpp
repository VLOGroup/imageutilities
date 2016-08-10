#include "openexrio.h"


using std::string;

namespace iu {

OpenEXRInputFile::OpenEXRInputFile(const std::string &filename)
{
    filename_ = filename;
    Imf::InputFile file(filename.c_str());
    Imath::Box2i dw = file.header().dataWindow();

    pixeltype_to_string_[Imf::UINT] = "uint";
    pixeltype_to_string_[Imf::HALF] = "half (16bit float)";
    pixeltype_to_string_[Imf::FLOAT] = "float";

    int width = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;
    sz_ = iu::Size<2>(width, height);

    channels_.clear();
    Imf::ChannelList cl = file.header().channels();
    Imf::ChannelList::Iterator it;
    for (it=cl.begin(); it != cl.end(); it++)
    {
        switch(it.channel().type)
        {
        case Imf::UINT:
            channels_.push_back(Channel(it.name(), pixeltype_to_string_[Imf::UINT]));
            break;
        case Imf::HALF:
            printf("OpenEXRFile: encountered channel of dataype HALF in file %s, corresponding datatype (16 bit float) not implemented in imageutilities!\n", it.name());
            break;
        case Imf::FLOAT:
            channels_.push_back(Channel(it.name(), pixeltype_to_string_[Imf::FLOAT]));
            break;
        case Imf::NUM_PIXELTYPES:   // silence warning
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

    unsigned int width = dw.max.x - dw.min.x + 1;
    unsigned int height = dw.max.y - dw.min.y + 1;

    if (img.width() != width || img.height() != height)
    {
        printf("OpenEXRFile get_channel(): image sizes do not match!\n");
        return;
    }

    if (!file.header().channels().findChannel(name.c_str()))
    {
        printf("OpenEXRFile get_channel(): couldn't find channel %s in exr file!\n", name.c_str());
        return;
    }

    if (file.header().channels().findChannel(name.c_str())->type != Imf::UINT)
    {
        printf("OpenEXRFile get_channel(): channel %s does not contain UINT data!\n", name.c_str());
        return;
    }

    Imf::FrameBuffer fb;

    fb.insert(name.c_str(), Imf::Slice(Imf::UINT, (char*)img.data(0,0), sizeof(unsigned int), img.pitch()));
    file.setFrameBuffer(fb);
    file.readPixels(dw.min.y, dw.max.y);
}

void OpenEXRInputFile::read_channel(const std::string &name, ImageCpu_32f_C1 &img)
{
    Imf::InputFile file(filename_.c_str());
    Imath::Box2i dw = file.header().dataWindow();

    unsigned int width = dw.max.x - dw.min.x + 1;
    unsigned int height = dw.max.y - dw.min.y + 1;

    if (img.width() != width || img.height() != height)
    {
        printf("OpenEXRFile read_channel(): image sizes do not match!\n");
        return;
    }

    if (!file.header().channels().findChannel(name.c_str()))
    {
        printf("OpenEXRFile read_channel(): couldn't find channel %s in %s!\n", name.c_str(), filename_.c_str());
        return;
    }

    if (file.header().channels().findChannel(name.c_str())->type != Imf::FLOAT)
    {
        printf("OpenEXRFile read_channel(): channel %s does not contain FLOAT data!\n", name.c_str());
        return;
    }

    Imf::FrameBuffer fb;

    fb.insert(name.c_str(), Imf::Slice(Imf::FLOAT, (char*)img.data(0,0), sizeof(float), img.pitch()));
    file.setFrameBuffer(fb);
    file.readPixels(dw.min.y, dw.max.y);

}

void OpenEXRInputFile::read_channel_32f(const std::string &name, ImageCpu_32f_C1 &img)
{
    Imf::InputFile file(filename_.c_str());
    Imath::Box2i dw = file.header().dataWindow();

    unsigned int width = dw.max.x - dw.min.x + 1;
    unsigned int height = dw.max.y - dw.min.y + 1;

    if (img.width() != width || img.height() != height)
    {
        printf("OpenEXRFile read_channel(): image sizes do not match!\n");
        return;
    }

    if (!file.header().channels().findChannel(name.c_str()))
    {
        printf("OpenEXRFile read_channel(): couldn't find channel %s in %s!\n", name.c_str(), filename_.c_str());
        return;
    }

    if (file.header().channels().findChannel(name.c_str())->type == Imf::UINT)
    {
        iu::ImageCpu_32u_C1 h_img32u(img.size());
        this->read_channel(name, h_img32u);
        iu::convert_32u32f_C1(&h_img32u, &img, 1);
        return;
    }
    else if (file.header().channels().findChannel(name.c_str())->type == Imf::FLOAT)
    {
        Imf::FrameBuffer fb;
        fb.insert(name.c_str(), Imf::Slice(Imf::FLOAT, (char*)img.data(0,0), sizeof(float), img.pitch()));
        file.setFrameBuffer(fb);
        file.readPixels(dw.min.y, dw.max.y);
    }
}


void OpenEXRInputFile::read_channel(const std::string &name, ImageGpu_32u_C1 &img)
{
    ImageCpu_32u_C1 h_img(img.size());
    this->read_channel(name, h_img);
    iu::copy(&h_img, &img);
}



void OpenEXRInputFile::read_channel(const std::string& name, ImageGpu_32f_C1 &img)
{
    ImageCpu_32f_C1 h_img(img.size());
    this->read_channel(name, h_img);
    iu::copy(&h_img, &img);
}

void OpenEXRInputFile::read_channel_32f(const std::string &name, ImageGpu_32f_C1 &img)
{
    ImageCpu_32f_C1 h_img(img.size());
    this->read_channel_32f(name, h_img);
    iu::copy(&h_img, &img);
}



#ifdef IUIO_EIGEN3
void OpenEXRInputFile::read_attribute(const std::string &name, Eigen::Ref<Eigen::Matrix3f> mat)
{
    Imf::InputFile file(filename_.c_str());
    const Imf::M33fAttribute *m33Attr = file.header().findTypedAttribute<Imf::M33fAttribute>(name.c_str());

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
    const Imf::M44fAttribute *m44Attr = file.header().findTypedAttribute<Imf::M44fAttribute>(name.c_str());

    if (!m44Attr)
    {
        printf("Error: Attribute %s (4x4) not found in %s!\n", name.c_str(), filename_.c_str());
        return;
    }


    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            mat(i,j) = m44Attr->value().x[i][j];

}
#endif

OpenEXROutputFile::OpenEXROutputFile(const std::string &filename, iu::Size<2> size)
{
    filename_ = filename;
    sz_ = size;
    header_ = Imf::Header(sz_.width, sz_.height);
    pool_32f_C1_.clear();
    pool_32f_C2_.clear();
    pool_32f_C4_.clear();
}

OpenEXROutputFile::~OpenEXROutputFile()
{
//    for (iu::ImageCpu_32f_C1* img : pool_32f_C1_)
//        delete img;
//    for (iu::ImageCpu_32f_C2* img : pool_32f_C2_)
//        delete img;
    for (unsigned int i=0; i < pool_32f_C1_.size(); i++)
        delete pool_32f_C1_.at(i);
    for (unsigned int i=0; i < pool_32f_C2_.size(); i++)
        delete pool_32f_C2_.at(i);
    for (unsigned int i=0; i < pool_32f_C4_.size(); i++)
        delete pool_32f_C4_.at(i);

    pool_32f_C1_.clear();
    pool_32f_C2_.clear();
    pool_32f_C4_.clear();
}


void OpenEXROutputFile::add_channel(const std::string &name, ImageCpu_8u_C1 &img)
{
    if (!check_channel_name(name))
        return;

    header_.channels().insert(name.c_str(), Imf::Channel(Imf::UINT));

    iu::ImageCpu_32u_C1 temp(img.size());
    for (unsigned int y=0; y < img.height(); y++)
    {
        for (unsigned int x=0; x < img.width(); x++)
        {
            *temp.data(x,y) = *img.data(x,y);
        }
    }
    fb_.insert(name.c_str(), Imf::Slice(Imf::UINT, (char*)temp.data(0,0), sizeof(unsigned int), temp.pitch()));
}



void OpenEXROutputFile::add_channel(const std::string &name, ImageCpu_32u_C1 &img)
{
    if (!check_channel_name(name))
        return;

    header_.channels().insert(name.c_str(), Imf::Channel(Imf::UINT));
    fb_.insert(name.c_str(), Imf::Slice(Imf::UINT, (char*)img.data(0,0), sizeof(unsigned int), img.pitch()));
}



void OpenEXROutputFile::add_channel(const std::string &name, ImageCpu_32f_C1 &img)
{
    if (!check_channel_name(name))
        return;

    header_.channels().insert(name.c_str(), Imf::Channel(Imf::FLOAT));
    fb_.insert(name.c_str(), Imf::Slice(Imf::FLOAT, (char*)img.data(0,0), sizeof(float), img.pitch()));
}



void OpenEXROutputFile::add_channel(const std::string &name1, const std::string &name2, ImageCpu_32f_C2 &img)
{
    if ( !(check_channel_name(name1) && check_channel_name(name2)) )
        return;

    header_.channels().insert(name1.c_str(), Imf::Channel(Imf::FLOAT));
    header_.channels().insert(name2.c_str(), Imf::Channel(Imf::FLOAT));

    fb_.insert(name1.c_str(), Imf::Slice(Imf::FLOAT, (char*)&(img.data(0,0)->x), sizeof(float2), img.pitch()));
    fb_.insert(name2.c_str(), Imf::Slice(Imf::FLOAT, (char*)&(img.data(0,0)->y), sizeof(float2), img.pitch()));
}

void OpenEXROutputFile::add_channel(const std::string &name1, const std::string &name2,
                                    const std::string &name3, const std::string &name4, ImageCpu_32f_C4 &img)
{
    if ( !(check_channel_name(name1) && check_channel_name(name2) &&
           check_channel_name(name3) && check_channel_name(name4)) )
        return;

    header_.channels().insert(name1.c_str(), Imf::Channel(Imf::FLOAT));
    header_.channels().insert(name2.c_str(), Imf::Channel(Imf::FLOAT));
    header_.channels().insert(name3.c_str(), Imf::Channel(Imf::FLOAT));
    header_.channels().insert(name4.c_str(), Imf::Channel(Imf::FLOAT));

    fb_.insert(name1.c_str(), Imf::Slice(Imf::FLOAT, (char*)&(img.data(0,0)->x), sizeof(float4), img.pitch()));
    fb_.insert(name2.c_str(), Imf::Slice(Imf::FLOAT, (char*)&(img.data(0,0)->y), sizeof(float4), img.pitch()));
    fb_.insert(name3.c_str(), Imf::Slice(Imf::FLOAT, (char*)&(img.data(0,0)->z), sizeof(float4), img.pitch()));
    fb_.insert(name4.c_str(), Imf::Slice(Imf::FLOAT, (char*)&(img.data(0,0)->w), sizeof(float4), img.pitch()));
}




void OpenEXROutputFile::add_channel(const std::string &name, ImageGpu_32f_C1 &img)
{
    iu::ImageCpu_32f_C1 *h_img = new iu::ImageCpu_32f_C1(img.size());
    iu::copy(&img, h_img);
    pool_32f_C1_.push_back(h_img);
    this->add_channel(name, *h_img);
}



void OpenEXROutputFile::add_channel(const std::string &name1, const std::string &name2, ImageGpu_32f_C2 &img)
{
    iu::ImageCpu_32f_C2 *h_img = new iu::ImageCpu_32f_C2(img.size());
    iu::copy(&img, h_img);
    pool_32f_C2_.push_back(h_img);
    this->add_channel(name1, name2, *h_img);
}



void OpenEXROutputFile::add_channel(const std::string &name1, const std::string &name2,
                                    const std::string &name3, const std::string &name4, ImageGpu_32f_C4 &img)
{
    iu::ImageCpu_32f_C4 *h_img = new iu::ImageCpu_32f_C4(img.size());
    iu::copy(&img, h_img);
    pool_32f_C4_.push_back(h_img);
    this->add_channel(name1, name2, name3, name4, *h_img);

}





#ifdef IUIO_EIGEN3
void OpenEXROutputFile::add_attribute(const std::string &name, Eigen::Ref<Eigen::Matrix3f> mat)
{
    if (!check_attachement_name(name))
        return;

    Imath::M33f attr(mat(0,0), mat(0,1), mat(0,2),
                     mat(1,0), mat(1,1), mat(1,2),
                     mat(2,0), mat(2,1), mat(2,2));

    header_.insert(name.c_str(), Imf::M33fAttribute(attr));
}

void OpenEXROutputFile::add_attribute(const std::string &name, Eigen::Ref<Eigen::Matrix4f> mat)
{
    if (!check_attachement_name(name))
        return;

    Imath::M44f attr(mat(0,0), mat(0,1), mat(0,2), mat(0,3),
                     mat(1,0), mat(1,1), mat(1,2), mat(1,3),
                     mat(2,0), mat(2,1), mat(2,2), mat(2,3),
                     mat(3,0), mat(3,1), mat(3,2), mat(3,3));

    header_.insert(name.c_str(), Imf::M44fAttribute(attr));

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
    if (header_.channels().findChannel(name.c_str()))
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
        header_[name.c_str()];
    }
    catch (Iex::ArgExc)
    {
        return true;
    }
    printf("OpenEXR: attachement %s already exists in %s!\n", name.c_str(), filename_.c_str());
    return false;
}


} // namespace iu
