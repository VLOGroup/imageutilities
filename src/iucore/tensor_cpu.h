#pragma once
#include "linearhostmemory.h"

namespace iu
{

template<typename PixelType>
class TensorCpu : public LinearHostMemory<PixelType>
{
public:
	// wrap LinearHostMemory constructors and initialize members
	TensorCpu() :
			LinearHostMemory<PixelType>(), samples_(0), channels_(0), height_(0), width_(0)
	{
	}

	virtual ~TensorCpu()
	{
	}

	TensorCpu(const unsigned int N, const unsigned int C, const unsigned int H, const unsigned int W) :
			LinearHostMemory<PixelType>(N * C * H * W), samples_(N), channels_(C), height_(H), width_(W)
	{
	}

	TensorCpu(PixelType* host_data, const unsigned int N, const unsigned int C, const unsigned int H,
			const unsigned int W, bool ext_data_pointer = false) :
			LinearHostMemory<PixelType>(host_data, N * C * H * W, ext_data_pointer), samples_(N), channels_(C), height_(H), width_(
					W)
	{
		// TODO: Check RGB inputs
	}

	TensorCpu(const TensorCpu<PixelType>& from) :
			LinearHostMemory<PixelType>(from), samples_(from.samples_), channels_(from.channels_), height_(from.height_), width_(
					from.width_)
	{
	}

	/** Get Pixel value at position x,y. */
	PixelType getPixel(unsigned int n, unsigned int c, unsigned int x, unsigned int y)
	{
		return *this->data(n * (channels_ * height_ * width_) + c * (width_ * height_) + x * width_ + y);
	}

	/** Returns the number of elements saved in the device buffer. (length of device buffer) */
	unsigned int samples() const
	{
		return samples_;
	}

	unsigned int channels() const
	{
		return channels_;
	}

	unsigned int height() const
	{
		return height_;
	}

	unsigned int width() const
	{
		return width_;
	}

  /** Operator<< overloading. Output of TensorCpu class. */
  friend std::ostream& operator<<(std::ostream & out,
                                  TensorCpu const& tensor)
  {
    out << "Tensor: height=" << tensor.height() << " width="
        << tensor.width() << " samples="  << tensor.samples() << " channels="
        << tensor.channels() << " onDevice=" << tensor.onDevice();
    return out;
  }

private:
	unsigned int samples_;
	unsigned int channels_;
	unsigned int height_;
	unsigned int width_;

};

} // namespace iu

