#pragma once
#include "linearhostmemory.h"

template<typename, int> class ndarray_ref;

namespace iu
{
/**  \brief Host 4D tensor class.
 *   \ingroup LinearMemory
 */
template<typename PixelType>
class TensorCpu : public LinearHostMemory<PixelType, 1>
{
public:
  /** Constructor. */
	TensorCpu() :
			LinearHostMemory<PixelType, 1>(), samples_(0), channels_(0), height_(0), width_(0)
	{
	}

  /** Destructor. */
	virtual ~TensorCpu()
	{
	}

  /** Special constructor.
   *  @param N Number of samples
   *  @param C Number of channels
   *  @param H Height
   *  @param W Width
   */
	TensorCpu(const unsigned int N, const unsigned int C, const unsigned int H, const unsigned int W) :
			LinearHostMemory<PixelType, 1>(N * C * H * W), samples_(N), channels_(C), height_(H), width_(W)
	{
	}

  /** Special constructor.
   *  @param host_data Host data pointer
   *  @param N Number of samples
   *  @param C Number of channels
   *  @param H Height
   *  @param W Width
   *  @param ext_data_pointer Use external data pointer as internal data pointer
   */
	TensorCpu(PixelType* host_data, const unsigned int N, const unsigned int C, const unsigned int H,
			const unsigned int W, bool ext_data_pointer = false) :
			LinearHostMemory<PixelType, 1>(host_data, N * C * H * W, ext_data_pointer), samples_(N), channels_(C), height_(H), width_(
					W)
	{
		/** \todo Check RGB inputs */
	}


	/** Get Pixel value at position x,y. */
	PixelType getPixel(unsigned int n, unsigned int c, unsigned int x, unsigned int y)
	{
		return *this->data(n * (channels_ * height_ * width_) + c * (width_ * height_) + x * width_ + y);
	}

	/** Returns the number of samples. */
	unsigned int samples() const
	{
		return samples_;
	}

  /** Returns the number of channels. */
	unsigned int channels() const
	{
		return channels_;
	}

  /** Returns the height. */
	unsigned int height() const
	{
		return height_;
	}

	/** Returns the width. */
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

  /** convert to ndarray_ref -- include ndarray/ndarray_iu.h*/
  ndarray_ref<PixelType,4> ref() const;

  /** construct from ndarray_ref  -- include ndarray/ndarray_iu.h*/
  TensorCpu(const ndarray_ref<PixelType,4> &x);

private:
  /** Number of samples. */
	unsigned int samples_;
	/** Number of channels. */
	unsigned int channels_;
	/** Height. */
	unsigned int height_;
	/** Width. */
	unsigned int width_;

private:
  /** Private copy constructor. */
  TensorCpu(const TensorCpu&);
  /** Private copy assignment operator. */
  TensorCpu& operator=(const TensorCpu&);
};

} // namespace iu

