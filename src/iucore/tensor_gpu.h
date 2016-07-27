#pragma once

#include "lineardevicememory.h"

template<typename, int> class ndarray_ref;

namespace iu
{
/**  \brief Device 4D tensor class.
 *   \ingroup LinearMemory
 */
template<typename PixelType>
class TensorGpu: public LinearDeviceMemory<PixelType>
{
public:
  /** \brief Memory layout to access the data elements.
   *
   *  Defines how the elements are laid out in the memory.
   * - NCHW: Samples - channels - height - width
   * - NHWC: Samples - height - width - channels
   */
	enum MemoryLayout
	{
		NCHW, NHWC
	};

  /** Constructor.
   *  @param memoryLayout MemoryLayout
   * */
	TensorGpu(MemoryLayout memoryLayout=NCHW) :
			LinearDeviceMemory<PixelType>(), samples_(0), channels_(0), height_(0), width_(0), memoryLayout_(
					memoryLayout)
	{
	}

	/** Destructor.*/
	virtual ~TensorGpu()
	{
	}

  /** Special constructor.
   *  @param N Number of samples
   *  @param C Number of channels
   *  @param H Height
   *  @param W Width
   *  @param memoryLayout MemoryLayout
   */
	TensorGpu(const unsigned int N, const unsigned int C, const unsigned int H, const unsigned int W, MemoryLayout memoryLayout = NCHW) :
			LinearDeviceMemory<PixelType>(N * C * H * W), samples_(N), channels_(C), height_(H), width_(W), memoryLayout_(
					memoryLayout)
	{
	}

  /** Special constructor.
   *  @param device_data Device data pointer
   *  @param N Number of samples
   *  @param C Number of channels
   *  @param H Height
   *  @param W Width
   *  @param ext_data_pointer Use external data pointer as internal data pointer
   *  @param memoryLayout MemoryLayout
   */
	TensorGpu(PixelType* device_data, const unsigned int N, const unsigned int C, const unsigned int H,
			const unsigned int W, bool ext_data_pointer = false, MemoryLayout memoryLayout = NCHW) :
			LinearDeviceMemory<PixelType>(device_data, N * C * H * W, ext_data_pointer), samples_(N), channels_(C), height_(
					H), width_(W), memoryLayout_(memoryLayout)
	{
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

	/** Returns height. */
	unsigned int height() const
	{
		return height_;
	}

  /** Returns width. */
	unsigned int width() const
	{
		return width_;
	}

  /** Returns MemoryLayout. */
	MemoryLayout memoryLayout() const
	{
		return memoryLayout_;
	}

  /** Operator<< overloading. Output of TensorGpu class. */
  friend std::ostream& operator<<(std::ostream & out,
                                  TensorGpu const& tensor)
  {
    out << "Tensor: height=" << tensor.height() << " width="
        << tensor.width() << " samples="  << tensor.samples() << " channels="
        << tensor.channels();
    if (tensor.memoryLayout() == NCHW)
      out << " memory_layout=NCHW";
    else if(tensor.memoryLayout() == NHWC)
      out << " memory_layout=NHWC";
    out << " onDevice=" << tensor.onDevice();
    return out;
  }

  /** \brief Struct pointer TensorKernelData that can be used in CUDA kernels.
   *
   *  This struct provides the device data pointer as well as important class
   *  properties.
   */
	struct TensorKernelData
	//struct KernelData
	{
	  /** Pointer to device buffer. */
		PixelType* data_;
	  /** Length of the memory.*/
		unsigned int length_;
	  /** Stride in first dimension.*/
		unsigned int stride0;
		/** Stride in second dimension.*/
		unsigned int stride1;
		/** Stride in third dimension.*/
		unsigned int stride2;

		/** Number of samples.*/
		unsigned short N;
    /** Number of channels.*/
		unsigned short C;
		/** Height. */
		unsigned short H;
		/** Width. */
		unsigned short W;

    /** Access the image via the () operator according to MemoryLayout.
     * @param pos0 Position in the first dimension.
     * @param pos1 Position in the second dimension.
     * @param pos2 Position in the third dimension.
     * @param pos3 Position in the forth dimension.
     * @return value at position (pos0, pos1, pos2, pos3).
     */
		__device__ PixelType& operator()(short pos0, short pos1, short pos2, short pos3)
		{
			return data_[pos0 * stride0 + pos1 * stride1 + pos2 * stride2 + pos3];
		}

    /** Get position / coordinates for a linear index.
     * @param[in] linearIdx Linear index.
     * @param[out] dim0 Position in the first dimension.
     * @param[out] dim1 Position in the second dimension.
     * @param[out] dim2 Position in the third dimension.
     * @param[out] dim3 Position in the forth dimension.
     */
		__device__ void coords(unsigned int linearIdx, short *dim0, short *dim1, short *dim2, short *dim3)
		{
			*dim0 = linearIdx / stride0;
			*dim1 = (linearIdx % stride0) / stride1;
			*dim2 = ((linearIdx % stride0) % stride1) / stride2;
			*dim3 = ((linearIdx % stride0) % stride1) % stride2;
		}

		/** Constructor */
		__host__ TensorKernelData(const TensorGpu<PixelType> &tensor) :
		//__host__ KernelData(const TensorGpu<PixelType> &tensor) :
				data_(const_cast<PixelType*>(tensor.data())), length_(tensor.length()), N(tensor.samples()), C(tensor.channels()),
				H(tensor.height()), W(tensor.width())
		{
			if (tensor.memoryLayout() == NCHW)
			{
				stride0 = tensor.channels() * tensor.height() * tensor.width();
				stride1 = tensor.height() * tensor.width();
				stride2 = tensor.width();
			}
			else if (tensor.memoryLayout() == NHWC)
			{
				stride0 = tensor.height() * tensor.width() * tensor.channels();
				stride1 = tensor.width() * tensor.channels();
				stride2 = tensor.channels();
			}
		}
	};

	/** convert to ndarray_ref -- include ndarray/ndarray_iu.h*/
	ndarray_ref<PixelType,1> ref() const;

	/** construct from ndarray_ref  -- include ndarray/ndarray_iu.h*/
	TensorGpu(const ndarray_ref<PixelType,1> &x);

private:
  /** Number of samples. */
	unsigned int samples_;
  /** Number of channels. */
	unsigned int channels_;
  /** Height. */
	unsigned int height_;
  /** Width. */
	unsigned int width_;
  /** MemoryLayout */
	MemoryLayout memoryLayout_;

private:
  /** Private copy constructor. */
  TensorGpu(const TensorGpu&);
  /** Private copy assignment operator. */
  TensorGpu& operator=(const TensorGpu&);
};

}  // namespace iu
