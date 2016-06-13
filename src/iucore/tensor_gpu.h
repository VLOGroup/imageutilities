#pragma once

//#include <cuda_runtime_api.h>
#include "lineardevicememory.h"

namespace iu
{

template<typename PixelType>
class TensorGpu: public LinearDeviceMemory<PixelType>
{
public:
	enum MemoryLayout
	{
		NCHW, NHWC
	};

	TensorGpu(MemoryLayout memoryLayout=NCHW) :
			LinearDeviceMemory<PixelType>(), samples_(0), channels_(0), height_(0), width_(0), memoryLayout_(
					memoryLayout)
	{
	}

	virtual ~TensorGpu()
	{
	}

	TensorGpu(const unsigned int N, const unsigned int C, const unsigned int H, const unsigned int W, MemoryLayout memoryLayout = NCHW) :
			LinearDeviceMemory<PixelType>(N * C * H * W), samples_(N), channels_(C), height_(H), width_(W), memoryLayout_(
					memoryLayout)
	{
	}

	TensorGpu(const TensorGpu<PixelType>& from) :
            LinearDeviceMemory<PixelType>(from), samples_(from.samples_), channels_(from.channels_), height_(
					from.height_), width_(from.width_), memoryLayout_(from.memoryLayout_)
	{
	}

	TensorGpu(PixelType* device_data, const unsigned int N, const unsigned int C, const unsigned int H,
			const unsigned int W, bool ext_data_pointer = false, MemoryLayout memoryLayout = NCHW) :
			LinearDeviceMemory<PixelType>(device_data, N * C * H * W, ext_data_pointer), samples_(N), channels_(C), height_(
					H), width_(W), memoryLayout_(memoryLayout)
	{
	}

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

	struct TensorKernelData
	//struct KernelData
	{
		PixelType* data_;
		unsigned int length_;
		unsigned int stride0;
		unsigned int stride1;
		unsigned int stride2;

		unsigned short N;
		unsigned short C;
		unsigned short H;
		unsigned short W;


		__device__ PixelType& operator()(short pos0, short pos1, short pos2, short pos3)
		{
			return data_[pos0 * stride0 + pos1 * stride1 + pos2 * stride2 + pos3];
		}

		__device__ void coords(unsigned int linearIdx, short *dim0, short *dim1, short *dim2, short *dim3)
		{
			*dim0 = linearIdx / stride0;
			*dim1 = (linearIdx % stride0) / stride1;
			*dim2 = ((linearIdx % stride0) % stride1) / stride2;
			*dim3 = ((linearIdx % stride0) % stride1) % stride2;
		}

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

private:
	unsigned int samples_;
	unsigned int channels_;
	unsigned int height_;
	unsigned int width_;

	MemoryLayout memoryLayout_;

};

}
