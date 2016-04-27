#pragma once

#include <cuda_runtime_api.h>
#include "lineardevicememory.h"

namespace iu
{

template<typename PixelType>
class TensorGpu: public LinearDeviceMemory<PixelType>
{
public:
	TensorGpu() :
			LinearDeviceMemory<PixelType>(), samples_(0), channels_(0), height_(0), width_(0)
	{
	}

	virtual ~TensorGpu()
	{
	}

	TensorGpu(const unsigned int N, const unsigned int C, const unsigned int H, const unsigned int W) :
			LinearDeviceMemory<PixelType>(N * C * H * W), samples_(N), channels_(C), height_(H), width_(W)
	{
	}

	TensorGpu(const TensorGpu<PixelType>& from) :
			LinearHostMemory<PixelType>(from), samples_(from.samples_), channels_(from.channels_), height_(from.height_), width_(
					from.width_)
	{
	}

	TensorGpu(PixelType* device_data, const unsigned int N, const unsigned int C, const unsigned int H,
			const unsigned int W, bool ext_data_pointer = false) :
			LinearDeviceMemory<PixelType>(device_data, N * C * H * W, ext_data_pointer), samples_(N), channels_(C), height_(H), width_(
					W)
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

private:
	unsigned int samples_;
	unsigned int channels_;
	unsigned int height_;
	unsigned int width_;

};

}
