//#include "ndarray_ref.h"
#include "ndarray_ref.host.h"
//
#include "../iucore.h"
#include "../iucore/image_cpu.h"
#include "../iucore/image_gpu.h"
#include "../iucore/volume_cpu.h"
#include "../iucore/volume_gpu.h"
#include "../iucore/memorydefs.h"
#include "../iucore/tensor_cpu.h"
#include "../iucore/tensor_gpu.h"
//
#include "type_expand_cuda.h"
//! implicit / explicit conversions between iu classes and ndarray_ref

namespace special2{
	//_______1D_____________
	template<typename type>
	ndarray_ref<type, 1> & ndarray_ref<type, 1>::set_ref(const iu::LinearDeviceMemory<type> & x){
		this->set_linear_ref(const_cast<type*>(x.data()), x.length(), ndarray_flags::device_only);
		return *this;
	}

	template<typename type>
	ndarray_ref<type, 1> & ndarray_ref<type, 1>::set_ref(const iu::LinearHostMemory<type> & x){
		this->set_linear_ref(const_cast<type*>(x.data()), x.length(), ndarray_flags::host_only);
		return *this;
	}

	//_______2D_____________
	template<typename type>
	template<class Allocator>
	ndarray_ref<type, 2> & ndarray_ref<type, 2>::set_ref(const iu::ImageCpu<type, Allocator> & x){
		intn<2> size(x.width(), x.height());
		intn<2> stride_bytes = intn<2>(x.data(1, 0) - x.data(0, 0), x.data(0, 1) - x.data(0, 0))*intsizeof(type);
		set_ref(const_cast<type*>(x.data(0, 0)), size, stride_bytes, ndarray_flags::host_only);
		return *this;
	}

	template<typename type>
	template<class Allocator>
	ndarray_ref<type, 2> & ndarray_ref<type, 2>::set_ref(const iu::ImageGpu<type, Allocator> & x){
		intn<2> size(x.width(), x.height());
		intn<2> stride_bytes = intn<2>(x.data(1, 0) - x.data(0, 0), x.data(0, 1) - x.data(0, 0))*intsizeof(type);
		set_ref(const_cast<type*>(x.data(0, 0)), size, stride_bytes, ndarray_flags::device_only);
		return *this;
	}

	//_______3D_____________
	template<typename type>
	template<class Allocator>
	ndarray_ref<type, 3> & ndarray_ref<type, 3>::set_ref(const iu::VolumeCpu<type, Allocator> & x){
		intn<3> size(x.width(), x.height(), x.depth());
		intn<3> stride_bytes = intn<3>(x.data(1, 0, 0) - x.data(0, 0, 0), x.data(0, 1, 0) - x.data(0, 0, 0), x.data(0, 0, 1) - x.data(0, 0, 0))*intsizeof(type);
		this->set_ref(const_cast<type*>(x.data(0, 0, 0)), size, stride_bytes, ndarray_flags::host_only);
		return *this;
	}

	template<typename type>
	template<class Allocator>
	ndarray_ref<type, 3> & ndarray_ref<type, 3>::set_ref(const iu::VolumeGpu<type, Allocator> & x){
		intn<3> size(x.width(), x.height(), x.depth());
		intn<3> stride_bytes = intn<3>(x.data(1, 0, 0) - x.data(0, 0, 0), x.data(0, 1, 0) - x.data(0, 0, 0), x.data(0, 0, 1) - x.data(0, 0, 0))*intsizeof(type);
		this->set_ref(const_cast<type*>(x.data(0, 0, 0)), size, stride_bytes, ndarray_flags::device_only);
		return *this;
	}

	//_______4D_____________
	template<typename type>
	ndarray_ref<type, 4> & ndarray_ref<type, 4>::set_ref(const iu::TensorCpu<type> & t){
		// according to TensorCpu:: getPixel(unsigned int n, unsigned int c, unsigned int x, unsigned int y)
		intn<4> size(t.samples(), t.channels(), t.height(), t.width());
		this->template set_linear_ref<false>(const_cast<type*>(t.data()), size, ndarray_flags::host_only); //assume TensorCpu has descending strides
		return *this;
	}

	template<typename type>
	ndarray_ref<type, 4> & ndarray_ref<type, 4>::set_ref(const iu::TensorGpu<type> & t){
		intn<4> size;
		if (t.memoryLayout() == iu::TensorGpu<type>::NCHW){
			size = intn<4>(t.samples(), t.channels(), t.height(), t.width());
		} else if (t.memoryLayout() == iu::TensorGpu<type>::NHWC){
			size = intn<4>(t.samples(), t.height(), t.width(), t.channels());
		} else{
			slperror("unknown TensorGPU shape");
		};
		this->template set_linear_ref<false>(const_cast<type*>(t.data()), size, ndarray_flags::device_only); //assume TensorGpu has descending strides
		return *this;
	}

}

template<typename type, int dims>
ndarray_ref<type,dims>::ndarray_ref(const iu::LinearHostMemory<type> & x, const intn<dims> & size){
	this->set_linear_ref(const_cast<type*>(x.data()), size, ndarray_flags::host_only);
}

template<typename type, int dims>
ndarray_ref<type,dims>::ndarray_ref(const iu::LinearDeviceMemory <type>& x, const intn<dims> & size){
	this->set_linear_ref(const_cast<type*>(x.data()), size, ndarray_flags::device_only);
}

template<typename type, int dims>
ndarray_ref<type,dims>::ndarray_ref(const iu::LinearHostMemory<type> * x, const intn<dims> & size){
	this->set_linear_ref(const_cast<type*>(x.data()), size, ndarray_flags::host_only);
}

template<typename type, int dims>
ndarray_ref<type,dims>::ndarray_ref(const iu::LinearDeviceMemory<type> * x, const intn<dims> & size){
	this->set_linear_ref(const_cast<type*>(x.data()), size, ndarray_flags::device_only);
}

namespace iu{

	//1D
	template<typename type>
	ndarray_ref<type,1> LinearHostMemory<type>::ref() const{
		return ndarray_ref<type,1>(*this);
	}

	/** construct from ndarray_ref*/
	template<typename type>
	LinearHostMemory<type>::LinearHostMemory(const ndarray_ref<type,1> &x)
	:LinearMemory(x.numel()),ext_data_pointer_(true){
		runtime_check(x.host_allowed());
		runtime_check(x.stride_bytes(0)==sizeof(type)); //contiguous in width
		data_ = x.ptr();
	}

	template<typename type>
	ndarray_ref<type,1> LinearDeviceMemory<type>::ref() const{
		return ndarray_ref<type,1>(*this);
	}

	/** construct from ndarray_ref*/
	template<typename type>
	LinearDeviceMemory<type>::LinearDeviceMemory(const ndarray_ref<type,1> &x)
	:LinearMemory(x.numel()),ext_data_pointer_(true){
		runtime_check(x.device_allowed());
		runtime_check(x.stride_bytes(0)==sizeof(type)); //contiguous in width
		data_ = x.ptr();
	}

	//2D
	//-------ImageCpu----------
	template<typename type, class Allocator>
	ndarray_ref<type,2> ImageCpu<type, Allocator>::ref() const{
		return ndarray_ref<type,2>(*this);
	}

	/** construct from ndarray_ref*/
	template<typename type, class Allocator>
	ImageCpu<type, Allocator>::ImageCpu(const ndarray_ref<type,2> &x):Image(x.size(0), x.size(1)), ext_data_pointer_(true), data_(0){
		runtime_check(x.host_allowed());
		runtime_check(x.stride_bytes(0)==sizeof(type)); //contiguous in width
		data_ = x.ptr();
		pitch_ = x.stride_bytes(1);
	}
	//------ImageGpu-----------
	template<typename type, class Allocator>
	ndarray_ref<type,2> ImageGpu<type, Allocator>::ref() const{
		return ndarray_ref<type,2>(*this);
	}

	/** construct from ndarray_ref*/
	template<typename type, class Allocator>
	ImageGpu<type, Allocator>::ImageGpu(const ndarray_ref<type,2> &x):Image(x.size(0), x.size(1)), ext_data_pointer_(true), data_(0){
		runtime_check(x.device_allowed());
		runtime_check(x.stride_bytes(0)==sizeof(type)); //contiguous in width
		data_ = x.ptr();
		pitch_ = x.stride_bytes(1);
	}
	//3D__________________
	template<typename type, class Allocator>
	ndarray_ref<type,3> VolumeCpu<type, Allocator>::ref() const{
		return ndarray_ref<type,3>(*this);
	}

	template<typename type, class Allocator>
	VolumeCpu<type, Allocator>::VolumeCpu(const ndarray_ref<type,3> &x)
	:Volume(size(0), size(1), size(2)), ext_data_pointer_(true){
		runtime_check(x.host_allowed());
		runtime_check(x.stride_bytes(0)==sizeof(type)); //contiguous in width
		data_ = x.ptr();
		pitch_ = x.stride_bytes(1);
	}

	template<typename type, class Allocator>
	ndarray_ref<type,3> VolumeGpu<type, Allocator>::ref() const{
		return ndarray_ref<type,3>(*this);
	}

	template<typename type, class Allocator>
	VolumeGpu<type, Allocator>::VolumeGpu(const ndarray_ref<type,3> &x)
	:Volume(size(0), size(1), size(2)), ext_data_pointer_(true){
		runtime_check(x.device_allowed());
		runtime_check(x.stride_bytes(0)==sizeof(type)); //contiguous in width
		data_ = x.ptr();
		pitch_ = x.stride_bytes(1);
	}

	//4D__________________
	template<typename type>
	ndarray_ref<type,4> TensorCpu<type>::ref() const{
		return ndarray_ref<type,4>(*this);
	}

	/** construct from ndarray_ref shapre: [N x C x H x W]*/
	template<typename type>
	TensorCpu<type>::TensorCpu(const ndarray_ref<type,4> &x):
	LinearHostMemory<type>(x.ptr(), x.numel(), true) {
		runtime_check(x.host_allowed());
		runtime_check(x.stride_bytes(0)==sizeof(type)); //contiguous in width
		runtime_check(x.strides_descending());
		width_ = x.size(3); // for descending strides this is fastest
		height_ = x.size(2);
		channels_ = x.size(1);
		samples_ = x.size(0);
	}
	//_____
	template<typename type>
	ndarray_ref<type,4> TensorGpu<type>::ref() const{
		return ndarray_ref<type,4>(*this);
	}

	/** construct from ndarray_ref shapre: [N x C x H x W]*/
	template<typename type>
	TensorGpu<type>::TensorGpu(const ndarray_ref<type,4> &x):
	LinearHostMemory<type>(x.ptr(), x.numel(), true) {
		runtime_check(x.host_allowed());
		runtime_check(x.strides_descending());
		//use [N x C x H x W] layout
		runtime_check(x.stride_bytes(3)==sizeof(type)); //contiguous in width
		samples_ = x.size(0);
		channels_ = x.size(1);
		height_ = x.size(2);
		width_ = x.size(3);
		memoryLayout_ = iu::TensorGpu<type>::NCHW;
		/*
		}else{                      // [N x H x W x C]
			runtime_check(x.strides_ascending());
			runtime_check(x.stride_bytes(3)==sizeof(type)); //contiguous in width
			samples_ = x.size(0);
			height_ = x.size(1);
			width_ = x.size(2);
			channels_ = x.size(3);
			memoryLayout_ = iu::TensorGpu<type>::NHWC;
		};
		 */
	}
	//_______________math______________________________________
	/*
	//--------single array input--
	//! addC: dest = src1 + val
	template<typename type int dims> addC(const ndarray_ref<type,dims> & src, type val, ndarray_ref<type,dims> & dest){
		add(dest,src,val);
	}
	//! mulC: dest = src1 * val
	template<typename type int dims> mulC(const ndarray_ref<type,dims> & src, type val, ndarray_ref<type,dims> & dest){
		mul(dest,src,val);
	}
	//! fill: dest = val
	template<typename type int dims> fill(ndarray_ref<type,dims> & dest, type val){
		dest << val;
	}

	//--------2 array input-------
	//! dest = src1 + src2
	template<typename type int dims> add(const ndarray_ref<type,dims> & src1, const ndarray_ref<type,dims> & src2, ndarray_ref<type,dims> & dest){
		add(dest,src1,src2);
	}
	//! dest = src1 * src2
	template<typename type int dims> mul(const ndarray_ref<type,dims> & src1, const ndarray_ref<type,dims> & src2, ndarray_ref<type,dims> & dest){
		mul(dest,src,val);
	}
	//! dest = w1*src1 + w2*src2
	template<typename type int dims> addWeighted(const ndarray_ref<type,dims> & src1, type w1, const ndarray_ref<type,dims> & src2, type w2, ndarray_ref<type,dims> & dest){
		mul(src,val,dest);
	}
	//! maxVal = min(src(:))
	template<typename type int dims> minMax(const ndarray_ref<type,dims> & src, float& minVal, float& maxVal){
	}
	//! sum of all elements
	template<typename type int dims> summation(const ndarray_ref<type,dims> & src, float& sum){
	}

	//! sum_i |src1(i)-src2(i)|
	void normDiffL1(const ndarray_ref<type,dims> & src1, const ndarray_ref<type,dims> & src2, float& norm){
	}

	//! sum_i |src1(i)-src2(i)|^2
	void normDiffL2(const ndarray_ref<type,dims> & src1, const ndarray_ref<type,dims> & src2, float& norm){
	}

	//! mean-squared error (MSE) sum_i ( x_i - y_i )^2
	void mse(const ndarray_ref<type,dims> & src1, const ndarray_ref<type,dims> & src2, float& norm){
	}

	//-- complex math------------
	void abs(iu::VolumeCpu_32f_C2& complex, iu::VolumeCpu_32f_C1& real);
	void real(iu::VolumeGpu_32f_C2& complex, iu::VolumeGpu_32f_C1& real);
	void imag(iu::ImageCpu_32f_C2& complex, iu::ImageCpu_32f_C1& real);
	void phase(iu::ImageCpu_32f_C2& complex, iu::ImageCpu_32f_C1& real);
	void scale(iu::VolumeGpu_32f_C2& complex_src, const float& scale, iu::VolumeGpu_32f_C2& complex_dst);
	void multiply(iu::VolumeGpu_32f_C2& complex_src, iu::VolumeGpu_32f_C1& real, iu::VolumeGpu_32f_C2& complex_dst);
	void multiply(iu::VolumeGpu_32f_C2& complex_src1, iu::VolumeGpu_32f_C2& complex_src2, iu::VolumeGpu_32f_C2& complex_dst);
	void multiplyConjugate(iu::ImageCpu_32f_C2& complex_src1, iu::ImageCpu_32f_C2& complex_src2, iu::ImageCpu_32f_C2& complex_dst);
	 */
}

