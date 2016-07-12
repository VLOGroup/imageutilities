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

//! functions that attach ndarray_ref to all iu classes

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

	/*
	template<typename type>
	ndarray_ref<type, 1>::operator iu::LinearHostMemory<type> (){
		runtime_check(this->host_allowed());
		runtime_check(this->is_contiguous());
		//static_assert(false, "not implemented, iu::LinearHostMemory's copy constructor is private");
		return iu::LinearHostMemory<type>(this->ptr(), (unsigned int)this->numel(), true);
	}

	template<typename type>
	ndarray_ref<type, 1>::operator iu::LinearDeviceMemory<type> (){
		runtime_check(this->device_allowed());
		runtime_check(this->is_contiguous());
		return iu::LinearDeviceMemory<type>(this->ptr(), (unsigned int)this->numel(), true);
	}
	*/

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
		intn<2> stride_bytes = intn<2>(x.data(1, 0) - x.data(0, 0), x.data(0, 1) - x.data(0, 0))*sizeof(type);
		set_ref(const_cast<type*>(x.data(0, 0)), size, stride_bytes, ndarray_flags::device_only);
		return *this;
	}
	/*
	template<typename type>
	template<class Allocator>
	ndarray_ref<type, 2>::operator iu::ImageCpu<type, Allocator>(){
		runtime_check(this->host_allowed());
		static_assert(false, "not implemented");
	};

	template<typename type>
	template<class Allocator>
	ndarray_ref<type, 2>::operator iu::ImageGpu<type, Allocator>(){
		runtime_check(this->device_allowed());
		static_assert(false, "not implemented");
	};
	*/

	//_______3D_____________
	template<typename type>
	template<class Allocator>
	ndarray_ref<type, 3> & ndarray_ref<type, 3>::set_ref(const iu::VolumeCpu<type, Allocator> & x){
		intn<3> size(x.width(), x.height(), x.depth());
		intn<3> stride_bytes = intn<3>(x.data(1, 0, 0) - x.data(0, 0, 0), x.data(0, 1, 0) - x.data(0, 0, 0), x.data(0, 0, 1) - x.data(0, 0, 0))*sizeof(type);
		this->set_ref(const_cast<type*>(x.data(0, 0, 0)), size, stride_bytes, ndarray_flags::host_only);
		return *this;
	}

	template<typename type>
	template<class Allocator>
	ndarray_ref<type, 3> & ndarray_ref<type, 3>::set_ref(const iu::VolumeGpu<type, Allocator> & x){
		intn<3> size(x.width(), x.height(), x.depth());
		intn<3> stride_bytes = intn<3>(x.data(1, 0, 0) - x.data(0, 0, 0), x.data(0, 1, 0) - x.data(0, 0, 0), x.data(0, 0, 1) - x.data(0, 0, 0))*sizeof(type);
		this->set_ref(const_cast<type*>(x.data(0, 0, 0)), size, stride_bytes, ndarray_flags::device_only);
		return *this;
	}

	/*
	template<typename type>
	template<class Allocator>
	ndarray_ref<type, 3>::operator iu::VolumeCpu<type, Allocator>(){
		runtime_check(this->host_allowed());
		static_assert(false, "not implemented");
	};

	template<typename type>
	template<class Allocator>
	ndarray_ref<type, 3>::operator iu::VolumeGpu<type, Allocator>(){
		runtime_check(this->device_allowed());
		static_assert(false, "not implemented");
	};
	*/

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
			size = (t.samples(), t.channels(), t.height(), t.width());
		} else if (t.memoryLayout() == iu::TensorGpu<type>::NHWC){
			size = (t.samples(), t.height(), t.width(), t.channels());
		} else{
			slperror("unknown TensorGPU shape");
		};
		this->template set_linear_ref<false>(const_cast<type*>(t.data()), size, ndarray_flags::device_only); //assume TensorGpu has descending strides
		return *this;
	}

	/*
	//reverse conversions
	template<typename type>
	ndarray_ref<type, 4>::operator iu::TensorCpu<type>(){
		runtime_check(this->host_allowed());
		static_assert(false, "not implemented");
	}

	template<typename type>
	ndarray_ref<type, 4>::operator iu::TensorGpu<type>(){
		runtime_check(this->device_allowed());
		static_assert(false, "not implemented");
	}
	*/
}

namespace iu{

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
		runtime_check(x.stride_bytes(0)==sizeof(type)); //contiguous in width
		if(x.strides_descending()){ // [N x C x H x W] layout
			samples_ = x.size(0);
			channels_ = x.size(1);
			height_ = x.size(2);
			width_ = x.size(3);
			memoryLayout_ = iu::TensorGpu<type>::NCHW;
		}else{                      // [N x H x W x C]
			runtime_check(x.strides_ascending());
			samples_ = x.size(0);
			height_ = x.size(1);
			width_ = x.size(2);
			channels_ = x.size(3);
			memoryLayout_ = iu::TensorGpu<type>::NHWC;
		};
	}
}

/*
template<typename type, class Allocator>
ndarray_ref<type, 2> & ref(const iu::ImageCpu<type, Allocator> & x){
	return ndarray_ref<type, 2>(x);
}
*/

//
////______________________
//
//
////! helper function for ImageCpu
//template<typename type, class Allocator>
//ndarray_ref<type, 2> make_ndarray_ref(const iu::ImageCpu<type, Allocator> & x){
//	intn<2> size(x.width(),x.height());
//	intn<2> stride_bytes = intn<2>(x.data(1, 0) - x.data(0, 0), x.data(0, 1) - x.data(0, 0))*sizeof(type);
//	return ndarray_ref<type, 2>(const_cast<type*>(x.data(0, 0)), size, stride_bytes, ndarray_flags::host_only );
//}
//
////! helper function for ImageGpu
//template<typename type, class Allocator>
//ndarray_ref<type, 2> make_ndarray_ref(const iu::ImageGpu<type, Allocator> & x){
//	intn<2> size(x.width(),x.height());
//	intn<2> stride_bytes = intn<2>(x.data(1, 0) - x.data(0, 0), x.data(0, 1) - x.data(0, 0))*sizeof(type);
//	return ndarray_ref<type, 2>(const_cast<type*>(x.data(0, 0)), size, stride_bytes, ndarray_flags::device_only );
//}
//
////! helper function for VolumeCpu
//template<typename type, class Allocator>
//ndarray_ref<type, 2> make_ndarray_ref(const iu::VolumeCpu<type, Allocator> & x){
//	intn<3> size(x.width(),x.height(),x.depth());
//	intn<3> stride_bytes = intn<3>(x.data(1, 0, 0) - x.data(0, 0, 0), x.data(0, 1, 0) - x.data(0, 0, 0), x.data(0, 0, 1) - x.data(0, 0, 0))*sizeof(type);
//	return ndarray_ref<type, 3>(const_cast<type*>(x.data(0, 0, 0)), size, stride_bytes, ndarray_flags::host_only );
//}
//
////! helper function for VolumeGpu
//template<typename type, class Allocator>
//ndarray_ref<type, 2> make_ndarray_ref(const iu::VolumeGpu<type, Allocator> & x){
//	intn<3> size(x.width(),x.height(),x.depth());
//	intn<3> stride_bytes = intn<3>(x.data(1, 0, 0) - x.data(0, 0, 0), x.data(0, 1, 0) - x.data(0, 0, 0), x.data(0, 0, 1) - x.data(0, 0, 0))*sizeof(type);
//	return ndarray_ref<type, 3>(const_cast<type*>(x.data(0, 0, 0)), size, stride_bytes, ndarray_flags::device_only );
//}
//
////! helper function for TensorCpu
//template<typename type>
//ndarray_ref<type, 4> make_ndarray_ref(const iu::TensorCpu<type> & t){
//	// according to TensorCpu:: getPixel(unsigned int n, unsigned int c, unsigned int x, unsigned int y)
//	intn<4> size(t.samples(),t.channels(), t.height(), t.width());
//	ndarray_ref<type, 4> r;
//	r.set_linear_ref<false>(const_cast<type*>(t.data()),size,ndarray_flags::host_only); //assume TensorCpu has descending strides
//	return r;
//}
//
////! helper function for TensorGpu - two layouts
//template<typename type>
//ndarray_ref<type, 4> make_ndarray_ref(const iu::TensorGpu<type> & t){
//	intn<4> size;
//	if(t.MemoryLayout == iu::TensorGpu<type>::NCHW){
//		size = (t.samples(), t.channels(), t.height(), t.width());
//	}else if(t.MemoryLayout == iu::TensorGpu<type>::NHWC){
//		size = (t.samples(), t.height(), t.width(), t.channels());
//	}else{
//		slperror("unknown TensorGPU shape");
//	};
//	ndarray_ref<type, 4> r;
//	r.set_linear_ref<false>(const_cast<type*>(t.data()),size,ndarray_flags::device_only); //assume TensorGpu has descending strides
//	return r;
//}
//
////__________ndarray_ref constructors: ImageCpu, ImageGpu, VolumeCpu, VolumeGpu
//template<typename type, int dims>
//template <class Allocator, template<typename,class> class Image>
//ndarray_ref<type,dims>& ndarray_ref<type,dims>::operator = (const Image<type,Allocator> & x){
//	(*this) = make_ndarray_ref(x);
//	return *this;
//}
//
///*
//template<typename type, int dims>
//template <class Allocator, template<typename,class> class Image>
//ndarray_ref<type,dims>::ndarray_ref(const Image<type,Allocator> & x){
//	(*this) = make_ndarray_ref(x);
//}
//*/
//
//template<typename type, int dims>
//ndarray_ref<type,dims>::ndarray_ref(const iu::LinearDeviceMemory<type> & x, const intn<dims> size){
//	set_linear_ref(const_cast<type*>(x.data()),size,ndarray_flags::device_only);
//}
//
//template<typename type, int dims>
//ndarray_ref<type,dims>::ndarray_ref(const iu::LinearHostMemory<type> & x, const intn<dims> size){
//	set_linear_ref(const_cast<type*>(x.data()),size,ndarray_flags::host_only);
//}
//
//template<typename type, int dims>
//ndarray_ref<type,dims> & ndarray_ref<type,dims>::operator = (const iu::TensorGpu<type> & t){
//	(*this) = make_ndarray_ref(t);
//	return *this;
//};
//
//template<typename type, int dims>
//ndarray_ref<type,dims> & ndarray_ref<type,dims>::operator = (const iu::TensorCpu<type> & t){
//	(*this) = make_ndarray_ref(t);
//	return *this;
//};
