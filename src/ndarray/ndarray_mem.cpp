#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <device_functions.hpp>

#include "ndarray_mem.h"
#include "error_cuda.h"


bool is_ptr_device_accessible(void * ptr){
	//cuda_check_errors();
	cudaPointerAttributes attr;
	if (cudaPointerGetAttributes(&attr, ptr) == cudaErrorInvalidValue) {
		cudaGetLastError(); // clear out the previous API error
		return false; // this is a pointer not known to cuda, i.e. host's malloc, new, etc.
	};
	return attr.devicePointer == ptr; // CUDA API 7.5: If the memory referred to by ptr cannot be accessed directly by the current device then this is NULL. See also cudaHostAlloc (all accessible by device)
	/* more conservative:
	if(attr.memoryType == cudaMemoryTypeDevice) return true;
	if(attr.isManaged) return true;
	return false;
	*/
};



bool is_ptr_host_accessible(void * ptr){
	//cuda_check_errors();
	cudaPointerAttributes attr;
	if (cudaPointerGetAttributes(&attr, ptr) == cudaErrorInvalidValue) {
		cudaGetLastError(); // clear out the previous API error
		return true; // this is a pointer not known to cuda, i.e. host's malloc, new, etc.
	};
	return attr.hostPointer == ptr; // CUDA API 7.5: If the memory referred to by ptr cannot be accessed directly by the current device then this is NULL. See also cudaHostAlloc (all accessible by device)
	//if(attr.isManaged) return true;
	//if(attr.memoryType == cudaMemoryTypeHost) return true;
	//return false;
};

int ptr_access_flags(void * ptr){
	cudaPointerAttributes attr;
	if (cudaPointerGetAttributes(&attr, ptr) == cudaErrorInvalidValue) {
		cudaGetLastError(); // clear out the previous API error
		return ndarray_flags::host_only; // this is a pointer not known to cuda, i.e. host's malloc, new, etc.
	};
	if(attr.hostPointer == ptr && attr.devicePointer == ptr) return ndarray_flags::host_device;
	if(attr.hostPointer == ptr) return ndarray_flags::host_only;
	if(attr.devicePointer == ptr) return ndarray_flags::device_only;
    throw_error("no access?");
    return ndarray_flags::no_access;
};

//_______________memory________________________________
namespace memory{
std::map<size_t, memory::base_allocator> allocators;
std::map<void *, ptr_info> journal;

void ptr_attr(void * ptr) {
	cudaPointerAttributes pa;
	std::cout << "Pointer Value:   " << (void*)ptr << std::endl;
	if (cudaPointerGetAttributes(&pa, ptr) == cudaErrorInvalidValue) {
		cudaGetLastError();
		std::cout << "pointer was not allocated / registered with cuda" << std::endl;
	} else{
		std::cout << "Pointer attributes:\n";
		std::string mt = pa.memoryType == cudaMemoryTypeHost ? "cudaMemoryTypeHost"
				: "cudaMemoryTypeDevice";
		std::cout << "  memoryType:    " << mt << std::endl;
		std::cout << "  device:        " << pa.device << std::endl;
		std::cout << "  devicePointer: " << pa.devicePointer << std::endl;
		std::cout << "  hostPointer:   " << pa.hostPointer << std::endl;
		std::cout << "  isManaged:     " << pa.isManaged << std::endl;
		if(pa.devicePointer != ptr){
			std::cout << "--checking pa.devicePointer--" << std::endl;
			ptr_attr(pa.devicePointer);
			std::cout << "--" << std::endl;
		};
		if(0 && pa.hostPointer != ptr){
			std::cout << "--checking pa.hostPointer--" << std::endl;
			ptr_attr(pa.hostPointer);
			std::cout << "--" << std::endl;
		};
	};
	//cudaHostGetDevicePointer(void *pDevice, void *pHost, unsigned int flags)
}


void base_allocator::journal_allocation(void * ptr, size_t size_bytes){
	journal[ptr].size_bytes =  size_bytes;
	journal[ptr].allocator =  this;
	//std::cout << "allocated:" << ptr <<" "<< journal_info(ptr) <<"\n";
	//ptr_attr(ptr);
}

void base_allocator::journal_deallocation(void * ptr){
	//journal.erase(ptr);
	//std::cout << "deallocating:" << ptr <<" "<< journal_info(ptr) <<"\n";
}

std::string journal_info(void * ptr){
	std::stringstream ss;
	ss << "mem journal info: ";
	if(journal.find(ptr)!=journal.end()){
		ss << "last allocated by " << typeid(*journal[ptr].allocator).name() << " " << journal[ptr].size_bytes/1024.0/1024.0 << "MB";
	}else{
		ss << "never allocated at this address.";
	};
	return ss.str();
};

};

//
void memory::base_allocator::allocate(void *& ptr, const int size[], int n, int element_size_bytes, int * stride_bytes){
	size_t sz = element_size_bytes;
	for (int i = n - 1; i >= 0; --i){
		sz *= size[i];
	};
	allocate(ptr, sz);
	long long sb = element_size_bytes;
	for (int i = 0; i < n; ++i){
		runtime_check(sb < std::numeric_limits<int>::max());
		stride_bytes[i] = (int)sb;
		sb *= size[i];
	};
};

void memory::CPU::allocate(void *& ptr, size_t size_bytes){
	ptr = malloc(size_bytes);
	journal_allocation(ptr,size_bytes);
};

void memory::CPU::deallocate(void * ptr){
	journal_deallocation(ptr);
	free(ptr);
};

void memory::GPU::allocate(void *& ptr, size_t size_bytes){
	//std::cout <<"cudaMalloc\n";
	cuda_check_error();
	cudaMalloc(&ptr, size_bytes);
	cudaError_t r = cudaGetLastError();
	runtime_check(r == cudaSuccess) << cudaGetErrorString(r) << "\n size requested:" << int(size_bytes/1024/1024) << "Mb\n";
	journal_allocation(ptr,size_bytes);
};

void memory::GPU::deallocate(void * ptr){
	cudaDeviceSynchronize();
	cuda_check_error();
	journal_deallocation(ptr);
	cudaFree(ptr);
	cuda_check_error();
};

void memory::GPU_managed::allocate(void *& ptr, size_t size_bytes){
	//std::cout <<"cudaMallocManaged\n";
	cudaDeviceSynchronize();
	cuda_check_error();
	cudaMallocManaged(&ptr, size_bytes, cudaMemAttachGlobal);
	cudaDeviceSynchronize();
	cuda_check_error();
	journal_allocation(ptr,size_bytes);
};

void memory::GPU_managed::deallocate(void * ptr){
	cudaDeviceSynchronize();
	cuda_check_error();
	journal_deallocation(ptr);
	cudaFree(ptr);
	cuda_check_error();
};

