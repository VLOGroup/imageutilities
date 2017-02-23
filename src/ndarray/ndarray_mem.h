#pragma once

#include "ndarray_ref.host.h"

#include <map>
#include <typeinfo>
#include <typeindex>
#include "ndarray_exports.h"

//_________________________memory__________________________________________________
namespace memory{

	//! check whether pointer device-accessible
	bool is_ptr_device_accessible(void * ptr);
	bool is_ptr_host_accessible(void * ptr);

	class base_allocator{ // class declaring allocator interface
	public:
		//! should return the associated access policy
		virtual int access_policy(){
			return ndarray_flags::no_access;
		}
		//! allocate a chunk
		virtual void allocate(void *& ptr, size_t size_bytes){
			throw_error("empty / mot implemented");
		}
		//! deallocate
		virtual void deallocate(void * ptr){
			throw_error("empty / mot implemented");
		}

		//! allocate array according to size, output the pointer and stride_bytes
		template<typename type, int dims>
		void allocate(void *& ptr, const intn<dims> size, intn<dims> & stride_bytes){
			allocate(ptr, size.begin() , dims, sizeof(type), stride_bytes.begin());
		}
		//! allocate array according to size, output the pointer and stride_bytes. Ddefault is to grab a linear chunk without aligning -- override as needed
		virtual void allocate(void *& ptr, const int size[], int n, int element_size_bytes, int * stride_bytes);
		bool is_base(){// true for this class and false for any derived class
			//error("not implemented");
			return std::type_index(typeid(this)) == typeid(base_allocator);
		}
		virtual ~base_allocator(){}
		void journal_allocation(void * ptr, size_t size_bytes);
		void journal_deallocation(void * ptr);
	};
	//! allocator using malloc()
	class CPU : public base_allocator{
	public:
		using base_allocator::allocate;
		void allocate(void *& ptr, size_t size_bytes) override;
		void deallocate(void * ptr) override;
		virtual int access_policy() override{
			return ndarray_flags::host_only;
		};
	};
	//! allocator using cudaMalloc()
	class GPU : public base_allocator{
	public:
		using base_allocator::allocate;
		void allocate(void *& ptr, size_t size_bytes) override;
		void deallocate(void * ptr) override;
		virtual int access_policy() override{
			return ndarray_flags::device_only;
		};
	};
	//! allocator using cudaMallocManaged()
	class GPU_managed : public base_allocator{
	public:
		using base_allocator::allocate;
		void allocate(void *& ptr, size_t size_bytes) override;
		void deallocate(void * ptr) override;
		virtual int access_policy() override{
			return ndarray_flags::host_device;
		};
	};
	//! allocator using cudaTexture()
	class GPU_texture : public base_allocator{
	public:
		using base_allocator::allocate;
		// not implemented
	};
	//... etc.
	//________________
	//! a pool of allocators, one of each type, constructed on first demand, host side
	extern NDARRAY_EXPORTS std::map<size_t, base_allocator> allocators;
	struct ptr_info{
		size_t size_bytes;
		base_allocator * allocator;
	};
	extern NDARRAY_EXPORTS std::map<void *, ptr_info> journal;
	std::string journal_info(void * ptr);
	template<class Allocator> static Allocator * get(){
		size_t a = typeid(Allocator).hash_code();
		auto it = allocators.find(a);
		if(it == allocators.end()){
			//std::cout << "Registering Allocator: " << a.name() << "\n";
			//it = allocators.insert(std::make_pair(a, Allocator()) ).first;
			//Allocator * al = dynamic_cast<Allocator *>( &(allocators[a]) );
			//&(allocators[a])
			base_allocator & al = allocators[a];
			static_assert(sizeof(Allocator) == sizeof(base_allocator),"some problem");
			new (&al) Allocator; // initialize polymorphic inplace
		};
		//Allocator * al = dynamic_cast<Allocator *>( &(allocators[a]) );
		//Allocator * al = dynamic_cast<Allocator *> it.second;
		Allocator * al  = dynamic_cast<Allocator *>( &(allocators[a]) );
		runtime_check(al!=0);
		return al;
	}
	/*
	static base_allocator * get(const std::type_index & tid){
		return &allocators[tid];
	}
	 */
	//! convinience function to get type_index
	/*
	template<class Allocator> static
		std::type_index tid(){
		return std::type_index(typeid(Allocator));
	}
	 */
	void ptr_attr(void * ptr);
}

//___________________________________ndarray__________________________________________________
template<typename type, int dims> class ndarray : public ndarray_ref < type, dims > {
public:
	typedef ndarray_ref < type, dims > parent;
private:
	//! remember which allocaotr was used to construct it
	memory::base_allocator * al;
public: // inherited stuff
	using parent::size;
	using parent::aligned;
	using parent::begin;
	using parent::end;
	using parent::ptr;
	using parent::flags;
protected:// inherited stuff
	using parent::_beg;
	using parent::sz;
	using parent::_stride_bytes;
	//using parent::set_flags;
public: //_________________ memory stuff
	void set_allocator(memory::base_allocator * _al){
		runtime_check(_al!=0);
		this->al = _al;
	}
	//! get the allocator
	memory::base_allocator & allocator()const{
		runtime_check(al!=0);
		return *al;
	}
	bool is_reference()const{
		return (al != 0);
	}
	bool allocated()const{
		return ptr() != 0 && al != 0 && !allocator().is_base();
	}
	void clear(){
		if (allocated()){
			allocator().deallocate(ptr());
		}
	}
	//
	~ndarray(){
		//std::cout <<"deallocating array" << *this;
		try{
			clear();
		}catch(std::exception & err){
			std::cerr << err.what() << "\n";
			std::cerr << *this;
			memory::ptr_attr(ptr());
			std::cerr << memory::journal_info(ptr()) << "\n";
			exit(1);
		};
	}
public://__________constructors / initializers
	//! uninitialized
	ndarray() :al(0){}
	/*
	//! given an Allocator and size
	template<class Allocator> ndarray(const intn<dims> & size){
		create<Allocator>(size);
	}
	//! given an Allocator and size
	template<class Allocator> ndarray(int sz0, int sz1=1, int sz2=1, int sz3=1){
		create<Allocator>(intn<dims>(sz0,sz1,sz2,sz3));
	}
	 */
	//! construct array of a given size using Allocator
	template<class Allocator> void create(const intn<dims> & size, const intn<dims> & order = intn<dims>::enumerate()){
		runtime_check(!allocated());
		intn<dims> szp;
		intn<dims> stp;
		// permute size to requested contiguity order
		for(int d=0; d<dims; ++d){
			szp[d] = size[order[d]];
		};
		set_allocator(memory::get<Allocator>());
		allocator().allocate((void*&)_beg, szp.begin(), dims, sizeof(type), stp.begin());
		// permute stride back to the original order
		for(int d=0; d<dims; ++d){
			_stride_bytes[order[d]] = stp[d];
		};
		sz = size;
		this->set_access(allocator().access_policy());
		this->find_linear_dim();
	}
	//! construct array of a given size using Allocator, copy shape from existing array
	template<class Allocator, typename type2> void create(const ndarray_ref<type2,dims> & x){
		runtime_check(!allocated());
		set_allocator(memory::get<Allocator>());
		allocator().allocate((void*&)_beg, x.size_bytes());
		sz = x.size();
		_stride_bytes = x.stride_bytes();
		this->set_access(allocator().access_policy());
		this->find_linear_dim();
	}
	//! construct array of a given size using Allocator, e.g. create<memory::GPU>(100,100)
	template<class Allocator, typename... Args>
	void create(int d0, Args... args){
		create<Allocator>(intn<dims>(d0,args...));
	}
	/*
	//! construct array of a given size using Allocator, e.g. create<memory::GPU>(100,100)
	template<class Allocator> void create(int sz0, int sz1 = 1, int sz2 = 1, int sz3 = 1){
		create<Allocator>(intn<dims>(sz0, sz1, sz2, sz3));
	}
	*/
	//! copy constructor - deep copy using the same allocator
	explicit ndarray(const ndarray<type, dims> & x){
		set_allocator(&x.allocator());
		allocator().allocate((void*&)&_beg, x.size.begin(), dims, sizeof(type), _stride_bytes.begin());
		sz = x.size;
		this->set_access(allocator().access_policy());
		this->find_linear_dim();
		// copy data
		copy_data(*this, x);
	}
public:
	//! set to a reference and free own data, if any
	void operator = (const ndarray_ref<type, dims> & x){
		if (ptr() == x.ptr()) return; // assignemnt to itself does nothing
		clear();
		al = 0; // indecate, arrays is a reference
		parent::operator = (x); // copy reference
	}
public:
	const ndarray_ref<type,dims> & ref()const{return *this;};
	ndarray_ref<type,dims> & ref(){return *this;};
private: //________ forbidden, it could be ambiguous what these operators should do. Use create + copy_data
	//! deep copy
	void operator = (const ndarray<type, dims> & x){
		// possible implementation, reallocating if necessary:
		/*
		if(size not matches){
			destroy();
		}
		if(has no allocator){
			use allocator of x;
		}
		if(not allocated){
			create with the size of x;
		}
		copy_data();
		 */
	}
};

