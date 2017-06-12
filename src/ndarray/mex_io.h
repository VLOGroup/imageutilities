#pragma once

#include <mex.h>
#include <typeinfo>

#include "ndarray.h"
#include "options.h"

//_________________redirect stdout to matlab_______________________
class mystream : public std::streambuf
{
protected:
	virtual std::streamsize xsputn(const char *s, std::streamsize n) { mexPrintf("%.*s", n, s); return n; }
	virtual int overflow(int c = EOF) { if (c != EOF) { mexPrintf("%.1s", &c); } return 1; }
};
class scoped_redirect_cout
{
public:
	scoped_redirect_cout() { old_buf = std::cout.rdbuf(); std::cout.rdbuf(&mout); }
	~scoped_redirect_cout() { std::cout.rdbuf(old_buf); }
private:
	mystream mout;
	std::streambuf *old_buf;
};
static scoped_redirect_cout mycout_redirect;
//__________________________________________________________________

class mx_exception : public std::exception{
public:
	std::string s;
public:
	mx_exception(const char * pc){
		mexErrMsgTxt(pc);
		s = std::string("mex_io error") + pc;
	}
	mx_exception(const std::string & s){
		mexErrMsgTxt(s.c_str());
		this->s = std::string("mex_io error") + s;
	}
	const char* what() const throw() override{
		return s.c_str();
	}
};


//! match c++ types to matlab type identifiers
template<typename type> mxClassID mexClassId();
template<typename type> mxClassID mexClassId(){
	throw mx_exception(std::string("Type is not recognized: ")+typeid(type).name());
};

template<> mxClassID mexClassId<char>(){
	return mxINT8_CLASS;
};

template<> mxClassID mexClassId<unsigned char>(){
	return mxUINT8_CLASS;
};

template<> mxClassID mexClassId<short>(){
	return mxINT16_CLASS;
};

template<> mxClassID mexClassId<unsigned short>(){
	return mxUINT16_CLASS;
};

template<> mxClassID mexClassId<int>(){
	return mxINT32_CLASS;
};

template<> mxClassID mexClassId<unsigned int>(){
	return mxUINT32_CLASS;
};

template<> mxClassID mexClassId<long long>(){
	return mxINT64_CLASS;
};

template<> mxClassID mexClassId<unsigned long long>(){
	return mxUINT64_CLASS;
};

template<> mxClassID mexClassId<float>(){
	return mxSINGLE_CLASS;
};

template<> mxClassID mexClassId<double>(){
	return mxDOUBLE_CLASS;
};

template<> mxClassID mexClassId<bool>(){
	return mxLOGICAL_CLASS;
};

template<> mxClassID mexClassId<char*>(){
	return mxCHAR_CLASS;
};

class mx_struct;
template<> mxClassID mexClassId<mx_struct>(){
	return mxSTRUCT_CLASS;
};


//! check mxArray has the desired type
template<typename type> bool is_type(const mxArray *A){
	return (mexClassId<type>()==mxGetClassID(A));
};

//! check mxArray has the desired type
template<typename type> void check_type(const mxArray *A){
	if(A==0)throw mx_exception("Bad mxArray");
	if(!is_type<type>(A))throw mx_exception(std::string("Type ")+typeid(type).name()+" expected instead of type "+mxGetClassName(A)+" provided.");
};

//! class to read from mxArray into string
class mx_string : public std::string{
public:
	explicit mx_string(const mxArray * A){
		check_type<char*>(A);
		size_t data_length = mxGetNumberOfElements(A);
		static_cast<std::string&>(*this) = mxArrayToString(A);
	};
	mx_string(){};
	mxArray * get_mxArray()const{
		//allocate full copy for matlab-managed output
		//matlab mem manager takes care this is not lost
		mxArray * p = mxCreateString(this->c_str());
		return p;
	};
};

//! attach to matlab's mxArray (inputs of mexFunction)
template<typename type, int dims> ndarray_ref<type,dims>::ndarray_ref(const mxArray * A){
	check_type<type>(A);
	type * data = (type*)mxGetData(A);
	size_t data_length = mxGetNumberOfElements(A);
	//read off size
	//note: in matlab number of dimensions is alvays >=2 and can be less then expected if trailing dimensions are 1
	const int ndims = mxGetNumberOfDimensions(A);
	const int * sz = mxGetDimensions(A);
	// if not enough dimensions to represent A throw exception
	if (ndims == 2 && dims == 1) {
		if(sz[0] > 1 && sz[1] > 1) throw mx_exception("Not enough dimensions to represent mxArray");
	} else {
		if (ndims != dims)  throw mx_exception(std::string("Dimensions ") + std::to_string(dims) + " expected instead of dimensions " + std::to_string(ndims) + " provided.");
	}
	int i0 = 0;
	//handle row vectors:
	intn<dims> Size;
	if(ndims==2 && dims==1 && sz[i0]==1)++i0;
	//read remaining dimensions
	for(int i=0;i<std::min(dims,ndims-i0);++i){
		Size[i] = sz[i+i0];
	};
	//set missing dimensions to 1
	for(int i=ndims;i<dims;++i){
		Size[i] = 1;
	};
	// take the pointer
	assert(Size.prod() == data_length);
	this->set_linear_ref(data,Size,ndarray_flags::host_only);
	//_A = (mxArray*)A;
}

//! make a deep copy from ndarray_ref to matlab (outputs of mexFunction)
template<typename type, int dims> void operator << (mxArray *& A, const ndarray_ref<type,dims> & x){
	A = mxCreateNumericArray(dims, x.size().begin(), mexClassId<type>(), mxREAL);
	auto Ar = ndarray_ref<type,dims>(A);
	Ar << x; // copy data
}


//__________________________________________________________________________________________

//! read an options structure from matlab
class mx_struct : public options {
public:
	typedef options parent;
public:
	explicit mx_struct(const mxArray * A){
		check_type<mx_struct>(A);
		//size_t data_length = mxGetNumberOfElements(A);
		int nfields = mxGetNumberOfFields(A);
		int NStructElems = mxGetNumberOfElements(A);
		if (NStructElems > 1)throw mx_exception("this struct cannot take arrays");
		int jstruct = 0;
		for (int i = 0; i < nfields; ++i){
			std::string name = mxGetFieldNameByNumber(A, i);
			mxArray * tmp = mxGetFieldByNumber(A, jstruct, i);
			ndarray_ref<double, 1> a(tmp);
			if (a.numel()>0){
				parent::operator[](name) = a[0];
			};
		};
	}
	mx_struct(){}
	mxArray * get_mxArray(){
		//allocate full copy for matlab-managed output
		//matlab mem manager takes care this is not lost
		int nfields = parent::size();
		typedef const char * tcc;
		tcc* fnames = new tcc[nfields];
		int ifield = 0;
		for (parent::iterator it = parent::begin(); it != end(); ++it, ++ifield){
			fnames[ifield] = it->first.data(); // it->first.c_str();
		};
		mxArray * A = mxCreateStructMatrix(1, 1, nfields, fnames);
		ifield = 0;
		for (parent::iterator it = parent::begin(); it != end(); ++it, ++ifield){
			int sz = 1;
			mxArray * p = mxCreateNumericArray(1, &sz, mexClassId<double>(), mxREAL);
			ndarray_ref<double, 1> a(p);
			a[0] = it->second;
			mxSetFieldByNumber(A, 0, ifield, p);
		};
		delete fnames;
		return A;
	}
};

/*
//! matlab allocator
namespace memory{
	//! allocator using malloc()
	template<typename type> class matlab : public base_allocator{
	public:
		using base_allocator::allocate;
		//! allocate array according to size, output the pointer and stride_bytes. Ddefault is to grab a linear chunk without aligning -- override as needed
		virtual void allocate(void *& ptr, const int size[], int dims, int element_size_bytes, int * stride_bytes) override{
			//allocate full copy for matlab-managed output
			//matlab mem manager takes care this is not lost
			mxArray * p = mxCreateNumericArray(dims, size, mexClassId<type>(), mxREAL);
			ptr = (type*)mxGetData(p);
		}

		virtual void allocate(void *& ptr, size_t size_bytes) override{
			mxArray * p = mxCreateNumericArray(1, &size_bytes, mexClassId<char>(), mxREAL);
			ptr = (type*)mxGetData(p);
		}

		void deallocate(void * ptr) override{
			//do nothing - matlab has garbage collector
		}

		virtual int access_policy() override{
			return ndarray_flags::host_only;
		}
	};
};
*/
