#pragma once

#include <assert.h>
#include <string>
#include <iostream>
#include <sstream>

#include "defines.h"

//_______________________error_stream____________________________________________________________
//! error_stream -- throw an error message with file tag and stack trace info
/* usage: throw error() << "something wrong with my_variable = " << my_variable << "\n" <<", thats bad.";
          throw error("not good at all");
          runtime_check(a<0); // throws if condition is not satisfied
*/

class error_stream : public std::exception, public host_stream{
private:
	std::string msg;
public:
	std::stringstream ss;
	std::stringstream stack;
	std::string file_line;
public:
	error_stream();
	virtual ~error_stream() throw();
	error_stream(const error_stream & b);
	error_stream & operator = (const error_stream & b);
	error_stream(const std::stringstream & _ss);
	error_stream(const std::string & s);
	error_stream(const char * s);
public:
	error_stream & set_file_line(const char * __file, int __line);
public:
	error_stream & operator << (int a){ ss << a; return *this;}
	error_stream & operator << (long long a){ ss << a; return *this;}
	error_stream & operator << (size_t a){ ss << a; return *this;}
	error_stream & operator << (float a){ ss << a; return *this;}
	error_stream & operator << (double a){ ss << a; return *this;}
	error_stream & operator << (void * a){ ss << a; return *this;}
	error_stream & operator << (const char * a){ ss << a; return *this;}
	error_stream & operator << (const std::string & a){ ss << a; return *this;}
	/*
	template<typename type>
	error_stream & operator << (const type & a){
		ss << a;
		return *this;
	}
	*/
	//error_stream & operator << (const error_stream & a);
public:
	const char * what()const throw() override;
};

#define throw_error(text_stuff)\
    throw error_stream(text_stuff).set_file_line(__FILE__,__LINE__)


//inline void error(const char * text = ""){
//    throw error_stream(text).set_file_line(__FILE__,__LINE__);
//    //return error_stream();
//}

#ifndef  __CUDA_ARCH__
	#define runtime_check(expression) if(!(expression)) error_stream().set_file_line(__FILE__,__LINE__) << "Runtime check failed: " << #expression << "\n"
#else
	#define runtime_check(expression) if(!(expression)) pf_error_stream(0) << __FILE__ << __LINE__ << "Runtime check failed " << #expression << "\n"
#endif
