#include "error.h"
#include "stacktrace.h"


error_stream::error_stream(){
	std::cout<<"ERROR\n"<<std::flush;
	print_stacktrace(stack);
}

error_stream::~error_stream() throw (){
	//std::cout << "~ERROR\n"<<std::flush;
	//std::cout << what();
	//__asm__ volatile("int3");
	try{
		if(msg.empty()){
			std::stringstream msgs;
			msgs << "\n___________________\n";
			msgs << ss.str().c_str();
			//std::cout << ss.str().c_str();
			msgs << file_line << "\n";
			msgs << "___________________\n";
			//std::cout<<"X1\n"<<std::flush;
			//std::cout << stack.str().c_str();
			msgs	<< stack.str().c_str();
			//std::cout<<"X2\n"<<std::flush;
			msgs << "___________________\n";
			std::cout << msgs.str().c_str() << std::flush;
			msg = msgs.str();
		};
	//} catch(std::exception & e){
	//	std::cerr << "Ops, " << e.what() << std::flush;
	}catch(...){
		std::cerr << "OOps " << std::flush;
	};
	__HALT__;
}


error_stream & error_stream::set_file_line(const char * __file, int __line){
	file_line = std::string("In ") + __file + " : " + std::to_string(__line);
	//std::cout << file_line << "\n";
	//std::cout << stack.str().c_str() <<" \n" << file_line << "\n" << std::flush;
	return *this;
}

error_stream::error_stream(const error_stream & b){
	//std::cout << "error_stream(const error_stream & b)\n";
	ss << b.ss.str();
	stack << b.stack.str();
	file_line = b.file_line;
	msg = b.msg;
}

error_stream & error_stream::operator = (const error_stream & b){
	//std::cout << "error_stream operator =(const error_stream & b)\n";
	ss << b.ss.str();
	return *this;
}

error_stream::error_stream(const std::stringstream & _ss):error_stream(){
	ss << _ss.str();
}

error_stream::error_stream(const std::string & s):error_stream(){
	ss << s;
}

error_stream::error_stream(const char * s):error_stream(){
	ss << s;
}
/*
error_stream & error_stream::operator << (const error_stream & a){
	return *this;
};
*/

const char * error_stream::what() const throw(){
	return msg.c_str();
}
