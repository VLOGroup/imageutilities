#pragma once

#include<map>

//template<class type> struct toption;

class options : private std::map < std::string, double > {
public:
	typedef std::map<std::string, double> parent;
public:
	using parent::operator [];
	using parent::begin;
	using parent::end;
	using parent::iterator;
	using parent::size;
	double & operator()(const std::string & key, double _default){
		parent::iterator i = find(key);
		if (i == end()){// not found, construct default
			double & v = parent::operator[](key);
			v = _default;
			return v;
		};
		return i->second;
	};

	// copy options, but keep what we had
	void operator << (const options & O1){
		for (parent::const_iterator it = O1.begin(); it != O1.end(); ++it){
			(*this)[it->first] = it->second;
		};
	};
};

inline std::ostream & operator << (std::ostream & ss, options & O){
	ss << "Options:\n";
	for (options::iterator it = O.begin(); it != O.end(); ++it){
		ss << it->first.c_str() << ": " << it->second << "\n";
	};
	return ss;
}

template<class type> struct toption{
public:
	typedef type parent;
	options * ops;
	std::string name;
	double * val; // pointer to where it is located in map, should be safe under insertionn, etc
public:
	//! quick read access
	operator type()const;
	
	//operator type() const;
	toption(std::string name, type dvalue, options * ops){
		static_assert(sizeof(type) <= sizeof(double), "bad");
		this->name = name;
		this->ops = ops;
		double & V = (*ops)[name];
		V = double(dvalue);
		this->val = &V;
	};
	//! quick write access
	void operator = (const type & new_val){ // if overwrite, then store in the map
		(*val) = (double)new_val;
		//(*ops)[name] = new_val;
	};

	//! copy and operator =
	toption(const toption<type> & x){ // do not copy the parent and member pointer
		*this = x;
	};
	// when setting from an option, only copy the value
	toption<type> & operator = (const toption<type> & x){
		val = &(*ops)[name]; // relink if the pointer has moved due to copy
		*val = *x.val;
		return * this;
	};
};

template <typename ValueType, typename Enable = void>
class Extractor {
public:
	static ValueType extract(double value) {
		return ValueType(value);
	};
};

template <typename ValueType>
class Extractor<ValueType, typename std::enable_if<std::is_enum<ValueType>::value>::type> {
public:
	static ValueType extract(double value) {
		return ValueType(int(value));
	};
};

template<class type> toption<type>::operator type()const {
	return Extractor<type>::extract(*val);
};

template<> inline toption<bool>::operator bool()const{
	return float(*val);
};

#define NEWOPTION(type,name,dvalue) toption<type> name = toption<type>(#name,dvalue, this);
