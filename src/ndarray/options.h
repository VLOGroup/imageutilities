#ifndef options_h
#define options_h

#include<map>

class options : public std::map < std::string, double > {
public:
	typedef std::map<std::string, double> parent;
public:
	//double & operator()(const std::string & key, double _default);
	//void print();
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
		//parent::operator = (O1);
	};

};

std::ostream & operator << (std::ostream & ss, options & O){
	ss << "Options:\n";
	for (O::iterator it = begin(); it != end(); ++it){
		ss << it->first << ": " << it->second << "\n";
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
	/*
	operator type()const{
		//return type((*ops)[name]);
		return (type)(*val);
	};
	*/
	
	//operator type() const;
	toption(std::string name, type dvalue, options * ops){
		this->name = name;
		this->ops = ops;
		double & V = (*ops)[name];
		V = dvalue;
		this->val = &V;
	};
	//! quick write access
	void operator = (const type & new_val){ // if overwrite, then store in the map
		(*val) = new_val;
		//(*ops)[name] = new_val;
	};
	// when setting from an option, only copy the value
	void operator = (const toption<type> & o2){
		(*this) = type(o2);
	};
};

template<class type> toption<type>::operator type()const{
	return (type)(*val);
};

//template<> toption<bool>::operator bool()const;

template<> inline toption<bool>::operator bool()const{
	return *val!=0;
};

#define NEWOPTION(type,name,dvalue) toption<type> name = toption<type>(#name,dvalue, this);

#endif
