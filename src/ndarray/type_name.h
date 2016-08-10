#pragma once

#include <string>
#include <typeinfo>

std::string demangle(const char* name);

template <class T>
std::string type_name(const T& t) {
    return demangle(typeid(t).name());
}

template <class T>
std::string type_name() {
    return demangle(typeid(T).name());
}
