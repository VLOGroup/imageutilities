struct A{
	A(){};
	A(const A &) = delete;
};

struct B : A{
	B(const B & x){};
	B(int x){};
};

struct C{
	operator B() {
		return B(1);
	};
};

void foo(const A & x){
};

float test_warn(int x){
	int n = 100;
	double p = 7.0/3.0f;
	float * bla = new float[n];
	int z = 8;
	bla[2] = x;
	//int * q =bla;
	z = 0.5f/ 0.0f;
	z = bla[z] + p == bla[5];
}

int main(){
	C c;
	foo(c.operator B()); // Ok	
	//foo(c); // Not Ok
	//const B & b = c;
	const A & a = c;
	A a1;
	return 0;
};
