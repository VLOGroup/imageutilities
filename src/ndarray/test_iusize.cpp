#include "ndarray/ndarray.h"

void test_IuSize(){
	//-----1D------
	intn<1> a(5);
	a = 3;
	a < 10;
	runtime_check(a.width == 3);
	runtime_check(a.width == a[0]);
	runtime_check(a.height == 1);
	runtime_check(a.depth == 1);
	//-----2D------
	intn<2> b(5,6);
	b >= 0;
	b < intn<2>(10,10);
	runtime_check(b.width ==5);
	runtime_check(b.height ==6);
	runtime_check(b.width ==b[0]);
	runtime_check(b.height ==b[1]);
	runtime_check(b.depth == 1);
	//-----3D------
	intn<3> c(5,6,7);
	c >= 0;
	c < intn<3>(10,10,10);
	c == intn<3>(10,10,10);
	runtime_check(c.width ==5);
	runtime_check(c.height ==6);
	runtime_check(c.depth ==7);
	runtime_check(c.width ==c[0]);
	runtime_check(c.height ==c[1]);
	runtime_check(c.depth ==c[2]);
	c = intn<3>(2,2,2);
	intn<3> d(c);
	d *= 1.5;
	std::cout << "d=" << d << "\n";
	//-----4D------
	intn<4> s(2,3,4,5);
	//s.height; // error
	std::cout <<"s=" << s << "\n";
	//  s=(2,3,4,5,)
	std::cout << "s.erase<1>() = " << s.erase<1>() << "\n";
	//	s.erase<1>() = (2,4,5,)
	std::cout << "s.erase<1>().height = " << s.erase<1>().height << "\n";
	//	s.erase<1>().height = 4
	intn<4> S(2*5,2*3*7,5*2*3,7);
	for(int i=0; i< S[0];++i){
		for(int j=0; j< S[1];++j){
			for(int k=0; k< S[2];++k){
				for(int l=0; l< S[3];++l){
					intn<4> ii(i,j,k,l);
					runtime_check( S.integer_to_index(S.index_to_integer(ii)) == ii );
				};
			};
		};
	};
	std::cout<<"integer_to_index checked for the range" << S <<"\n";
}