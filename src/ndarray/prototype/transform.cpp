#include "transform.h"
#include "ndarray_mem.h"

void test_transform(){
	ndarray<int,2> A1;
	ndarray<int,2> A2;
	ndarray<int,2> A3;
	A1.create<memory::GPU>({100,200});
	A2.create<memory::GPU>({100,200});
	A3.create<memory::GPU>({100,200});
	auto tup = std::make_tuple(A1.transpose(),A2.ref(),A3.ref());
	auto transform = make_transform_tup(tup);
	bool same = transform.same_size();
	//transform.apply(add_op<float,2>());
	//transform.apply(binary_op<int>::add());
	//transform.apply(binary_op<int>::madd2(2.0f, 3.0f));

	//nd_transform(binary_op<int>::add(), A1.transpose(), A2.ref(), A3.ref());
}
