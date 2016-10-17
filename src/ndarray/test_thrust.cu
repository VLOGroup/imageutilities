#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

int main1(){
	thrust::device_vector<int> v(4);
	thrust::fill(thrust::device, v.begin(), v.end(), 137);
	return 0;
};
