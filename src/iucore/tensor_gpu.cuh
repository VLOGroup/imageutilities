namespace cuda
{

__host__  __device__ tensorDim4 getCoords(unsigned int linearIdx, unsigned int *n, unsigned int *c, unsigned int *h,
		unsigned int *w, unsigned int dimS, unsigned int dimC, unsigned int dimH, unsigned int dimW)
{
	*n = linearIdx / (dimC * dimH * dimW);
	*c = (linearIdx % (dimC * dimH * dimW)) / (dimH * dimW);
	*h = ((linearIdx % (dimC * dimH * dimW)) % (dimH * dimW)) / dim.w;
	*w = ((linearIdx % (dimC * dimH * dimW)) % (dimH * dimW)) % dim.w;

	return coords;
}

}
