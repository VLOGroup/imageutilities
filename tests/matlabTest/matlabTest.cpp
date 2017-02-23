#include <mex.h>

// system includes
#include <iostream>

#include "../../src/iumath/typetraits.h"
#include "../../src/iumatlab.h"
#include "../../src/iudefs.h"

#include "../config.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// This function does not check a single thing, it just assumes two images
	// as parameters, does something with them on the GPU and returns the result
	char err_msg[128];

	// Checking number of arguments
	if (nrhs != 1)
		mexErrMsgIdAndTxt(
		"MATLAB:tgvreconstruction:invalidNumInputs",
		"One input required (2D array)");
	if (nlhs > 1)
		mexErrMsgIdAndTxt("MATLAB:tvreconstruction:maxlhs",
		"Too many output arguments.");

	iu::LinearHostMemory<float, 2> test(*(prhs[0]));

	// Convert to MATLAB Output
	iu::matlab::convertCToMatlab(test, &plhs[0]);

	std::cout << "finished" << std::endl;
}
