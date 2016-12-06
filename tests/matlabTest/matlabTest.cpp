#include <mex.h>

// system includes
#include <iostream>

#include "../config.h"

#include "../src/iucore.h"
#include "../src/iumath/typetraits.h"
#include "../src/iumatlab.h"
#include "../src/iumath.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  typedef double pixel_type;
  typedef iu::LinearHostMemory<pixel_type, 2> HostMemType;

  // This function does not check a single thing, it just assumes two images
  // as parameters, does something with them on the GPU and returns the result
  char err_msg[128];

  // Checking number of arguments
  if (nrhs != 1)
    mexErrMsgIdAndTxt(
        "MATLAB:tgvreconstruction:invalidNumInputs",
        "One input required (input)");
  if (nlhs > 1)
    mexErrMsgIdAndTxt("MATLAB:tvreconstruction:maxlhs",
                      "Too many output arguments.");

  HostMemType test(*(prhs[0]));

  // Convert to MATLAB Output
  iu::matlab::convertCToMatlab(test, &plhs[0]);

  std::cout << "finished" << std::endl;
}
