

import os,sys
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt
from image_tools import rgb2gray


# import the c++/cuda module
import libpyTest as pt

if __name__ == '__main__':
    
    print 'hello'

    datapath = os.path.join(os.getenv('IMAGEUTILITIES_ROOT'), 'tests', 'data')
    
    I1 = rgb2gray(scipy.misc.imread(os.path.join(datapath, 'army_1.png'))).astype('uint8')
    pt.test1(I1)
    
    #print I1[0:5,0:10]
    #plt.figure(); plt.imshow(I1, cmap='gray', interpolation='none'); plt.colorbar()
    #plt.show()
    
    
