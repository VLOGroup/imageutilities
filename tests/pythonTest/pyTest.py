
import os,sys
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt

# import the c++/cuda module
import libpyTest as pt

if __name__ == '__main__':
    
    datapath = os.path.join(os.getenv('IMAGEUTILITIES_ROOT'), 'tests', 'data')
    
    I1 = scipy.misc.imread(os.path.join(datapath, 'army_1.png'))
    I1 = np.dot(I1[...,:3], [0.299, 0.587, 0.114]).astype('float32')
    I2 = I1.copy()
    
    print 'max before:', np.max(I1)
    pt.test1(I1)   # c++ function that modifies I1 inplace
    print 'max after:', np.max(I1)
    
    if np.max(I1) > 1:
        print 'Error', np.max(I1)
        sys.exit(1)
    
    I2_upside = pt.test2(I2)  # c++ function that returns a flipped version of I2
    if np.sum(np.abs(np.flipud(I2)-I2_upside)) > 1e-6:
        print 'Error', np.sum(np.flipud(I2)-I2_upside)
        sys.exit(1)
    
    #plt.figure(); plt.imshow(I2, cmap='gray', interpolation='none'); plt.colorbar()
    #plt.figure(); plt.imshow(I2_upside, cmap='gray', interpolation='none'); plt.colorbar()
    #plt.show()
    
    m1 = np.zeros((4,4)); m1[0] = np.arange(1,5); m1[1] = np.arange(1,5)*2; m1[2] = np.arange(1,5)*3; m1[3] = np.arange(1,5)*4
    m2 = np.zeros((4,4)); m2[:,0] = np.arange(4,0,-1)*4; m2[:,1] = np.arange(4,0,-1)*3; m2[:,2] = np.arange(4,0,-1)*2; m2[:,3] = np.arange(4,0,-1)
    pt.test3(m1,m2)    # c++ function that computes the product of m1*m2
    print 'python product', np.dot(m1,m2)
    
    param = 0.5
    testC = pt.MyClass()    # instantiate exposed c++ class
    testC.set_image(I2)     # call methods
    testC.compute(param)    # this calls a class method that executes a cuda kernel
    I3 = testC.get_result()
    
    if np.sum(np.abs(np.fliplr(np.flipud(I2))*param - I3)) > 1e-6:
        print np.sum(np.abs(np.fliplr(np.flipud(I2))*param-I3))
        sys.exit(1)
        
    #plt.figure(); plt.imshow(I3, cmap='gray', interpolation='none'); plt.colorbar()
    #plt.show()
    
    print 'Success'
    sys.exit(0)
    
    
    
