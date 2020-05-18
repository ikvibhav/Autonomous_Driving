import numpy as np


def conv_naive(image, kernel):
    
    """ 
    Naive implementation of convolution.
    ------------------------------------
    This function should compute convolution of a given input  with a kernel and the 
    output ahould the same shape as the input. This is a naive implmentation, we trade 
    performance to get familiar with implementing convolution. This function should loop 
    over dimensions of the image and kernel(H_i,W_i ; H_j,W_j) multiplying respective 
    elements from both arrays.

    Arguments :
        ip: (input) numpy array of shape (H_i, W_i).
        kernel: numpy array of shape (H_j, W_j).
    
    Returns:
        output: numpy array of shape (H_i, W_i).

    """
    H_i, W_i = image.shape
    H_k, W_k = kernel.shape
    output = np.zeros((H_i, W_i))

    # Your implementation goes here
    pass

    return output

def zero_pad(image, pad_vert, pad_horz):
    
    """ 
    Pad the input with zeros.
    -----------------------
    If you have a 1x1 input array = [1] (just a scalar element = 1) with pad_vert = 2, 
    pad_horz = 1, this function should create the following array :

        0 0 0
        0 0 0
        0 1 0      
        0 0 0
        0 0 0

    Arguments :
        ip: numpy array of shape (H, W).
        pad_horz: width of the zero padding (left and right of the input).
        pad_vert: height of the zero padding (top and bottom of input).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    
    """

    H_zp, W_zp = image.shape

    # Your implementation goes here
    pass

    return out


def conv_fast(image, kernel):
    """
    Efficient implementation of convolution.
    ------------------------------------
    This function should make use of element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each pixel.
    
    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).
    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    H_i, W_i = image.shape
    H_k, W_k = kernel.shape
    out = np.zeros((Hi, Wi))

    # Your implementation goes here
    pass
    

    return out
