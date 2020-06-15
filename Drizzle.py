#!/usr/bin/env python
# coding: utf-8
""""Author: Arend-Jan Quist
Date of creation: 17 February 2020
Last modified: 15 June 2020

Description: This program is an implementation of the drizzle algorithm as explained by 
Fruchter and Hook in 2002. This program has been built as part of the Bachelor Project
of Arend-Jan Quist at Leiden University in, Spring 2020. 
This program contains multiple drizzle implementations. 

Usage: Use the drizzle() function to drizzle multiple images at once. Use drizzle_image() 
to drizzle one image. This program works fast only for scale factor of 1/n with n integer. 
For other scale factors a slow implementation is used. This program works for all pixfracs, 
also for p>1. 

Examples: Some examples to use implementation are stated below functions in comments. 

Time complexity: Duration to drizzle one image of size 300 x 300 pixels with n = 2, p = 0.6 
is approximately 0.4 sec on a 2020 pc. For larger imagesize this time increases fast. 
See Bachelor's Thesis of Arend-Jan Quist for more detailed time duration results.

To be improved: 
-> For p>1 a boundary effect occurs in the matrix_shifting version of drizzle.
-> The drizzle implementation with for loops could be improved by numba.
"""


import numpy as np
from scipy import ndimage
from skimage import io



def enlarge_matrix_entries(k, n = 3):
    """Enlarge both dimensions of matrix k by factor n, using convolution
    All entries are replaced by blockmatrices of size (n x n) with entry in left up
    
    Written by Tobias de Jong
    """
    K = np.concatenate([k] + (n-1)*[np.full_like(k,0.0)], axis=1)
    K = K.reshape((-1,k.shape[1]))
    L = np.concatenate([K.T] + (n-1)*[np.full_like(K.T,0.0)], axis=1)
    L = L.reshape((-1,K.shape[0])).T
    return(L)

def calculate_overlap(shifting, p = 0.6, n = 3):
    """Calculate 1D overlap matrix for given shifting
    
    The output is in units of output pixels: full overlap by input pixel means overlap 1, etc.
    For formulas, see sheet Formulas calculation overlap.
    """ 
    # Maximum and minimum overlapping pixels in units of output
    _min = int(np.floor((shifting + (1-p)/2)*n))
    _max = int(np.ceil((shifting + (1+p)/2)*n) - 1)

    if (_max == _min):
        overlap = np.array([p*n]) #overlap calculation in else gives wrong value in this case
    else: 
        # Calculate overlaps in units of outputpixels
        min_overlap = 1 - (shifting + (1-p)/2)*n + _min
        max_overlap = -_max + (shifting + (1+p)/2)*n
        res_overlap = 1.0 #overlap for intermediate subpixels
        
        # Calculate overlap matrix in units of outputpixels
        overlap = np.full(_max - _min + 1, res_overlap) 
        overlap[0] = min_overlap
        overlap[-1] = max_overlap
    return (overlap,_min)

def enlarge_overlaparray(overlap,_min):
    """Add zeros such that a shift of _min can be done in the convolution.
    
    For _min>=0 we can simply enlarge the array with the number of shiftvalues. 
    For _min<0 we need to add zeros such that the origin (shift) in the convolution 
    function is in the array.
    """
    if _min >= 0:
        return(np.append(np.zeros(_min),overlap))
    else:
        return(np.append(overlap,np.zeros(max(0,-_min-len(overlap))+1)))
    
def shift_matrix(K, shift):
    """ Shifts 2D matrix with integer shift
    
    This function does the same as ndimage.shift, but only for integer shifts.
    We need case separation for shift is >0, == 0, and <0 for both x- and y-shift.
    
    """
    L = np.zeros_like(K)
    if shift[0] > 0:
        if shift[1] > 0:
            L[shift[0]:,shift[1]:] = K[:-shift[0],:-shift[1]] 
        elif shift[1] == 0:
            L[shift[0]:,:] = K[:-shift[0],:] 
        else:
            L[shift[0]:,:shift[1]] = K[:-shift[0],-shift[1]:] 
            
    elif shift[0] == 0:
        if shift[1] > 0:
            L[:,shift[1]:] = K[:,:-shift[1]] 
        elif shift[1] < 0:
            L[:,:shift[1]] = K[:,-shift[1]:] 
        else:
            L = K
        
    else:
        if shift[1] > 0:
            L[:shift[0],shift[1]:] = K[-shift[0]:,:-shift[1]] 
        elif shift[1] == 0:
            L[:shift[0],:] = K[-shift[0]:,:] 
        else:
            L[:shift[0],:shift[1]] = K[-shift[0]:,-shift[1]:] 
    
    return L

def drizzle_image_only_convolution(k, shifting = [0,0], p = 0.6, n = 3):
    """Apply the drizzle algorithm for input image k with given parameters.
    
    For integer value k matrices the return is also integer valued (rounded).
    Scale factor is s = 1/n with n integer. Parameter p is the pixfrac, i.e.
    the linear size of the drop with respect to linear size of input pixel.
    Parameter shifting is the shifting to drizzle in units of the linear size of 
    the input pixel. 
    
    Performance:
    This function uses convolution with big matrices and no matrix shifting. It is 
    useful for large matrices. The function drizzle_image_matrix_shifting does the 
    same and is somewhat faster for small matrices or large shifts.
    """
    n = int(n) # Prevent for non-integer values of n

    K = enlarge_matrix_entries(k,n)  
    K = K.astype(float) # To prevent for rounding errors in convolution
    
    x_overlap,x_conv_shift = calculate_overlap(-shifting[0],p,n)
    y_overlap,y_conv_shift = calculate_overlap(-shifting[1],p,n)
    
    # Extend overlaparrays with zeros such that convolution goes well
    x_overlap = enlarge_overlaparray(x_overlap,x_conv_shift)
    y_overlap = enlarge_overlaparray(y_overlap,y_conv_shift)
    
    # Calculate convolution matrix
    m = np.outer(x_overlap, y_overlap)
    #print(m)
    
    # Calculate origin of convolution matrix
    x_orig = -(m.shape[0]//2) # Take origin upper left
    y_orig = -(m.shape[1]//2)
    
    x_orig -= min(x_conv_shift,0) # Change origin for negative shift on convolution
    y_orig -= min(y_conv_shift,0)
    
    # Calculate convolution
    K = ndimage.convolve(K, m, mode='constant', cval=0.0, origin = [x_orig,y_orig])
    
    # Calculate weight
    w = np.ones(np.shape(k))
    W = enlarge_matrix_entries(w,n)
    weight = ndimage.convolve(W, m, mode='constant', cval=0.0, origin = [x_orig,y_orig])
    #print(weight)
    
    return (K,weight)

def drizzle_image_matrix_shifting(k, shifting = [0,0], p = 0.6, n = 3):
    """Apply the drizzle algorithm for input image k with given parameters.
    
    For integer value k matrices the return is also integer valued (rounded).
    Scale factor is s = 1/n with n integer. Parameter p is the pixfrac, i.e.
    the linear size of the drop with respect to linear size of input pixel.
    Parameter shifting is the shifting to drizzle in units of the linear size of 
    the input pixel. 
    
    Performance:
    This function uses matrix shifting and convolution with a small matrix. It is 
    only useful in some cases e.g. small (100x100) input matrices. The function 
    drizzle_image_only_convolution does the same and is faster for large matrices.
    
    """
    n = int(n) # Prevent for non-integer values of n

    
    K = enlarge_matrix_entries(k,n)  
    K = K.astype(float) # To prevent for rounding errors in convolution

    x_overlap,x_conv_shift = calculate_overlap(-shifting[0],p,n)
    y_overlap,y_conv_shift = calculate_overlap(-shifting[1],p,n)    
    
    # Calculate convolution matrix
    m = np.outer(x_overlap, y_overlap)
    #print(m)
    
    # Calculate origin of convolution matrix
    x_orig = -(m.shape[0]//2) # Take origin upper left
    y_orig = -(m.shape[1]//2)
    
    # Calculate convolution
    K = ndimage.convolve(K, m, mode='constant', cval=0.0, origin = [x_orig,y_orig])
    
    # Shift output matrix with conv_shift
    K = shift_matrix(K,[x_conv_shift,y_conv_shift])
    
    
    # Calculate weight
    w = np.ones(np.shape(k))
    W = enlarge_matrix_entries(w,n)
    weight = ndimage.convolve(W, m, mode='constant', cval=0.0, origin = [x_orig,y_orig])
    
    weight = shift_matrix(weight,[x_conv_shift,y_conv_shift])
    #print(weight)

    return (K,weight)

def drizzle_image_for_loops(input_image, shift = [0,0], p = 0.5, s = 0.6):    
    """Add a drizzled image of the input image with shift to output image.
    
    An input image is shifted and added. The parameter p is the pixfrac. The
    parameter s is the scale factor. This function works for any scale factor 
    smaller or equal 1. 
    This drizzle implementation is slower than drizzle_image_only_convolution() and
    drizzle_image_matrix_shifting() but it handles general scalefactor. 
    
    The speed of this function could be improved by using numba.
    """
    
    # Calculate size of input image
    x_len_i = len(input_image)
    y_len_i = len(input_image[0])
    
    # Create output image and weight array
    x_len_o = int(np.ceil(x_len_i/s)) # x length of output image
    y_len_o = int(np.ceil(y_len_i/s))
    
    output_image = np.zeros([x_len_o,y_len_o])
    
    shift_x = -shift[0]
    shift_y = -shift[1]
    
    x_i = np.arange(0, x_len_i)
    y_i = np.arange(0, y_len_i)
    
    a = np.zeros((x_len_o,y_len_o)) # overlap matrix
    
    a_x = np.zeros((x_len_i,x_len_o))
    a_y = np.zeros((y_len_i,y_len_o))
    
    
    for x1 in x_i: 
        lower_x = max(int(np.floor((x1+shift_x+(1-p)/2)/s - 1)),0)
        upper_x = min(int(np.ceil((x1+shift_x+(1-p)/2+p)/s)),x_len_o)
        x_o = np.arange(lower_x, upper_x)
        for x2 in x_o:
            a_x[x1][x2] = max(min((x2+1)*s,x1+shift_x+(1-p)/2+p)-max(x2*s,x1+shift_x+(1-p)/2),0)
            
            for y1 in y_i:
                lower_y = max(int(np.floor((y1+shift_y+(1-p)/2)/s - 1)),0)
                upper_y = min(int(np.ceil((y1+shift_y+(1-p)/2+p)/s)),y_len_o)
                y_o = np.arange(lower_y, upper_y)

                for y2 in y_o:
                    a_y[y1][y2] = max(min((y2+1)*s,y1+shift_y+(1-p)/2+p)-max(y2*s,y1+shift_y+(1-p)/2),0)
                    
                    overlap = a_x[x1][x2]*a_y[y1][y2]
                    a[x2][y2] += overlap #total overlap
                    output_image[x2][y2] += overlap * input_image[x1][y1]
    
    weight = a

    return (output_image,weight)
    
def drizzle_image(k, shifting = [0,0], p = 0.6, n = 3, matrix_shifting = False, s = None):
    """Apply drizzle to input image k with a given shift.
    
    If parameter s (scale factor) is assigned the slow drizzle function with for loops 
    is called. 
    
    Else the image is drizzled using matrix_shifting or only_convolution. The only_convolution
    is faster for large matrices, while for small matrices (e.g. 100x100) the matrix_shifting
    function is faster. The difference in time cost is maximum approx. 10%. 
    Default is only_convolution.
    
    """
    if s != None:
        return drizzle_image_for_loops(k, shifting, p, s)
    elif matrix_shifting:
        return drizzle_image_matrix_shifting(k, shifting, p, n)
    else:
        return drizzle_image_only_convolution(k, shifting, p, n)
    return
    
def get_image(K, weight, plot = True, matrix = False, return_matrix = False):
    """Calculate image from sum of drizzled output images and weight of the images.
    
    There are options to plot, print and return the output image or matrix.
    """
    weight = weight + [weight==0] #if weigth is 0, make it 1 to prevent devision by 0

    K = K/weight
    if plot:
        io.imshow(K[0])
        io.show()
    if matrix:
        print(K[0])
    if return_matrix:
        return(K[0])
    return

def drizzle(ims, shifts, p = 0.6, n = 3, plot = False, matrix = False, return_matrix = True, 
            matrix_shifting = False, s = None):
    """Apply drizzle to input images ims with given shifts and parameters p and n.
    
    ims is a 3 dimensional matrix of multiple 2 dimensional input images. Shifts is an 
    2 dimensional array with the shift of each image in the x- and y- direction. Parameter
    p is the pixfrac. Parameter n is s=1/n with s the scale factor of the output image.
    If parameter s is used as scale factor, the slow implementation with for loops is used.
    
    The output image is returned as matrix. There are options to plot and print the output
    image and matrix.
    """

    if s == None: #scale factor is 1/n
        L = np.zeros(n*np.array(ims[0].shape), dtype = float)
        W = np.zeros(n*np.array(ims[0].shape), dtype = float)
    else: #scale factor is s
        L = np.zeros(np.array(np.ceil(np.array(ims[0].shape)/s),dtype=int), dtype = float)
        W = np.zeros(np.array(np.ceil(np.array(ims[0].shape)/s),dtype=int), dtype = float)


    for im in range(len(ims)):
        l,w = drizzle_image(ims[im],shifts[im],p,n,matrix_shifting,s=s)
        L += l 
        W += w

    F = get_image(L,W,plot,matrix,True)
    
    if return_matrix:
        return F
    else:
        return

#======================================================================================


# Examples to use implementation of drizzle
"""
from skimage import data

# Drizzle single image
n = 3
p = 0.8
shift = [5.4,-15.2]
k = data.coins()
K,weight=drizzle_image(k, shift, p, n)
get_image(K, weight)


# Drizzle multiple images
n = 1
p = 0.6
ims = [data.coins(),data.coins(),data.coins()]
shifts = [[2.4,2.8],[3,4.865],[-2.4,7.90]]
drizzled_im = drizzle(ims,shifts,p,n)
io.imshow(drizzled_im)
"""

