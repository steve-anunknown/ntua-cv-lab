import cv2
import numpy as np
from cv23_lab1_part2_utils import disk_strel

def getpsnr(image, noisestd):
    return 20*np.log10((np.max(image)-np.min(image))/noisestd)

def getstd(image, psnr):
    return (np.max(image)-np.min(image))/(10**(psnr/20))

def my2dconv(image, kernel):
    ix, iy = image.shape
    nx, ny = kernel.shape
    result = np.zeros((ix + nx - 1, iy + ny - 1))
    padded = np.pad(image, [(nx//2, nx//2),(ny//2,ny//2)], mode='constant')
    for i in range(nx//2, ix + nx//2):
        for j in range(ny//2, iy + ny//2):
            result[i,j] = np.sum(padded[i-nx//2:i+nx//2+1,j-ny//2:j+ny//2+1] * kernel)
    return result[nx//2:ix+nx//2, ny//2:iy+ny//2]

def myfilter(sigma, method):
    if (not (method == "gaussian" or method == "log")):
        print("Error: method has to be either \"gaussian\" or \"log\"")
        exit(2)
    n = int(2*np.ceil(3*sigma)+1)
    gauss1D = cv2.getGaussianKernel(n, sigma) # Column vector
    gauss2D = gauss1D @ gauss1D.T # Symmetric gaussian kernel
    if (method == "gaussian"):
        return gauss2D
    laplacian = np.array([[0,1,0],
        [1,-4,1],
        [0,1,0]])
    # perform the convolution between the gaussian kernel
    # and the laplacian, in order to create the log kernel
    # for sake of demonstration, we use our own convolution
    # but later on we use the cv2 implementation.
    logkernel = my2dconv(gauss2D, laplacian)
    return logkernel

def smooth_gradient(image, sigma, deg):
    # define the filters according to the arguments
    Gs = myfilter(sigma, "gaussian")
    # smoothen the image
    smooth = cv2.filter2D(image, -1, Gs)
    # calculate the gradient on both directions 
    gradx, grady = np.gradient(smooth)
    if (deg==1):
        return (gradx, grady)
    elif (deg==2):
        gradxx, gradxy = np.gradient(gradx)
        _ , gradyy = np.gradient(grady)
        return (gradxx, gradxy, gradyy)
    elif (deg==3):
        gradxx, gradxy = np.gradient(gradx)
        _ , gradyy = np.gradient(grady)
        return (gradx, grady, gradxx, gradxy, gradyy)
    print("deg = 1 for (gradx, grady)")
    print("deg = 2 for (gradxx, gradxy, gradyy)")
    print("deg = 3 for (gradx, grady, gradxx, gradxy, gradyy)")
    exit(2)

def InterestPointCoord(r, sigma, theta):
    # r is a previously evaluated criterion
    # sigma is used for the size of the structure
    # theta is a threshold
    
    # evaluate the following 2 conditions 
    # condition 1
    ns = 2*np.ceil(3*sigma) + 1
    bsq = disk_strel(ns)
    cond1 = ( r == cv2.dilate(r, bsq) )
    # condition 2
    maxr = np.max(r)
    cond2 = ( r > theta * maxr )
    # choose the pixels that satisfy both of them
    # return their coordinates and their scale
    x, y = np.where(cond1 & cond2)
    # for compatibility with the utility function
    # provided by the lab staff, the y coordinate
    # has to come before the x coordinate
    indices = np.column_stack((y,x))
    return indices

def LogMetric(logs, itemsperscale, N):
    # log((x,y), s) = (s^2)|Lxx((x,y),s) + Lyy((x,y),s)|
    # returns the coordinates of the points that maximize
    # the log metric in a neighborhood of 3 scales
    # (prev scale), (curr scale), (next scale)
    final = []
    for index, items in enumerate(itemsperscale):
        logp = logs[max(index-1,0)]
        logc = logs[index]
        logn = logs[min(index+1,N-1)]
        for triplet in items:
            x = int(triplet[1])
            y = int(triplet[0])
            prev = logp[x,y]
            curr = logc[x,y]
            next = logn[x,y]
            if (curr >= prev) and (curr >= next):
                final.append(triplet)
    return np.array(final)
