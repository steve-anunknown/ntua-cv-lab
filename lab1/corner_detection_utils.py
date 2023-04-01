import cv2
import numpy as np
from intro_utils import myfilter
from intro_utils import smooth_gradient
from intro_utils import InterestPointCoord
from intro_utils import LogMetric

def CornerDetection(image, sigma, rho, theta, k):
    # Keep in mind that the image is smoothed
    # by default in the criterion function.
    # r = CornerCriterion(image, sigma, rho, k)

    # calculate the gradient on both directions 
    gradx, grady = smooth_gradient(image, sigma, 1)
    # calculate whatevere these elements are 
    Gr = myfilter(rho, "gaussian")
    j1 = cv2.filter2D(gradx * gradx, -1, Gr)
    j2 = cv2.filter2D(gradx * grady, -1, Gr)
    j3 = cv2.filter2D(grady * grady, -1, Gr)

    # efficient way to calculate the eigenvalues
    temp = j1 + j3 
    lplus = 1/2*(temp + np.sqrt( (j1 - j3)**2 + 4*j2**2))
    lminus = temp - lplus 

    # calculate the cornerness criterion
    r = lplus * lminus - k*((lplus + lminus)**2)
    
    indices = InterestPointCoord(r, sigma, theta)
    scale = sigma*np.ones((indices.shape[0], 1))
    corners = np.concatenate((indices, scale), axis=1)

    return corners

def HarrisLaplacian(image, sigma, rho, theta, k, scale, N):
    # Multiscale Corner Detection
    scales = [scale**i for i in list(range(N))]
    sigmas = [scale * sigma for scale in scales]
    rhos = [scale * rho for scale in scales]
    params = list(zip(sigmas, rhos))
    # call the edge detection function for each pair of parameters
    # if the image is MxM, then the resulting array is
    # M x (3*N), because the CornerDetection method returns a M x 3 array
    # and the iterations happens N times, once for every scale.
    corners_per_scale = [CornerDetection(image, s, r, theta, k) for (s, r) in params]

    # now we calculate the LoG for the pixels of every scale
    gradsscales = [smooth_gradient(image, s, 2) for s in sigmas]
    gradsxx = [gs[0] for gs in gradsscales]
    gradsyy = [gs[2] for gs in gradsscales]
    grads = list(zip(scales, gradsxx, gradsyy))
    logs = [(s**2)*np.abs(xx + yy) for (s, xx, yy) in grads]
    return LogMetric(logs, corners_per_scale, N)
