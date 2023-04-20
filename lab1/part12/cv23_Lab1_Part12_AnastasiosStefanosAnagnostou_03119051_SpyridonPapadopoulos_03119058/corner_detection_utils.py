import cv2
import numpy as np
from intro_utils import myfilter
from intro_utils import smooth_gradient
from intro_utils import InterestPointCoord
from intro_utils import LogMetric
import matplotlib.pyplot as plt

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

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(lminus, cmap='gray')
    axs[0].set_title("l- eigenvalue")
    axs[1].imshow(lplus, cmap='gray')
    axs[1].set_title("l+ eigenvalue")
    plt.show(block=False)
    plt.pause(0.01)
    plt.savefig("image-plots/corner-detection-eigenvalues.jpg")

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

    gradsxx = []
    gradsyy = []
    corners_per_scale = []
    for (s, r) in params:
        # calculate the gradient on both directions 
        gradx, grady, gradxx, _, gradyy = smooth_gradient(image, s, 3)
        
        # calculate whatevere these elements are 
        Gr = myfilter(r, "gaussian")
        j1 = cv2.filter2D(gradx * gradx, -1, Gr)
        j2 = cv2.filter2D(gradx * grady, -1, Gr)
        j3 = cv2.filter2D(grady * grady, -1, Gr)

        # efficient way to calculate the eigenvalues
        temp = j1 + j3 
        lplus = 1/2*(temp + np.sqrt( (j1 - j3)**2 + 4*j2**2))
        lminus = temp - lplus 
        # calculate the cornerness criterion
        r = lplus * lminus - k*((lplus + lminus)**2)
        
        indices = InterestPointCoord(r, s, theta)
        scale = s*np.ones((indices.shape[0], 1))
        corners = np.concatenate((indices, scale), axis=1)
        
        gradsxx.append(gradxx)
        gradsyy.append(gradyy)
        corners_per_scale.append(corners)

    grads = list(zip(scales, gradsxx, gradsyy))
    logs = [(s**2)*np.abs(xx + yy) for (s, xx, yy) in grads]
    return LogMetric(logs, corners_per_scale, N)
