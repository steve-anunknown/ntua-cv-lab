import cv2
import numpy as np
import scipy.ndimage as scp


def HarrisDetector(video, s, sigma, tau, rho):
    """
    Harris Corner Detector
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames)
    s -- Gaussian kernel size
    sigma -- Gaussian kernel space standard deviation
    tau -- Gaussian kernel time standard deviation
    rho -- Harris response threshold
    """
    # normalize video
    video = video.astype(np.float32)/255
    # define Gaussian kernel
    space_size = int(2*np.ceil(3*sigma)+1)
    time_size = int(2*np.ceil(3*tau)+1)
    x_kernel = cv2.getGaussianKernel(space_size, sigma)[0]
    y_kernel = cv2.getGaussianKernel(space_size, sigma)[0]
    t_kernel = cv2.getGaussianKernel(time_size, tau)[0]
    for i, kernel in enumerate([x_kernel, y_kernel, t_kernel]):
        kernel = np.reshape(kernel, (1, 1, -1))
        kernel = np.repeat(kernel, video.shape[i], axis=i)
        video = scp.convolve(video, kernel, mode='constant')
    # compute gradients
    Ly, Lx, Lt = np.gradient(video)
    raise NotImplementedError


    

