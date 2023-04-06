import numpy as np
import time
from intro_utils import smooth_gradient
from intro_utils import InterestPointCoord
from intro_utils import LogMetric

def BlobDetection(image, sigma, theta):
    gradxx, gradxy, gradyy = smooth_gradient(image, sigma, 2)
    r = gradxx * gradyy - gradxy * gradxy # determinant of hessian
    indices = InterestPointCoord(r, sigma, theta)
    scale = sigma*np.ones((indices.shape[0], 1))
    blobs = np.concatenate((indices, scale), axis=1)
    return blobs

def HessianLaplacian(image, sigma, theta, scale, N):
    scales = [scale**i for i in list(range(N))]
    sigmas = [scale * sigma for scale in scales]
    
    gradsxx = []
    gradsyy = []
    blobs_per_scale = []
    for s in sigmas:
        lxx, lxy, lyy = smooth_gradient(image, s, 2)
        r = lxx * lyy - lxy * lxy # determinant of hessian
        indices = InterestPointCoord(r, s, theta)
        scale = s*np.ones((indices.shape[0], 1))
        blobs = np.concatenate((indices, scale), axis=1)
        
        gradsxx.append(lxx)
        gradsyy.append(lyy)
        blobs_per_scale.append(blobs)
    
    grads = list(zip(scales, gradsxx, gradsyy))
    logs = [(s**2)*np.abs(xx + yy) for (s, xx, yy) in grads]
    return LogMetric(logs, blobs_per_scale, N)
