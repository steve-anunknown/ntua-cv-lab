import numpy as np
from intro_utils import LogMetric
from intro_utils import InterestPointCoord

import time
from intro_utils import smooth_gradient


def IntegralImage(i):
    # this definitely works
    return np.cumsum(np.cumsum(i, axis=0), axis=1)

def BoxDerivative(ii, sigma):
    # this needs extremely hard checking

    # THIS METHOD IS SUPPOSED TO BE MUCH FASTER
    # THAN CALCULATING THE SECOND ORDER DERIVATIVE
    # USING CONVOLUTIONS (smooth_gradient function).
    # HOWEVER IT IS MUCH SLOWER THAN EXPECTED
    # THIS CAN MEAN TWO THINGS
    # 1) I'M DOING SOMETHING WRONG HERE (BoxDerivative function)
    # 2) I WAS DOING SOMETHING WRONG EARLIER (smooth_gradient function)

    
    print("initializing box filtering")
    n = int(2*np.ceil(3*sigma) + 1)
    x, y = ii.shape

    lxx = np.zeros((x,y))
    lyy = np.zeros((x,y))
    lxy = np.zeros((x,y))
    height, width = int(4*np.floor(n/6) + 1), int(2*np.floor(n/6) + 1)

    heightx, widthx = height, width
    midhx, midwx = int((heightx - 1)/2), int((widthx - 1)/2)
    heighty, widthy = width, height
    midhy, midwy = int((heighty - 1)/2), int((widthy - 1)/2)
    heightxy, widthxy = width, width

    # pad with zeroes so as to handle the edge cases
    # height is greater than width, therefore pad with that
    ii = np.pad(ii, ((height, height), (height, height)), mode='constant')
    print("initializing box filtering over")

    # (ix, iy) is the centre of the box. iterate through every
    # pixel of the integral image
    print("Entering for loop")
    for ix in range(height, x):
        # these are some optimization tricks
        # don't let them scare you
        heightx1 = ix - midhx
        heightx2 = ix + midhx
        for iy in range(height, y):
            widthx1 = iy - midwx
            widthx2 = iy + midwx 
            widthy1 = iy - midwy
            widthy2 = iy + midwy 

            tl1x = ii[heightx1, widthx1 - widthx]
            tr1x = ii[heightx1, widthx1]
            br1x = ii[heightx2, widthx1]
            bl1x = ii[heightx2, widthx1 - widthx]

            tl2x = ii[heightx1, widthx1]
            tr2x = ii[heightx1, widthx2]
            br2x = ii[heightx2, widthx2]
            bl2x = ii[heightx2, widthx1]

            tl3x = ii[heightx1, widthx2]
            tr3x = ii[heightx1, widthx2 + widthx]
            br3x = ii[heightx2, widthx2 + widthx]
            bl3x = ii[heightx2, widthx2]

            tl1y = ii[ix - midhy - heighty, widthy1]
            tr1y = ii[ix - midhy - heighty, widthy2]
            br1y = ii[ix - midhy, widthy2]
            bl1y = ii[ix - midhy, widthy1]

            tl2y = ii[ix - midhy, widthy1]
            tr2y = ii[ix - midhy, widthy2]
            br2y = ii[ix + midhy, widthy2]
            bl2y = ii[ix + midhy, widthy1]

            tl3y = ii[ix + midhy, widthy1]
            tr3y = ii[ix + midhy, widthy2]
            br3y = ii[ix + midhy + heighty, widthy2]
            bl3y = ii[ix + midhy + heighty, widthy1]

            tl1xy = ii[ix - 1 - heightxy, iy - 1 - widthxy]
            tr1xy = ii[ix - 1 - heightxy, iy - 1]
            br1xy = ii[ix - 1, iy - 1]
            bl1xy = ii[ix - 1, iy - 1 - widthxy]

            tl2xy = ii[ix - 1 - heightxy, iy + 1]
            tr2xy = ii[ix - 1 - heightxy, iy + 1 + widthxy]
            br2xy = ii[ix - 1, iy + 1 + widthxy]
            bl2xy = ii[ix - 1, iy + 1]

            tl3xy = ii[ix + 1, iy + 1]
            tr3xy = ii[ix + 1, iy + 1 + widthxy]
            br3xy = ii[ix + 1 + heightxy, iy + 1 + widthxy]
            bl3xy = ii[ix + 1 + heightxy, iy + 1]

            tl4xy = ii[ix + 1, iy - 1 - widthxy]
            tr4xy = ii[ix + 1, iy - 1]
            br4xy = ii[ix + 1 + heightxy, iy - 1]
            bl4xy = ii[ix + 1 + heightxy, iy - 1 - widthxy]
            
            lxx[ix,iy] = (tl1x - tr1x + br1x - bl1x) - 2*(tl2x - tr2x + br2x - bl2x) + (tl3x - tr3x + br3x - bl3x)
            lyy[ix,iy] = (tl1y - tr1y + br1y - bl1y) - 2*(tl2y - tr2y + br2y - bl2y) + (tl3y - tr3y + br3y - bl3y)
            lxy[ix,iy] = (tl1xy - tr1xy + br1xy - bl1xy) - (tl2xy - tr2xy + br2xy - bl2xy) + (tl3xy - tr3xy + br3xy - bl3xy) - (tl4xy - tr4xy + br4xy - bl4xy)
        
    print("Exiting for loop")
    return (lxx, lxy, lyy)

def BoxFilters(image, sigma, theta):
    print("Timing integral image")
    start = time.time()
    ii = IntegralImage(image)
    end = time.time()
    print(end - start)

    print("Timing Box Derivative")
    start = time.time()
    lxx, lxy, lyy = BoxDerivative(ii, sigma)
    end = time.time()
    print(end - start)

    print("Timing Smooth Gradient")
    start = time.time()
    sxx, sxy, syy = smooth_gradient(image, sigma, 2)
    end = time.time()
    print(end - start)

    r = lxx*lyy - (0.9*lxy)**2
    indices = InterestPointCoord(r, sigma, theta)
    scale = sigma*np.ones((indices.shape[0], 1))
    blobs = np.concatenate((indices, scale), axis=1)
    return blobs

def BoxLaplacian(image, sigma, theta, scale, N):
    scales = [scale**i for i in list(range(N))]
    sigmas = [sigma*s for s in scales]
    ii = IntegralImage(image)    

    gradsxx = []
    gradsyy = []
    blobs_per_scale = []
    for s in sigmas:
        lxx, lxy, lyy = BoxDerivative(ii, s)
        r = lxx*lyy - (0.9*lxy)**2
        indices = InterestPointCoord(r, s, theta)
        scale = sigma*np.ones((indices.shape[0], 1))
        blobs = np.concatenate((indices, scale), axis=1)

        gradsxx.append(lxx)
        gradsyy.append(lyy)
        blobs_per_scale.append(blobs)

    grads = list(zip(scales, gradsxx, gradsyy))
    logs = [(s**2)*np.abs(xx + yy) for (s, xx, yy) in grads]
    return LogMetric(logs, blobs_per_scale, N)