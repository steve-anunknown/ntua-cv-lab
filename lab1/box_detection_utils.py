import numba
import numpy as np
from intro_utils import LogMetric
from intro_utils import InterestPointCoord


def IntegralImage(i):
    # this definitely works
    return np.cumsum(np.cumsum(i, axis=0), axis=1)

@numba.jit
def BoxDerivative(ii, sigma):
    # this needs extremely hard checking

    # THIS METHOD IS SUPPOSED TO BE MUCH FASTER
    # THAN CALCULATING THE SECOND ORDER DERIVATIVE
    # USING CONVOLUTIONS (smooth_gradient function).
    # HOWEVER IT IS MUCH SLOWER THAN EXPECTED
    # THIS CAN MEAN TWO THINGS
    # 1) I'M DOING SOMETHING WRONG HERE (BoxDerivative function)
    # 2) I WAS DOING SOMETHING WRONG EARLIER (smooth_gradient function)

    # The smooth gradient functions uses the np.gradient function,
    # that is precompiled code and runs much faster than python 'for' loops
    # As per my understanding, the following for loop cannot be expected
    # to match the speed of the other method.

    n = int(2*np.ceil(3*sigma) + 1)
    height = int(4*np.floor(n/6) + 1)
    width = int(2*np.floor(n/6) + 1)
    padding = int(np.ceil(n/2))
    mid = int(np.ceil((n-height)/2))
    iip = np.pad(ii, ((padding, 0), (padding, 0)), 'constant')
    iip = np.pad(iip, ((0, padding), (0, padding)), 'edge')
    
    x, y = ii.shape
    lxx = np.zeros((x,y))
    lyy = np.zeros((x,y))
    lxy = np.zeros((x,y))
    for ix in range(x):
        tlx = ix + mid
        for iy in range(y):
            # ============ calculate x gradient ============ #
            tl = (tlx, iy)
            tr = (tlx, iy + width - 1)
            bl = (tlx + height, iy)
            br = (tlx + height, iy + width - 1)
            lxx[ix, iy] = (iip[tl] - iip[tr] + iip[br] - iip[bl])

            tl = (tlx, iy + width)
            tr = (tlx, iy + 2*width - 1)
            bl = (tlx + height - 1, iy + width)
            br = (tlx + height - 1, iy + 2*width - 1)
            lxx[ix, iy] -= 2*(iip[tl] - iip[tr] + iip[br] - iip[bl])

            tl = (tlx, iy + 2*width)
            tr = (tlx, iy + 3*width - 1)
            bl = (tlx + height - 1, iy + 2*width)
            br = (tlx + height - 1, iy + 3*width - 1)
            lxx[ix, iy] += (iip[tl] - iip[tr] + iip[br] - iip[bl])

            # ============ calculate y gradient ============ #            
            tly = iy + mid
            tl = (ix, tly)
            tr = (ix, tly + height - 1)
            bl = (ix + width - 1, tly + height - 1)
            br = (ix + width - 1, tly)
            lyy[ix, iy] = (iip[tl] - iip[tr] + iip[br] - iip[bl])

            tl = (ix + width, tly + height - 1)
            tr = (ix + width, tly)
            bl = (ix + 2*width - 1, tly)
            br = (ix + 2*width - 1, tly + height - 1)
            lyy[ix, iy] -= 2*(iip[tl] - iip[tr] + iip[br] - iip[bl])

            tl = (ix + 2*width, tly)
            tr = (ix + 2*width, tly + height - 1)
            bl = (ix + 3*width - 1, tly)
            br = (ix + 3*width - 1, tly + height - 1)
            lyy[ix, iy] += (iip[tl] - iip[tr] + iip[br] - iip[bl])

            # ============ calculate xy gradient ============ #
            tlh = ix + 1
            tlw = iy + 1
            tl = (tlh, tlw)
            tr = (tlh, tlw + width - 1)
            bl = (tlh + width - 1, tlw)
            br = (tlh + width - 1, tlw + width - 1)
            lxy[ix, iy] = (iip[tl] - iip[tr] + iip[br] - iip[bl])

            tl = (tlh, tlw + width + 1)
            tr = (tlh, tlw + 2*width)
            bl = (tlh + width - 1, tlw + width + 1)
            br = (tlh + width - 1, tlw + 2*width)
            lxy[ix, iy] -= (iip[tl] - iip[tr] + iip[br] - iip[bl])

            tl = (tlh + width + 1, tlw + width + 1)
            tr = (tlh + width + 1, tlw + 2*width )
            bl = (tlh + 2*width, tlw + width + 1)
            br = (tlh + 2*width, tlw + 2*width)
            lxy[ix, iy] = (iip[tl] - iip[tr] + iip[br] - iip[bl])

            tl = (tlh + width + 1, tlw)
            tr = (tlh + width + 1, tlw + width - 1)
            bl = (tlh + 2*width, tlw)
            br = (tlh + 2*width, tlw + width - 1)
            lxy[ix, iy] -= (iip[tl] - iip[tr] + iip[br] - iip[bl])

    return (lxx, lxy, lyy)

def BoxFilters(image, sigma, theta):
    ii = IntegralImage(image)
    lxx, lxy, lyy = BoxDerivative(ii, sigma)
    r = lxx*lyy - (0.9*lxy)**2
    indices = InterestPointCoord(r, sigma, theta)
    scale = sigma*np.ones((indices.shape[0], 1))
    blobs = np.concatenate((indices, scale), axis=1)
    return blobs

def BoxLaplacian(image, sigma, theta, scale, N):
    scales = [scale**i for i in list(range(N))]
    sigmas = [scale * sigma for scale in scales]
    ii = IntegralImage(image)    

    gradsxx = []
    gradsyy = []
    blobs_per_scale = []
    for s in sigmas:
        lxx, lxy, lyy = BoxDerivative(ii, s)
        r = lxx*lyy - (0.9*lxy)**2
        r = (r - r.min())/r.max() # turn to binary
        indices = InterestPointCoord(r, s, theta)
        scale = s*np.ones((indices.shape[0], 1))
        blobs = np.concatenate((indices, scale), axis=1)

        gradsxx.append(lxx)
        gradsyy.append(lyy)
        blobs_per_scale.append(blobs)

    grads = list(zip(scales, gradsxx, gradsyy))
    logs = [(s**2)*np.abs(xx + yy) for (s, xx, yy) in grads]
    return LogMetric(logs, blobs_per_scale, N)