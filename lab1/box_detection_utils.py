import numpy as np
from intro_utils import LogMetric
from intro_utils import InterestPointCoord


def IntegralImage(i):
    # this definitely works
    return np.cumsum(np.cumsum(i, axis=0), axis=1)

def BoxDerivative(image, sigma):
    def roi(ii, x, y, padw):
        # rectangle of interest
        # the input image is padded, otherwise the shifts
        # would have peculiar results

        # therefore remeber to unpad in the end

        # take the whole integral image and move it around
        # the area outside of the rectangle will cancel itself out
        # leaving only the values inside the rectangle of interest
        shiftx = int((x + 1)/2 - 1)
        shifty = int((y + 1)/2 - 1)

        # this isolates the upper and lower line
        upline = np.roll(ii, shifty + 1, axis=0)
        loline = np.roll(ii, -shifty, axis=0)
        
        # this isolates the upper corners
        uprigh = np.roll(upline, -shiftx, axis=1)
        upleft = np.roll(upline, shiftx+1, axis=1)
        
        # this isolates the lower corners
        downri = np.roll(loline, -shiftx, axis=1)
        downle = np.roll(loline, shiftx+1, axis=1)
        
        result = upleft - uprigh + downri - downle
        # unpad in the end
        result = result[padw:-padw, padw:-padw]
        return result
    
    n = int(2*np.ceil(3*sigma) + 1)
    height = int(4*np.floor(n/6) + 1)
    width = int(2*np.floor(n/6) + 1)
    padding = 2 * width
    
    padded = np.pad(image, padding, 'reflect')
    ii = IntegralImage(padded)

    # sum the middle rectangle x(-3) and then add the whole rectangle
    # (0 -3 0) + (1 1 1) = (1 -2 1)
    lxx = -3*(roi(ii, width, height, padding)) + roi(ii, 3*width, height, padding)
    lyy = -3*(roi(ii, height, width, padding)) + roi(ii, height, 3*width, padding)

    # the xy derivative requires a little special handling
    smallbox = roi(ii, width, width, padding)
    padded = np.pad(smallbox, padding, 'reflect')
    shiftv = int((width-1)/2 + 1)

    ul = np.roll(padded, [shiftv, shiftv], axis=(0, 1))
    ur = np.roll(padded, [-shiftv, shiftv], axis=(0, 1))
    dl = np.roll(padded, [shiftv, -shiftv], axis=(0, 1))
    dr = np.roll(padded, [-shiftv, -shiftv], axis=(0, 1))

    lxy = ul + dr - ur - dl
    lxy = lxy[padding:-padding, padding:-padding]

    return (lxx, lxy, lyy)

def BoxFilters(image, sigma, theta):
    lxx, lxy, lyy = BoxDerivative(image, sigma)
    r = lxx*lyy - (0.9*lxy)**2
    indices = InterestPointCoord(r, sigma, theta)
    scale = sigma*np.ones((indices.shape[0], 1))
    blobs = np.concatenate((indices, scale), axis=1)
    return blobs

def BoxLaplacian(image, sigma, theta, scale, N):
    scales = [scale**i for i in list(range(N))]
    sigmas = [scale * sigma for scale in scales]

    gradsxx = []
    gradsyy = []
    blobs_per_scale = []
    for s in sigmas:
        lxx, lxy, lyy = BoxDerivative(image, s)
        r = lxx*lyy - (0.9*lxy)**2
        indices = InterestPointCoord(r, s, theta)
        scale = s*np.ones((indices.shape[0], 1))
        blobs = np.concatenate((indices, scale), axis=1)

        gradsxx.append(lxx)
        gradsyy.append(lyy)
        blobs_per_scale.append(blobs)

    grads = list(zip(scales, gradsxx, gradsyy))
    logs = [(s**2)*np.abs(xx + yy) for (s, xx, yy) in grads]
    return LogMetric(logs, blobs_per_scale, N)