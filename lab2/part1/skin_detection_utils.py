import cv2
import numpy as np
from scipy import stats
from scipy import ndimage

SKIN_THRESHOLD = 0.1

def FitSkinGaussian(ycbcr_image):
    """ Return mean and covariance of interest points' gaussian distribution
    
    Keyword arguments:
    ycbcr_image -- input image in ycbcr color space.
    """
    cbcr_image = ycbcr_image[:, :, 1:3]
    height, width, channels = cbcr_image.shape
    # calculate the average of cb and cr channels
    mu = np.mean(cbcr_image[:, :, :], axis=0)
    # calculate the covariance of cb and cr channels
    cov = np.cov(cbcr_image[:, :, 0].flatten(), cbcr_image[:, :, 1].flatten())

    return (mu, cov)

def fd(image, mean, covariance):
    """ Return bounding box of area of interest.

    Keyword arguments:
    image -- the YCbCr input image.
    mean -- the mean value of the gaussian distribution.
    covariance -- the covariance of the gaussian distribution.
    """

    distribution = scipy.stats.multivariate_normal(mean, covariance)
    
    # dstack means depth stack
    # we give the Cb channel and the Cr channel
    # and it forms an array of vectors (Cb, Cr)
    pixels = np.dstack((image[:,:,1], image[:,:,2]))
    skin_image = (distribution.pdf(pixels) >= SKIN_THRESHOLD).astype(int)

    # the skin image probably has holes
    # we will attempt to close them by performing
    # the opening with a small structural element and
    # the closing with a bigger structural element
    opening_strel = np.ones((3,3))
    closing_strel = np.ones((9,9))
    skin_image = cv2.morphologyEx(skin_image, cv2.MORPH_OPEN, opening_strel)
    skin_image = cv2.morphologyEx(skin_image, cv2.MORPH_CLOSE, closing_strel)
    labels, features = np.ndimage.label(skin_image)
    boundaries = []
    for label in labels:
        x, y = np.where(features == label)
        x = x.sort()
        y = y.sort()
        width = x[-1] - x[0]
        height = y[-1] - y[0]
        boundaries.append( x[0], y[0], width, height)
    return boundaries



