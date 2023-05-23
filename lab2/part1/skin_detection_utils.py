import cv2
import numpy as np
from scipy import stats
from scipy import ndimage

SKIN_THRESHOLD = 0.05

def FitSkinGaussian(ycbcr_image):
    """ Return mean and covariance of interest points' gaussian distribution
    
    Keyword arguments:
    ycbcr_image -- input image in ycbcr color space.
    """
    cbcr_image = ycbcr_image[:, :, 1:3]
    height, width, channels = cbcr_image.shape
    # calculate the average of cb and cr channels
    mu = np.mean(cbcr_image.reshape(height*width, channels), axis=0)
    # calculate the covariance of cb and cr channels
    cov = np.cov(cbcr_image.reshape(height*width, channels).T)
    return (mu, cov)

def fd(image, mean, covariance):
    """ Return bounding box of area of interest.

    Keyword arguments:
    image -- the YCbCr input image.
    mean -- the mean value of the gaussian distribution.
    covariance -- the covariance of the gaussian distribution.
    """
    # define the gaussian distribution according to the mean and covariance
    distribution = stats.multivariate_normal(mean, covariance)
    
    # dstack means depth stack
    # we give the Cb channel and the Cr channel
    # and it forms an array of vectors (Cb, Cr)
    pixels = np.dstack((image[:,:,1], image[:,:,2]))
    # for the thresholding to work, it seems to be necessary to
    # normalize the probability. i'm not sure why.
    skin_image = (distribution.pdf(pixels)/np.max(distribution.pdf(pixels)) >= SKIN_THRESHOLD).astype('uint8')

    # the skin image probably has holes
    # we will attempt to close them by performing
    # the opening with a small structural element and
    # the closing with a bigger structural element
    opening_strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing_strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    skin_image = cv2.morphologyEx(skin_image, cv2.MORPH_OPEN, opening_strel)
    skin_image = cv2.morphologyEx(skin_image, cv2.MORPH_CLOSE, closing_strel)
    # find the connected components
    # and return the bounding boxes of the
    # connected components
    labels, features = ndimage.label(skin_image)
    boundaries = []
    for label in range(1, features+1):
        # find the bounding box of the feature
        # and append it to the list of boundaries
        indices = np.where(labels == label)
        x, y = indices[0], indices[1]
        boundaries.append((np.min(x), np.min(y), np.max(x)-np.min(x), np.max(y)-np.min(y)))
    return boundaries



