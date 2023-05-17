import cv2
from scipy import stats
from scipy import ndimage

def fd(image, mean, covariance):
    """ Return bounding box of area of interest.

    Keyword arguments:
    image -- the image which the area of interest is located in
    mean -- the mean value of the gaussian distribution
    covariance -- the covariance of the gaussian distribution
    """

    
