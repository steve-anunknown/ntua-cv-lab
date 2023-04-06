import numpy as np
import cv2
from intro_utils import myfilter
from intro_utils import my2dconv

def EdgeDetect(image, sigma, theta, method):
    if (not (method == "linear" or method == "nonlinear")):
        print("Error: method has to be either \"linear\" or \"nonlinear\"")
        exit(2)
    
    gaussf = myfilter(sigma, "gaussian")
    smooth = cv2.filter2D(image, -1, gaussf)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    if (method == "linear"):
        # construct the laplacian of gaussian kernel
        # and use it to filter the image
        logfilter = myfilter(sigma, "log")
        #imgloged = cv2.filter2D(image, -1, logfilter)
        imgloged = my2dconv(image, logfilter)
    elif (method == "nonlinear"):
        # Perform morphological operations using
        # the cross structuring element
        imgloged = cv2.dilate(smooth, cross) + cv2.erode(smooth, cross) - 2*smooth
    
    # the imgloged variable is visible only if one of the if blocks
    # get executed. hopefully, this always happens
    L = imgloged
    
    # type uin8 is needed for compatibility with the
    # dilate and erode functions. otherwise, the matrix's
    # elements would have boolean type.
    X = (L > 0).astype(np.uint8)
    Y = (cv2.dilate(X, cross)) - (cv2.erode(X, cross))

    gradx, grady = np.gradient(smooth)
    grad = np.abs(gradx + 1j * grady)
    D = ((Y == 1) & (grad > (theta * np.max(grad))))
    return D

def QualityMetric(real, computed):
    # use the following names for compatibility
    # with the project's guide.
    T = real
    D = computed
    DT = (D & T)
    # the matrices are supposed to be boolean
    # therefore the sum() functions counts the
    # elements that are true / 1.
    cardT = T.sum()
    cardD = D.sum()
    cardDT = DT.sum()

    prTD = cardDT/cardT
    prDT = cardDT/cardD

    C = (prDT + prTD)/2
    return C
