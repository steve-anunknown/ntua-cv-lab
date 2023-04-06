def myfilter(sigma, method):
    if (not (method == "gaussian" or method == "log")):
        print("Error: method has to be either \"gaussian\" or \"log\"")
        exit(2)
    n = int(2*np.ceil(3*sigma)+1)
    gauss1D = cv2.getGaussianKernel(n, sigma) # Column vector
    gauss2D = gauss1D @ gauss1D.T # Symmetric gaussian kernel
    if (method == "gaussian"):
        return gauss2D
    laplacian = np.array([[0,1,0],
        [1,-4,1],
        [0,1,0]])
    # perform the convolution between the gaussian kernel
    # and the laplacian, in order to create the log kernel
    logkernel = my2dconv(gauss2D, laplacian)
    return logkernel

def smooth_gradient(image, sigma, deg):
    # define the filters according to the arguments
    n = int(2*np.ceil(3*sigma)+1)
    gauss1D = cv2.getGaussianKernel(n, sigma) # Column vector
    gauss2D = gauss1D @ gauss1D.T # Symmetric gaussian kernel
    # smoothen the image
    smooth = cv2.filter2D(image, -1, gauss2D)
    # calculate the gradient on both directions 
    gradx, grady = np.gradient(smooth)
    gradxx, gradxy = np.gradient(gradx)
    _, gradyy = np.gradient(grady)
    return (gradxx, gradxy, gradyy)

def smooth_gradient(image, sigma, deg):
    # define the filters according to the arguments
    Gs = myfilter(sigma, "gaussian")
    # smoothen the image
    smooth = cv2.filter2D(image, -1, Gs)
    # calculate the gradient on both directions 
    gradx, grady = np.gradient(smooth)
    if (deg==1):
        return (gradx, grady)
    elif (deg==2):
        gradxx, gradxy = np.gradient(gradx)
        _ , gradyy = np.gradient(grady)
        return (gradxx, gradxy, gradyy)
    elif (deg==3):
        gradxx, gradxy = np.gradient(gradx)
        _ , gradyy = np.gradient(grady)
        return (gradx, grady, gradxx, gradxy, gradyy)
    print("deg = 1 for (gradx, grady))"
    print("deg = 2 for (gradxx, gradxy, gradyy)")
    print("deg = 3 for (gradx, grady, gradxx, gradxy, gradyy)")
    exit(2)

def BlobDetection(image, sigma, theta):
    gradxx, gradxy, gradyy = smooth_gradient(image, sigma, 2)
    r = gradxx * gradyy - gradxy * gradxy # determinant of hessian

    indices = InterestPointCoord(r, sigma, theta)
    scale = sigma*np.ones((indices.shape[0], 1))
    blobs = np.concatenate((indices, scale), axis=1)
    return blobs

def BoxDerivative(ii, sigma):
    def roi(ii, x, y, padw):
        # rectangle of interest
        # the input image is padded, otherwise the shifts
        # would have peculiar results

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
