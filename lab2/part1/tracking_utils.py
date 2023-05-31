import numpy as np
from cv2 import getGaussianKernel, filter2D, goodFeaturesToTrack
from scipy.ndimage import map_coordinates

def shift_image(image, shift):
    """ Shift the image

    Keyword arguments:
    image -- image to be shifted
    shift -- shift amount
    """
    dx, dy = shift
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    return map_coordinates(image,[np.ravel(y + dy), np.ravel(x + dx)], order=1).reshape(image.shape)

def lk(i1, i2, features, rho, epsilon, dx0, dy0):
    """ Lucas-Kanade algorithm
    
    Keyword arguments:
    i1 -- initial frame
    i2 -- next frame
    features -- coordinates of interest-points
    rho -- gaussian width parameter
    epsilon -- minimum constant to avoid zeroes
    dx0 -- initial guesses for movement of features in x axis
    dy0 -- initial guesses for movement of features in y axis
    returns -- [dx, dy] actual estimates for movement of features
    """
    # convert the images to float in [0,1]
    # for the parameters to make sense
    i1 = i1.astype(np.float)/255
    i2 = i2.astype(np.float)/255
    # define limit of iterations
    # and threshold of change
    limit = 150
    threshold = 0.001
    # setup the kernel for the convolutions
    # this kernel acts as a gaussian filter
    size = int(2*np.ceil(3*rho)+1)
    mid = (size-1)//2
    kernel = getGaussianKernel(size, rho)
    kernel = kernel @ kernel.T
    
    # initialize result vectors
    returnx = np.zeros(len(features))
    returny = np.zeros(len(features))
    
    for index, feature in enumerate(features):
        # get the coordinates of the feature
        x, y = feature
        # get area around the feature in both images
        initial_image = i1[max(0, y-mid):min(y+mid, i1.shape[0]),
                           max(0, x-mid):min(x+mid, i1.shape[1])]
        next_image = i2[max(0, y-mid):min(y+mid, i2.shape[0]),
                        max(0, x-mid):min(x+mid, i2.shape[1])]
        # compute the gradient of the initial image
        gradient_y, gradient_x = np.gradient(initial_image)

        # initialize iteration counter and change
        iterations = 0
        change = float('inf')
        dy, dx = dy0[index], dx0[index]
        while (iterations < limit and change > threshold):
            # shift the initial image by the current displacement
            shifted_image = shift_image(initial_image, [dx, dy])
            # shift the gradient of the initial image by the current displacement
            shifted_gradient_x = shift_image(gradient_x, [dx, dy])
            shifted_gradient_y = shift_image(gradient_y, [dx, dy])
            # compute the error between the shifted image and the next image
            error = next_image - shifted_image
            # compute the Lucas-Kanade equations
            s11 = filter2D(shifted_gradient_x**2, -1, kernel)[mid,mid] + epsilon
            s12 = filter2D(shifted_gradient_x*shifted_gradient_y, -1, kernel)[mid,mid]
            s22 = filter2D(shifted_gradient_y**2, -1, kernel)[mid,mid] + epsilon
            b1 = filter2D(shifted_gradient_x*error, -1, kernel)[mid,mid]
            b2 = filter2D(shifted_gradient_y*error, -1, kernel)[mid,mid]
            # compute the determinant of the system
            det = s11*s22 - s12*s12
            # compute the improvement
            delta_x = (s22*b1 - s12*b2)/det
            delta_y = (s11*b2 - s12*b1)/det
            # update the displacement estimates
            dx += delta_x
            dy += delta_y
            # compute the change
            change = np.linalg.norm([delta_x, delta_y])
            # update the number of iterations
            iterations += 1
        # update the displacement estimates
        returnx[index] = dx
        returny[index] = dy
    return np.array([returnx, returny])

def multiscale_lk(i1, i2, num_features, rho, epsilon, scale):
    """ Multiscale Lucas-Kanade algorithm
    
    Keyword arguments:
    i1 -- initial frame
    i2 -- next frame
    num_features -- number of features to track
    rho -- gaussian width parameter
    epsilon -- minimum constant to avoid zeroes
    scale -- number of scales in the pyramid
    returns -- [dx, dy] actual estimates for movement of features
    """
    # first define some helper functions
    # that do not need to be visible outside
    # of this function
    def pyramid(image, levels):
        """ Create a pyramid of images
        
        Keyword arguments:
        image -- image to be scaled
        levels -- number of levels in the pyramid
        """
        
        def downscale(image):
            """ Downscale the image by a factor of 2
            
            Keyword arguments:
            image -- image to be downscaled
            """
            gauss = getGaussianKernel(3, 1)
            gauss = gauss @ gauss.T
            return filter2D(image, -1, gauss)[::2,::2]
        
        result = [image]
        for i in range(levels - 1):
            result.append(downscale(result[i]))
        
        result.reverse()
        return result
    
    # we should normalize the images,
    # but we do not need to do it here
    # because we already do it in lk.

    # therefore, build the pyramids
    # with the original images
    # from deep level to shallow level.
    pyramid1 = pyramid(i1, scale)
    pyramid2 = pyramid(i2, scale)

    # initialize result vectors.
    dx0, dy0 = np.zeros(num_features), np.zeros(num_features)
    
    for level in range(scale):
        features = np.squeeze(goodFeaturesToTrack(pyramid2[level], num_features, 0.05, 5).astype(int))
        
        [dx, dy] = lk(pyramid1[level], pyramid2[level], features, rho, epsilon, dx0, dy0)
        [flowx, flowy] = displ(dx, dy, 0.7)
        # update the initial guesses
        if level < scale - 1:
            dx0 = np.full((num_features), 2*flowx)
            dy0 = np.full((num_features), 2*flowy)

    return np.array([dx, dy])

def displ(dx, dy, threshold):
    """ Display the optical flow
    
    Keyword arguments:
    dx -- displacement x axis
    dy -- displacement y axis
    threshold -- threshold for the optical flow
    """
    energies = np.array([x**2 + y**2 for x, y in zip(dx, dy)])
    mean_energy = np.mean(energies)
    energy = np.array([np.array([dx, dy])
                       for x, y in zip(dx, dy)
                       if (x**2 + y**2) > threshold*mean_energy])
    
    # check if there are any features
    # with enough energy
    if energy.shape[0] == 0:
        return [0, 0]
    return [np.mean(energy[:,0]), np.mean(energy[:,1])]
