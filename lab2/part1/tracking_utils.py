import numpy as np
from cv2 import getGaussianKernel, filter2D
from scipy.ndimage import map_coordinates
from matplotlib.pyplot import quiver

def shift_image(image, shift):
    """ Shift the image

    Keyword arguments:
    image -- image to be shifted
    shift -- shift amount
    """
    dx, dy = shift
    # perhaps this could also be written in the form
    # y, x = meshgrid( arange (shape[0]), arange (shape[1]) )
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    return map_coordinates(image, [np.ravel(y + dy), np.ravel(x + dx)], order=1).reshape(image.shape)

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

    limit = 150
    threshold = 0.001
    # setup the kernel for the convolutions
    # this kernel acts as a gaussian filter
    size = int(2*np.ceil(3*rho)+1)
    mid = (size-1)//2
    kernel = getGaussianKernel(size, rho)
    kernel = kernel @ kernel.T
    
    returnx = np.zeros(len(features))
    returny = np.zeros(len(features))
    
    for index, feature in enumerate(features):
        # I think that this should be y, x, but 
        # if this changes to y, x we get an error:
        # "shape of array too small to calculate numerical gradient"
        # something is messed up with the order of the coordinates
        x, y = feature
        # get area around the feature
        initial_image = i1[max(0, y-mid):min(y+mid, i1.shape[0]),
                           max(0, x-mid):min(x+mid, i1.shape[1])]
        next_image = i2[max(0, y-mid):min(y+mid, i2.shape[0]),
                        max(0, x-mid):min(x+mid, i2.shape[1])]
        # compute the gradient of the initial image
        gradient_y, gradient_x = np.gradient(initial_image)

        iterations = 0
        change = float('inf') # infinity
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
            # change = np.max(np.abs(delta_x)) + np.max(np.abs(delta_y))
            change = np.linalg.norm([delta_x, delta_y])
            # change = error[mid, mid]
            # update the number of iterations
            iterations += 1
        # update the displacement estimates
        returnx[index] = dx
        returny[index] = dy


    return np.array([returnx, returny])

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
    if energy.shape[0] == 0:
        return [0, 0]
    return [np.mean(energy[:,0]), np.mean(energy[:,1])]
