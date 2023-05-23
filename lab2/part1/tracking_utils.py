import numpy as np
from cv2 import getGaussianKernel, filter2D
from scipy.ndimage import map_coordinates
from matplotlib.pyplot import quiver

def lk(i1, i2, features, rho, epsilon, dx0, dy0):
    """ Lucas-Kanade algorithm
    
    Keyword arguments:
    i1 -- initial frame
    i2 -- next frame
    features -- coordinates of interest-points
    rho -- gaussian width parameter
    epsilon -- minimum constant to avoid zeroes
    dx0 -- initial condition x axis
    dy0 -- initial condition y axis
    """
    iterations = 0
    limit = 200
    change = 0
    threshold = 0.001

    # setup the kernel for the convolutions
    size = int(2*np.ceil(3*rho)+1)
    kernel = getGaussianKernel(size, rho)
    kernel = kernel @ kernel.T

    # compute the intensity
    x0, y0 = np.meshgrid(i1.shape[1], i1.shape[0])
    a = map_coordinates(i1, [np.ravel(y0 + dy0), np.ravel(x0 + dx0)], order=1)

    for feature in features:
        while (iterations < limit and change < threshold):
            iterations += 1

            # compute the gradient
            x, y = np.meshgrid(feature[0], feature[1])
            b = map_coordinates(i2, [np.ravel(y + dy0), np.ravel(x + dx0)], order=1)
            b = b - a

            # compute the jacobian
            jacobian = np.array([np.ravel(filter2D(i1, -1, kernel, borderType=1)),
                                 np.ravel(filter2D(i2, -1, kernel, borderType=1))]).T

            # compute the steepest descent
            steepest_descent = jacobian @ np.array([dx0, dy0])

            # compute the hessian
            hessian = jacobian.T @ jacobian

            # compute the inverse hessian
            inverse_hessian = np.linalg.inv(hessian)

            # compute the delta
            delta = inverse_hessian @ steepest_descent

            # update the parameters
            dx0 += delta[0]
            dy0 += delta[1]

            # compute the change
            change = np.linalg.norm(delta)
    
    return np.array([dx0, dy0])

def displ(dx, dy, method):
    """ Display the optical flow
    
    Keyword arguments:
    dx -- displacement x axis
    dy -- displacement y axis
    method -- method to display the optical flow
    """

    # It is known that the optical flow vectors
    # tend to have greatest norm at points that
    # belong to the areas of high textural information
    # and low norm at points that belong to the areas
    # of low textural information and uniform texture.
    # Therefore, since the majority of interest points
    # are located in areas of high textural information,
    # we could simply use the mean value of the displacement
    # vectors.

    # In order to achieve a better visualization of the
    # optical flow or to reject outliers, we could implement
    # different techniques such as:
    # - Computing the mean value of the displacement vectors
    #   that have greater energy than a certain threshold.

    if method == "energy":
        # compute the energy of the optical flow
        energy = np.sqrt(dx**2 + dy**2)

        # compute the mean value of the energy
        mean = np.mean(energy)

        # compute the threshold
        threshold = 0.5*mean

        # compute the indices of the optical flow
        indices = np.where(energy > threshold)

        # compute the optical flow
        dx = dx[indices]
        dy = dy[indices]
    elif method == "texture":
        pass
    else:
        print("Method has to be either 'energy' or 'texture'.")
    
    return np.array([dx, dy])
