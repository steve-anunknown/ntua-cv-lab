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
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    x = x + shift[0]
    y = y + shift[1]
    coordinates = np.array([y.ravel(), x.ravel()])
    shifted_image = map_coordinates(image, coordinates, order=1)
    return shifted_image.reshape(image.shape)

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
    change = ~0 # infinity
    threshold = 0.001

    # setup the kernel for the convolutions
    # this kernel acts as a gaussian filter
    size = int(2*np.ceil(3*rho)+1)
    kernel = getGaussianKernel(size, rho)
    kernel = kernel @ kernel.T
    dx = dx0
    dy = dy0
    for feature in features:
        # define a small area around the feature
        x, y = feature
        x = int(x)
        y = int(y)
        cropped1 = i1[x-5:x+5, y-5:y+5]
        cropped2 = i2[x-5:x+5, y-5:y+5]
        # iterate until convergence
        while (iterations < limit and change > threshold):
            # compute the shifted image of the first frame
            initial = shift_image(cropped1, [dx, dy])
            # compute the gradient of the shifted image
            grady, gradx = np.gradient(initial)
            # compute the error between the second frame
            # and the shifted first frame
            error = cropped2 - initial
            # setup the matrices to calculate the improvement
            system = np.array([[filter2D(gradx**2, -1, kernel, borderType=1) + epsilon, filter2D(gradx*grady, -1, kernel, borderType=1)],
                               [filter2D(gradx*grady, -1, kernel, borderType=1), filter2D(grady**2, -1, kernel, borderType=1) + epsilon]])
            steps = np.array([filter2D(gradx*error, -1, kernel, borderType=1), filter2D(grady*error, -1, kernel, borderType=1)]).T
            # calculate the improvement
            improvement = np.linalg.inverse(system) @ steps
            # update the parameters
            dx += improvement[0]
            dy += improvement[1]
            # calculate the change
            change = np.linalg.norm(improvement)
            # update the iterations
            iterations += 1
    
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
