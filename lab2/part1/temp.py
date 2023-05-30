


def displ(dx, dy, method):
    """ Display the optical flow
    
    Keyword arguments:
    dx -- displacement x axis
    dy -- displacement y axis
    method -- method to display the optical flow

    returns -- optical flow for visualization
    """

    # In order to achieve a better visualization of the
    # optical flow or to reject outliers, we could implement
    # different techniques such as:
    # - Computing the mean value of the displacement vectors
    #   that have greater energy than a certain threshold.

    if method == "energy":
        # compute the energy of the optical flow
        energies = np.array([np.sqrt(x**2 + y**2) for x, y in zip(dx, dy)])

        # compute the mean value of the energy
        mean_energy = np.mean(energies)

        # compute the threshold
        threshold = 0.8*mean_energy

        # compute the indices of the optical flow
        indices = np.where(energies <= threshold)
        disp_x, disp_y = np.zeros(dx.shape), np.zeros(dy.shape)
        disp_x[indices], disp_y[indices] = dx[indices], dy[indices]
        return np.array([np.mean(disp_x), np.mean(disp_y)])
    elif method == "texture":
        # It is known that the optical flow vectors
        # tend to have greatest norm at points that
        # belong to the areas of high textural information
        # and low norm at points that belong to the areas
        # of low textural information and uniform texture.
        # Therefore, since the majority of interest points
        # are located in areas of high textural information,
        # we could simply use the mean value of the displacement
        # vectors.
        return np.array([np.mean(dx), np.mean(dy)])
    else:
        print("Method has to be either 'energy' or 'texture'.")
        exit(5)


def lk(i1, i2, features, rho, epsilon, dx0, dy0):
    """
    Lucas-Kanade algorithm
    
    Arguments:
    i1 -- initial frame
    i2 -- next frame
    features -- coordinates of interest-points
    rho -- Gaussian width parameter
    epsilon -- minimum constant to avoid zeroes
    dx0 -- initial condition x-axis
    dy0 -- initial condition y-axis
    
    Returns:
    dx -- computed displacement along x-axis
    dy -- computed displacement along y-axis
    """
    
    # Calculate gradient of the initial frame
    gradient_x = np.gradient(i1, axis=1)
    gradient_y = np.gradient(i1, axis=0)
    
    dx = dx0
    dy = dy0
    
    # Iterative refinement
    while True:
        # Warp the second frame using the current displacement estimates
        warped_i2 = np.zeros_like(i2)
        for idx, (x, y) in enumerate(features):
            x = int(x)
            y = int(y)
            warped_i2[y, x] = i2[y + int(dy[idx]), x + int(dx[idx])]
        
        # Calculate the error between the initial and warped frames
        error = i1 - warped_i2
        
        # Compute the Lucas-Kanade equations
        A = np.zeros((len(features), 2, 2))
        b = np.zeros((len(features), 2))
        
        for idx, (x, y) in enumerate(features):
            x = int(x)
            y = int(y)
            
            A[idx, 0, 0] = np.sum(gradient_x[y, x]**2)
            A[idx, 0, 1] = np.sum(gradient_x[y, x] * gradient_y[y, x])
            A[idx, 1, 0] = np.sum(gradient_x[y, x] * gradient_y[y, x])
            A[idx, 1, 1] = np.sum(gradient_y[y, x]**2)
            
            b[idx, 0] = np.sum(gradient_x[y, x] * error[y, x])
            b[idx, 1] = np.sum(gradient_y[y, x] * error[y, x])
        
        # Solve the linear equations using least squares
        try:
            deltas = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        
        delta_x = deltas[:, 0]
        delta_y = deltas[:, 1]
        
        # Update the displacement estimates
        dx += delta_x
        dy += delta_y
        
        # Check convergence
        if np.max(np.abs(delta_x)) < epsilon and np.max(np.abs(delta_y)) < epsilon:
            break
    
    return dx, dy

#system = np.array([[filter2D(gradx**2, -1, kernel, borderType=1)[x,y] + epsilon,
#                    filter2D(gradx*grady, -1, kernel, borderType=1)[x,y]],
#                    [filter2D(gradx*grady, -1, kernel, borderType=1)[x,y],
#                    filter2D(grady**2, -1, kernel, borderType=1)[x,y] + epsilon]])
#steps = np.array([filter2D(gradx*error, -1, kernel, borderType=1)[x,y],
#                    filter2D(grady*error, -1, kernel, borderType=1)[x,y]])

#
#    returnx, returny = np.zeros(len(features)), np.zeros(len(features))
#
#    grady, gradx = np.gradient(i1)
#    s11 = filter2D(gradx**2, -1, kernel, borderType=1) + epsilon
#    s12 = filter2D(gradx*grady, -1, kernel, borderType=1)
#    s22 = filter2D(grady**2, -1, kernel, borderType=1) + epsilon
#
#    for index, feature in enumerate(features):
#        # define a small area around the feature
#        x, y = int(feature[0]), int(feature[1])
#        original = i1[x-2:x+2, y-2:y+2]
#        following = i2[x-2:x+2, y-2:y+2]
#        # use the initial guess for the optical flow
#        # of the current feature
#        dx, dy = dx0[index], dy0[index]
#
#
#        while (iterations < limit and change > threshold):
#            # compute the shifted image of the first frame
#            initial = shift_image(original, [dx, dy])
#            print(f"dx: {dx}, dy: {dy}")
#            print(f"initial shape: {initial.shape}")
#            # compute the error between the second frame
#            # and the shifted first frame
#            error = following - initial
#
#            # compute the shifted image of the gradient
#            a1 = shift_image(gradx[x-2:x+2, y-2:y+2], [dx, dy])
#            a2 = shift_image(grady[x-2:x+2, y-2:y+2], [dx, dy])
#            print(f"gradx shape: {gradx.shape}")
#            print(f"grady shape: {grady.shape}")
#
#            b1 = filter2D(a1*error, -1, kernel, borderType=1)
#            b2 = filter2D(a2*error, -1, kernel, borderType=1)
#
#            # setup the matrices to calculate the improvement
#            system = np.array([[s11[x,y], s12[x,y]],
#                                 [s12[x,y], s22[x,y]]])
#            steps = np.array([b1[x,y], b2[x,y]])
#            # calculate the improvement
#            det = system[0, 0]*system[1, 1] - system[0, 1]*system[1, 0]
#            imp_x = (system[1, 1]*steps[0] - system[0, 1]*steps[1])/det
#            imp_y = (system[0, 0]*steps[1] - system[1, 0]*steps[0])/det
#            improvement = np.array([imp_x, imp_y])
#            # update the parameters
#            dx, dy = dx + improvement[0], dy + improvement[1]
#            # calculate the change
#            change = np.linalg.norm(improvement)
#            # update the iterations
#            iterations += 1
#        returnx[index], returny[index] = dx, dy
#