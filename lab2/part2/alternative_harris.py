import cv2
import numpy as np

def HarrisDetector(v, s, sigma, tau, kappa):
    """
    Harris Corner Detector
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames)
    s -- Gaussian kernel size
    sigma -- Gaussian kernel space standard deviation
    tau -- Gaussian kernel time standard deviation
    rho -- Harris response threshold
    """
    # define Gaussian kernel
    space_size = int(2*np.ceil(3*sigma)+1)
    time_size = int(2*np.ceil(3*tau)+1)
    space_kernel = cv2.getGaussianKernel(space_size, sigma).T[0]
    time_kernel = cv2.getGaussianKernel(time_size, tau).T[0]
    space_kernel = space_kernel @ space_kernel.T
    time_kernel = time_kernel @ time_kernel.T
    # setup video
    video = v.copy()
    video = video.astype(float)/video.max()
    Ly, Lx, Lt = np.gradient(video)
    # smoothen the gradient products
    Lxx = cv2.filter2D(cv2.filter2D(Lx * Lx, -1, space_kernel), -1, time_kernel)
    Lyy = cv2.filter2D(cv2.filter2D(Ly * Ly, -1, space_kernel), -1, time_kernel)
    Ltt = cv2.filter2D(cv2.filter2D(Lt * Lt, -1, space_kernel), -1, time_kernel)
    Lxy = cv2.filter2D(cv2.filter2D(Lx * Ly, -1, space_kernel), -1, time_kernel)
    Lxt = cv2.filter2D(cv2.filter2D(Lx * Lt, -1, space_kernel), -1, time_kernel)
    Lyt = cv2.filter2D(cv2.filter2D(Ly * Lt, -1, space_kernel), -1, time_kernel)
    # compute Harris response
    trace = Lxx + Lyy + Ltt
    det = Lxx*(Lyy*Ltt - Lyt*Lyt) - Lxy*(Lxy*Ltt - Lyt*Lxt) + Lxt*(Lxy*Lyt - Lyy*Lxt)
    response = (det - kappa * trace * trace * trace)
    # find interest points
    points = interest_points(response, 550, s)
    return points
def interest_points(response, num, s):
    """
    Find interest points in a video.
    
    Keyword arguments:
    response -- response (y_len, x_len, frames)
    num -- number of interest points
    s -- scale
    """
    descending = np.flip(np.argsort(response.flatten()))
    points = []
    print("response shape: ", response.shape)
    print("response: ", response)
    for i in range(min(num, len(descending))):
        y, x, t = np.unravel_index(i, response.shape)
        points.append((x, y, t, s))
    points = np.array(points)
    return points


def MultiscaleHarrisDetector(v, s, sigma, tau, kappa, scale, N):
    """
    Multiscale Harris Corner Detector
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames)
    sigma -- Gaussian kernel space standard deviation
    tau -- Gaussian kernel time standard deviation
    kappa -- Harris response threshold
    scale -- Gaussian kernel space standard deviation for scale
    N -- number of scales
    """
    video = v.copy()
    video = video.astype(float)/video.max()
    # define the scales
    scales = [scale**i for i in range(N)]
    sigmas = [sigma*s for s in scales]

    gradsxx = []
    gradsyy = []
    gradstt = []
    points_per_scale = []
    time_size = int(2*np.ceil(3*tau)+1)
    time_kernel = cv2.getGaussianKernel(time_size, tau).T[0]
    time_size = int(2*np.ceil(3*s*tau)+1)
    time_kernel_2 = cv2.getGaussianKernel(time_size, s*tau).T[0]
    for sigm in sigmas:
        # define Gaussian kernel
        space_size = int(2*np.ceil(3*sigm)+1)
        space_kernel = cv2.getGaussianKernel(space_size, sigma).T[0]
        video = video_smoothen(video, space_kernel, time_kernel)

        Ly, Lx, Lt = video_gradients(video)
        # define Gaussian kernel
        space_size = int(2*np.ceil(3*s*sigm)+1)
        space_kernel = cv2.getGaussianKernel(space_size, sigma).T[0]
        Lxx = video_smoothen(Lx * Lx, space_kernel, time_kernel_2)
        Lyy = video_smoothen(Ly * Ly, space_kernel, time_kernel_2)
        Ltt = video_smoothen(Lt * Lt, space_kernel, time_kernel_2)
        Lxy = video_smoothen(Lx * Ly, space_kernel, time_kernel_2)
        Lxt = video_smoothen(Lx * Lt, space_kernel, time_kernel_2)
        Lyt = video_smoothen(Ly * Lt, space_kernel, time_kernel_2)
        trace = Lxx + Lyy + Ltt
        det = Lxx*(Lyy*Ltt - Lyt*Lyt) - Lxy*(Lxy*Ltt - Lyt*Lxt) + Lxt*(Lxy*Lyt - Lyy*Lxt)
        response = (det - kappa * trace * trace * trace)
        points = interest_points(response, 0.01, sigm)
        
        gradsxx.append(Lxx)
        gradsyy.append(Lyy)
        gradstt.append(Ltt)
        points_per_scale.append(points)

    grads = list(zip(scales, gradsxx, gradsyy, gradstt))
    logs = [(s**2)*np.abs(Lxx+Lyy) for s, Lxx, Lyy, Ltt in grads]
    return LogMetric(logs, points_per_scale, N)