import cv2
import numpy as np
import scipy.ndimage as scp

def video_gradients(video):
    """
    Compute the gradients of a video.
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames) normalized to [0, 1]
    """
    # compute gradients
    Ly = scp.convolve1d(video, np.array([-1, 0, 1]), axis=0)
    Lx = scp.convolve1d(video, np.array([-1, 0, 1]), axis=1)
    Lt = scp.convolve1d(video, np.array([-1, 0, 1]), axis=2)
    return Ly, Lx, Lt

def video_smoothen(video, space_kernel, time_kernel):
    """
    Smoothen a video.
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames) normalized to [0, 1]
    space_kernel -- Gaussian kernel space standard deviation
    time_kernel -- Gaussian kernel time standard deviation
    """
    video = scp.convolve1d(video, space_kernel, axis=0)
    video = scp.convolve1d(video, space_kernel, axis=1)
    video = scp.convolve1d(video, time_kernel, axis=2)
    return video

def video_smoothen_space(video, sigma):
    """
    Smoothen a video in space.
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames) normalized to [0, 1]
    sigma -- Gaussian kernel space standard deviation
    """
    # define Gaussian kernel
    space_size = int(2*np.ceil(3*sigma)+1)
    kernel = cv2.getGaussianKernel(space_size, sigma).T[0]
    # smoothen the video
    video = scp.convolve1d(video, kernel, axis=0)
    video = scp.convolve1d(video, kernel, axis=1)
    return video

def interest_points(response, rho, s):
    """
    Find interest points in a video.
    
    Keyword arguments:
    response -- response (y_len, x_len, frames)
    num -- number of interest points
    s -- scale
    """
    def disk_strel(n):
        '''
            Return a structural element, which is a disk of radius n.
        '''
        r = int(np.round(n))
        d = 2*r+1
        x = np.arange(d) - r
        y = np.arange(d) - r
        x, y = np.meshgrid(x,y)
        strel = x**2 + y**2 <= r**2
        return strel.astype(np.uint8)
    ns = int(2*np.ceil(3*s)+1)
    strel = disk_strel(ns)
    cond1 = ( response == cv2.dilate(response, strel) )
    maxr = np.max(response.flatten())
    cond2 = ( response > rho*maxr )
    x, y, t = np.where(cond1 & cond2)
    points = np.column_stack((y, x, t, s*np.ones(len(x))))
    return points

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
    # setup video
    video = v.copy()
    video = video.astype(float)/video.max()
    video = video_smoothen(video, space_kernel, time_kernel)
    # compute gradients
    Ly, Lx, Lt = video_gradients(video)
    # define Gaussian kernel
    space_size = int(2*np.ceil(3*s*sigma)+1)
    time_size = int(2*np.ceil(3*s*tau)+1)
    space_kernel = cv2.getGaussianKernel(space_size, s*sigma).T[0]
    time_kernel = cv2.getGaussianKernel(time_size, s*tau).T[0]
    # smoothen the gradient products
    Lxy = video_smoothen(Lx * Ly, space_kernel, time_kernel)
    Lxt = video_smoothen(Lx * Lt, space_kernel, time_kernel)
    Lyt = video_smoothen(Ly * Lt, space_kernel, time_kernel)
    Lxx = video_smoothen(Lx * Lx, space_kernel, time_kernel)
    Lyy = video_smoothen(Ly * Ly, space_kernel, time_kernel)
    Ltt = video_smoothen(Lt * Lt, space_kernel, time_kernel)
    # compute Harris response
    trace = Lxx + Lyy + Ltt
    det = Lxx*(Lyy*Ltt - Lyt*Lyt) - Lxy*(Lxy*Ltt - Lyt*Lxt) + Lxt*(Lxy*Lyt - Lyy*Lxt)
    response = (det - kappa * trace * trace * trace)
    # find interest points
    points = interest_points(response, 0.01, sigma)
    return points

def GaborDetector(v, sigma, tau):
    """
    Gabor Detector
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames)
    sigma -- Gaussian kernel space standard deviation
    tau -- Gaussian kernel time standard deviation
    kappa -- Gabor response threshold
    """
    # setup video
    video = v.copy()
    video = video.astype(float)/video.max()
    video = video_smoothen_space(video, sigma)
    # first define a linspace of width -2tau to 2tau
    time = np.linspace(-2*tau, 2*tau, int(4*tau+1))
    omega = 4/tau
    # define the gabor filters
    h_ev = np.exp(-time**2/(2*tau**2)) * np.cos(2*np.pi*omega*time)
    h_od = np.exp(-time**2/(2*tau**2)) * np.sin(2*np.pi*omega*time)
    # normalize the L1 norm
    h_ev /= np.linalg.norm(h_ev, ord=1)
    h_od /= np.linalg.norm(h_od, ord=1)
    # compute the response
    response = (scp.convolve1d(video, h_ev, axis=2) ** 2) + (scp.convolve1d(video, h_od, axis=2) ** 2)
    points = interest_points(response, 0.2, sigma)
    return points

def LogMetric(logs, itemsperscale, N):
    # log((x,y), s) = (s^2)|Lxx((x,y),s) + Lyy((x,y),s)|
    # returns the coordinates of the points that maximize
    # the log metric in a neighborhood of 3 scales
    # (prev scale), (curr scale), (next scale)
    final = []
    for index, items in enumerate(itemsperscale):
        logp = logs[max(index-1,0)]
        logc = logs[index]
        logn = logs[min(index+1,N-1)]
        for triplet in items:
            x = int(triplet[1])
            y = int(triplet[0])
            prev = logp[x,y]
            curr = logc[x,y]
            next = logn[x,y]
            if (curr >= prev) and (curr >= next):
                final.append(triplet)
    return np.array(final)

def MultiscaleHarrisDetector(v, s, sigma, tau, kappa, scale, N):
    """
    Multiscale Harris Corner Detector

    Executes the Harris Corner Detector at multiple scales.
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames)
    sigma -- Gaussian kernel space standard deviation
    tau -- Gaussian kernel time standard deviation
    kappa -- Harris response threshold
    scale -- Gaussian kernel space standard deviation for scale
    N -- number of scales
    """
    # define scales
    scales = (s**i for i in range(N))
    sigmas = (sigma*scale for scale in scales)
    # for every scale, compute the Harris response
    points = []
    for sigm in sigmas:
        points.append(HarrisDetector(v, s, sigm, tau, kappa))
    # compute the laplacian of gaussian (log) metric
    # log((x,y), s) = (s^2)|Lxx((x,y),s) + Lyy((x,y),s)|
    # smoothen the video for every scale
    video = v.copy()
    video = video.astype(float)/video.max()
    logs = []
    for sigm in sigmas:
        video = video_smoothen(video, sigm, tau)
        # compute gradients
        Ly, Lx, _ = video_gradients(video)
        # compute second order derivatives
        Lyy, _, _ = video_gradients(Ly)
        _, Lxx, _ = video_gradients(Lx)
        # compute the log metric
        log = (sigm**2) * np.abs(Lxx + Lyy)
        logs.append(log)
    # find the points that maximize the log metric
    final_points = LogMetric(logs, points, N)
    return final_points

        


