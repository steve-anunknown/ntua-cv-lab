import cv2
import numba
import numpy as np
import scipy.ndimage as scp
from functools import lru_cache
from matplotlib import pyplot as plt
from cv23_lab2_2_utils import orientation_histogram

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

def interest_points(response, num, threshold, scale):
    """
    Find interest points in a video.
    
    Keyword arguments:
    response -- response (y_len, x_len, frames)
    num -- number of interest points
    threshold -- threshold
    scale -- scale
    """
    maxr = np.max(response.flatten())
    x, y, t = np.where(response > threshold*maxr)
    points = np.column_stack((y, x, t, scale*np.ones(len(x))))
    
    # Sort points based on response values in descending order
    response_values = response[x, y, t]
    sorted_indices = np.argsort(response_values)[::-1]  # Sort in descending order
    
    # Select the top 'num' points
    # top_points = points[sorted_indices[:num]]
    top_points = points[sorted_indices[:(len(points)//num)*num:len(points)//num]]
    return top_points

def HarrisDetector(v, s, sigma, tau, kappa, threshold, num_points):
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
    video = v.copy() # copy so we don't modify the original
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
    response = abs(det - kappa * trace * trace * trace)
    # find interest points
    points = interest_points(response, num=num_points, threshold=threshold, scale=sigma)
    return points

def GaborDetector(v, sigma, tau, threshold, num_points):
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
    points = interest_points(response, num=num_points, threshold=threshold, scale=sigma)
    return points


@lru_cache(maxsize=None)
def time_function(tau):
    time = np.linspace(-2*tau, 2*tau, int(4*tau+1))
    omega = 4/tau
    # define the gabor filters
    h_ev = np.exp(-time**2/(2*tau**2)) * np.cos(2*np.pi*omega*time)
    h_od = np.exp(-time**2/(2*tau**2)) * np.sin(2*np.pi*omega*time)
    # normalize the L1 norm
    h_ev /= np.linalg.norm(h_ev, ord=1)
    h_od /= np.linalg.norm(h_od, ord=1)
    return (h_ev, h_od)

def GaborDetectorTrial(v, sigma, tau, threshold, num_points):
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
    # compute the response
    h_ev, h_od = time_function(tau)
    response = (scp.convolve1d(video, h_ev, axis=2) ** 2) + (scp.convolve1d(video, h_od, axis=2) ** 2)
    points = interest_points(response, num=num_points, threshold=threshold, scale=sigma)
    return points

def GaborDetectorTrial2(v, sigma, h_ev, h_od, threshold, num_points):
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
    # compute the response
    response = (scp.convolve1d(video, h_ev, axis=2) ** 2) + (scp.convolve1d(video, h_od, axis=2) ** 2)
    points = interest_points(response, num=num_points, threshold=threshold, scale=sigma)
    
    return points

def MultiscaleDetector(detector, video, sigmas, tau, num_points):
    """
    Multiscale Detector

    Executes a detector at multiple scales. Detector has to be a function that
    takes a video as input, along with other parameters, and returns a list of interest points.

    
    Keyword arguments:
    detector -- function that returns interest points
    video -- input video (y_len, x_len, frames)
    sigmas -- list of scales
    """
    # FIXME: probably needs refactoring.
    # the code is clear but super inefficient.
    # the gradients are computed a gazillion times.
    
    # for every scale, compute the response
    points = []
    for sigm in sigmas:
        found = detector(video, sigm, tau)
        points.append(found)
    return LogMetricFilter(video, points, tau, num_points)

def LogMetricFilter(video, points_per_scale, tau, num_points):
    """
    Filters interest points according to the log metric
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames)
    points_per_scale -- list of interest points
    """
    def LogMetric(logs, itemsperscale, N):
        # log((x,y), s) = (s^2)|Lxx((x,y),s) + Lyy((x,y),s)|
        # returns the coordinates of the points that maximize
        # the log metric in a neighborhood of 3 scales
        # (prev scale), (curr scale), (next scale)
        final = []
        final_logs = []
        for index, items in enumerate(itemsperscale):
            logp = logs[max(index-1,0)]
            logc = logs[index]
            logn = logs[min(index+1,N-1)]
            for triplet in items:
                y, x, t = int(triplet[0]), int(triplet[1]), int(triplet[2])
                prev = logp[x, y, t]
                curr = logc[x, y, t]
                next = logn[x, y, t]
                if (curr >= prev) and (curr >= next):
                    final.append(triplet)
                    final_logs.append(curr)
        # get the points with top num_points log metric values
        if len(final) > num_points:
            indices = np.argsort(final_logs)[::-1]
            final_points = [final[i] for i in indices[:num_points]]
            return np.array(final_points)
        else:
            return np.array(final)
    v = video.copy()
    vnorm = v.astype(float)/video.max()
    # compute the laplacian of gaussian (log) metric
    logs = []
    time_size = int(2*np.ceil(3*tau)+1)
    time_kernel = cv2.getGaussianKernel(time_size, tau).T[0]
    # get the sigmas from the points
    sigmas = [item[0, 3] for item in points_per_scale]
    for sigma in sigmas:
        # define Gaussian kernel
        space_size = int(2*np.ceil(3*sigma)+1)
        space_kernel = cv2.getGaussianKernel(space_size, sigma).T[0]
        v = video_smoothen(vnorm, space_kernel, time_kernel)
        # compute gradients
        Ly, Lx, _ = video_gradients(v)
        # compute second order derivatives
        Lyy, _, _ = video_gradients(Ly)
        _, Lxx, _ = video_gradients(Lx)
        # compute the log metric
        log = (sigma**2) * np.abs(Lxx + Lyy)
        logs.append(log)
    # find the points that maximize the log metric
    return LogMetric(logs, points_per_scale, len(points_per_scale))

def get_hog_descriptors(video, interest_points, nbins):
    """
    Compute the HOG descriptors of a video.
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames)
    interest_points -- interest points (y, x, t, s)
    sigma -- Gaussian kernel space standard deviation
    nbins -- number of bins
    """
    # gradients
    Ly, Lx, _ = video_gradients(video.astype(float))
    descriptors = []
    for point in interest_points:
        side      = int(round(4*point[3]))
        leftmost  = int(max(0,                point[0]-side))
        rightmost = int(min(video.shape[1]-1, point[0]+side+1))
        upmost    = int(max(0,                point[1]-side))
        downmost  = int(min(video.shape[0]-1, point[1]+side+1))
        
        descriptor = orientation_histogram(Lx[upmost:downmost, leftmost:rightmost, int(point[2])],
                                           Ly[upmost:downmost, leftmost:rightmost, int(point[2])],
                                           nbins, np.array([side, side]))
        descriptors.append(descriptor)
    return np.array(descriptors, dtype=object)

def get_hof_descriptors(video, interest_points, nbins):
    """
    Compute the HOF descriptors of a video.
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames)
    interest_points -- interest points (y, x, t, s)
    sigma -- Gaussian kernel space standard deviation
    nbins -- number of bins
    """
    oflow = cv2.DualTVL1OpticalFlow_create(nscales=1)
    descriptors = []
    for point in interest_points:
        side      = int(round(4*point[3]))
        leftmost  = int(max(0,                point[0]-side))
        rightmost = int(min(video.shape[1]-1, point[0]+side+1))
        upmost    = int(max(0,                point[1]-side))
        downmost  = int(min(video.shape[0]-1, point[1]+side+1))
        
        flow = oflow.calc(video[upmost:downmost, leftmost:rightmost, int(point[2]-1)],
                          video[upmost:downmost, leftmost:rightmost, int(point[2])], None)
        descriptor = orientation_histogram(flow[...,0], flow[...,1],
                                           nbins, np.array([side, side]))
        descriptors.append(descriptor)
    return np.array(descriptors, dtype=object)

def get_hog_hof(video, interest_points, nbins):
    """
    Compute the HOG and HOF descriptors of a video.
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames)
    interest_points -- interest points (y, x, t, s)
    sigma -- Gaussian kernel space standard deviation
    nbins -- number of bins
    """
    hog = get_hog_descriptors(video, interest_points, nbins)
    print("HOG: ", hog.shape)
    hof = get_hof_descriptors(video, interest_points, nbins)
    print("HOF: ", hof.shape)
    return np.concatenate((hog, hof))
        
    

