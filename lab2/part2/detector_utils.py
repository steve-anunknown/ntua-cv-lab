import cv2
import numpy as np
import scipy.ndimage as scp
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import show_detection


def video_gradients(video):
    """
    Compute the gradients of a video.
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames)
    """
    # normalize video
    video = video.astype(np.float32)/255
    # compute gradients
    Ly = scp.convolve1d(video, np.array([-1, 0, 1]), axis=0)
    Lx = scp.convolve1d(video, np.array([-1, 0, 1]), axis=1)
    Lt = scp.convolve1d(video, np.array([-1, 0, 1]), axis=2)
    return Ly, Lx, Lt

def video_smoothen(video, sigma, tau):
    """
    Smoothen a video.
    
    Keyword arguments:
    video -- input video (y_len, x_len, frames) normalized to [0, 1]
    sigma -- Gaussian kernel space standard deviation
    tau -- Gaussian kernel time standard deviation
    """
    # define Gaussian kernel
    space_size = int(2*np.ceil(3*sigma)+1)
    time_size = int(2*np.ceil(3*tau)+1)
    space_kernel = cv2.getGaussianKernel(space_size, sigma).T[0]
    time_kernel = cv2.getGaussianKernel(time_size, tau).T[0]
    # smoothen the video
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
    for i in range(min(num, len(descending))):
        y, x, t = np.unravel_index(i, response.shape)
        points.append((x, y, t, s))
    points = np.array(points)
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
    video = v.copy()
    video = video.astype(np.float32)/255
    video = video_smoothen(video, sigma, tau)
    # Lx, Ly, Lt = video_gradients(video)
    Ly, Lx, Lt = np.gradient(video)
    Lxy = video_smoothen(Lx * Ly, s*sigma, s*tau)
    Lxt = video_smoothen(Lx * Lt, s*sigma, s*tau)
    Lyt = video_smoothen(Ly * Lt, s*sigma, s*tau)
    Lxx = video_smoothen(Lx * Lx, s*sigma, s*tau)
    Lyy = video_smoothen(Ly * Ly, s*sigma, s*tau)
    Ltt = video_smoothen(Lt * Lt, s*sigma, s*tau)

    # compute Harris response
    trace = Lxx + Lyy + Ltt
    det = (Lxx * Lyy * Ltt) + (2 * Lxy * Lyt * Lxt) - (Lxx * Lyt * Lyt) - (Lxt * Lyy * Lxt) - (Lxy * Lxy * Ltt) 
    response = np.abs(det - kappa * trace * trace * trace)
    # find interest points
    points = interest_points(response, 550, s)
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
    video = v.copy()
    video = video.astype(np.float32)/255
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
    response = (scp.convolve1d(video, h_ev, axis=2) ** 2) + (scp.convolve1d(video, h_od, axis=2) ** 2)
    points = interest_points(response, 550, sigma)
    return points
        


    

