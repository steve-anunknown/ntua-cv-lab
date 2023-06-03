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
    Lx = np.zeros_like(video)
    Ly = np.zeros_like(video)
    Lt = np.zeros_like(video)
    for t_slice in range(video.shape[2]):
        Lx[:,:,t_slice] = cv2.filter2D(video[:,:,t_slice], -1, np.array([[-1, 0, 1]]))
        Ly[:,:,t_slice] = cv2.filter2D(video[:,:,t_slice], -1, np.array([[-1, 0, 1]]).T)
    for x in range(video.shape[0]):
        for y in range(video.shape[1]):
            Lt[x,y,:] = scp.convolve1d(video[x,y,:], np.array([-1, 0, 1]), mode='constant', cval=0.0)
    return Lx, Ly, Lt

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
    x_kernel = cv2.getGaussianKernel(space_size, sigma)
    y_kernel = cv2.getGaussianKernel(space_size, sigma)
    space_kernel = x_kernel @ y_kernel.T
    # the representation of arrays is by lines
    # therefore, a line is a 1d array but a column is a 2d array
    # even though in real life they are the same thing.

    # the gaussian kernel returns a column, that is, a 2d array.
    # if you transpose it you get a 2d array with one line.
    # if you get the first element, you get a 1d array.
    # the weird thing is that you can have a line as a 1d array
    # but not a column.

    # the indices of the line return elements.
    # the indices of the column return lines.

    time_kernel = cv2.getGaussianKernel(time_size, tau).T[0]
    # smoothen the video
    for t_slice in range(video.shape[2]):
        video[:,:,t_slice] = cv2.filter2D(video[:,:,t_slice], -1, space_kernel)
    for x in range(video.shape[0]):
        for y in range(video.shape[1]):
            video[x,y,:] = scp.convolve1d(video[x,y,:], time_kernel, mode='constant', cval=0.0)
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
    x_kernel = cv2.getGaussianKernel(space_size, sigma)
    y_kernel = cv2.getGaussianKernel(space_size, sigma)
    space_kernel = x_kernel @ y_kernel.T
    # smoothen the video
    for t_slice in range(video.shape[2]):
        video[:,:,t_slice] = cv2.filter2D(video[:,:,t_slice], -1, space_kernel)
    return video

def interest_points(response, rho, s):
    """
    Find interest points in a video.
    
    Keyword arguments:
    response -- response (y_len, x_len, frames)
    rho -- response threshold
    """
    maxr = np.max(response.flatten())
    x, y, t = np.where(response > rho * maxr)
    values = response[x, y, t]
    indices = np.argsort(values)[-500:]
    rx, ry, rt = x[indices], y[indices], t[indices]
    result = np.column_stack((rx, ry, rt, np.ones_like(rx)*s))
    return result

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
    Lx, Ly, Lt = video_gradients(video)
    Lxy = Lx * Ly
    Lxt = Lx * Lt
    Lyt = Ly * Lt
    Lxx = Lx * Lx
    Lyy = Ly * Ly
    Ltt = Lt * Lt

    # smoothen the elements
    space_size = int(2*np.ceil(3*s*sigma)+1)
    space_kernel = cv2.getGaussianKernel(space_size, s*sigma)

    time_size = int(2*np.ceil(3*s*tau)+1)
    time_kernel = cv2.getGaussianKernel(time_size, s*tau).T[0]
    for t_slice in range(video.shape[2]):
        Lxy[:,:,t_slice] = cv2.filter2D(Lxy[:,:,t_slice], -1, space_kernel)
        Lxt[:,:,t_slice] = cv2.filter2D(Lxt[:,:,t_slice], -1, space_kernel)
        Lyt[:,:,t_slice] = cv2.filter2D(Lyt[:,:,t_slice], -1, space_kernel)
        Lxx[:,:,t_slice] = cv2.filter2D(Lxx[:,:,t_slice], -1, space_kernel)
        Lyy[:,:,t_slice] = cv2.filter2D(Lyy[:,:,t_slice], -1, space_kernel)
        Ltt[:,:,t_slice] = cv2.filter2D(Ltt[:,:,t_slice], -1, space_kernel)
    for x in range(video.shape[0]):
        for y in range(video.shape[1]):
            Lxy[x,y,:] = scp.convolve1d(Lxy[x,y,:], time_kernel, mode='constant', cval=0.0)
            Lxt[x,y,:] = scp.convolve1d(Lxt[x,y,:], time_kernel, mode='constant', cval=0.0)
            Lyt[x,y,:] = scp.convolve1d(Lyt[x,y,:], time_kernel, mode='constant', cval=0.0)
            Lxx[x,y,:] = scp.convolve1d(Lxx[x,y,:], time_kernel, mode='constant', cval=0.0)
            Lyy[x,y,:] = scp.convolve1d(Lyy[x,y,:], time_kernel, mode='constant', cval=0.0)
            Ltt[x,y,:] = scp.convolve1d(Ltt[x,y,:], time_kernel, mode='constant', cval=0.0)
    # compute Harris response
    trace = Lxx + Lyy + Ltt
    det = (Lxx * Lyy * Ltt) + (2 * Lxy * Lyt * Lxt - Lxt * Lyy * Lxt) - (Lxy * Lxy * Ltt) - (Lxx * Lyt * Lyt)
    response = det - kappa * trace * trace * trace
    # find interest points
    points = interest_points(response, 0.05, s)
    return points

def GaborDetector(v, sigma, tau, kappa):
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
    # define the gabor filters
    # first define a linspace of width -2tau to 2tau
    time = np.linspace(-2*tau, 2*tau, int(4*tau+1))
    omega = 4/tau
    # define the gabor filters
    h_ev = np.exp(-time**2/(2*tau**2)) * np.cos(2*np.pi*omega*time)
    h_od = np.exp(-time**2/(2*tau**2)) * np.sin(2*np.pi*omega*time)
    # normalize the L1 norm
    h_ev = h_ev / np.sum(np.abs(h_ev))
    h_od = h_od / np.sum(np.abs(h_od))
    response = np.zeros_like(video)
    print(f"video[0,0,:] shape: {video[0,0,:].shape}")
    print(f"response[0,0,:] shape: {response[0,0,:].shape}")
    for x in range(video.shape[0]):
        for y in range(video.shape[1]):
            response[x,y,:] = (cv2.filter2D(video[x,y,:], -1, h_ev) ** 2).T[0] + (cv2.filter2D(video[x,y,:], -1, h_od) ** 2).T[0]
            
    # find interest points
    print(f"respone shape: {response.shape}")
    points = interest_points(response, 0.1, sigma)
    # keep the top 500 points
    return points


if __name__ == "__main__":
    num_frames = 200
    METHOD = "Harris"
    video = read_video("SpatioTemporal/running/person07_running_d3_uncomp.avi", num_frames, 0)
    if METHOD == "Harris":
        harris_points = HarrisDetector(video, 1.5, 2.5, 0.7, 0.005)
        print("Harris points: ", harris_points.shape)
        show_detection(video, harris_points, "Harris Detector")
    elif METHOD == "Gabor":
        gabor_points = GaborDetector(video, 4, 1.5, 0.005)
        print("Gabor points: ", gabor_points.shape)
        show_detection(video, gabor_points, "Gabor Detector")
        


    

