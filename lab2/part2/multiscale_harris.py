import sys
from detector_utils import HarrisDetector
from detector_utils import MultiscaleDetector
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import show_detection


if __name__ == "__main__":
    num_frames = 200
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
    video = read_video(video_name, num_frames, 0)
    harris_points = MultiscaleDetector(lambda video, sigma, tau: 
                                       HarrisDetector(video, 2, sigma, tau, kappa=0.005, threshold=0.05, num_points=500),
                                       video, [3*(1.1**i) for i in range(8)], tau=1.5, num_points=500)
    print("Multiscale Harris points: ", harris_points.shape)
    show_detection(video, harris_points, "Harris Detector")