import sys
from detector_utils import GaborDetector
from detector_utils import MultiscaleDetector
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import show_detection

if __name__ == "__main__":
    num_frames = 200
    # get video name from the command line
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
    video = read_video(video_name, num_frames, 0)
    gabor_points = MultiscaleDetector(lambda video, sigma, tau: GaborDetector(video, sigma, tau, threshold=0.3),
                                        video, sigmas=[3*(1.1**i) for i in range(6)], tau=1.5)
    print("Multiscale Gabor points: ", gabor_points.shape)
    show_detection(video, gabor_points, "Gabor Detector")
