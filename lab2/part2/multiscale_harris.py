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
    harris_points = MultiscaleDetector(lambda video, sigma, tau: HarrisDetector(video, 2, sigma, tau, 0.005),
                                       video, [3*(1.1**i) for i in range(6)], 1.5)
    print("Multiscale Harris points: ", harris_points.shape)
    show_detection(video, harris_points, "Harris Detector")