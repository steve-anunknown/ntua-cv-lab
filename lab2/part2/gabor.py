import sys
from detector_utils import GaborDetector
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import show_detection

if __name__ == "__main__":
    num_frames = 100
    # get video name from the command line
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
    video = read_video(video_name, num_frames, 0)
    gabor_points = GaborDetector(video, sigma=3, tau=1.5,
                                 threshold=0.25, num_points=500)
    print("Gabor points: ", gabor_points.shape)
    print("Gabor points: ", gabor_points)
    show_detection(video, gabor_points, "Gabor Detector")
