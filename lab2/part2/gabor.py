import sys
from detector_utils import GaborDetector
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import show_detection

if __name__ == "__main__":
    num_frames = 200
    # get video name from the command line
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
    video = read_video(video_name, num_frames, 0)
    gabor_points = GaborDetector(video, sigma=4, tau=1.5, threshold=0.25)
    print("Gabor points: ", gabor_points.shape)
    show_detection(video, gabor_points, "Gabor Detector")
