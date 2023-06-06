import sys
from detector_utils import HarrisDetector
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import show_detection


if __name__ == "__main__":
    num_frames = 200
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
    video = read_video(video_name, num_frames, 0)
    harris_points = HarrisDetector(video, 2, 4, 1.5, 0.005)
    print("Harris points: ", harris_points.shape)
    show_detection(video, harris_points, "Harris Detector")