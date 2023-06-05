from detector_utils import HarrisDetector
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import show_detection


if __name__ == "__main__":
    num_frames = 200
    video = read_video("SpatioTemporal/running/person07_running_d3_uncomp.avi", num_frames, 0)
    harris_points = HarrisDetector(video, 2, 4, 1.5, 0.005)
    print("Harris points: ", harris_points.shape)
    show_detection(video, harris_points, "Harris Detector")