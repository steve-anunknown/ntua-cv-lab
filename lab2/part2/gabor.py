from detector_utils import GaborDetector
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import show_detection

if __name__ == "__main__":
    num_frames = 200
    video = read_video("SpatioTemporal/running/person07_running_d3_uncomp.avi", num_frames, 0)
    print(f"Video shape: {video.shape}")
    gabor_points = GaborDetector(video, 4, 1.5)
    print("Gabor points: ", gabor_points.shape)
    show_detection(video, gabor_points, "Gabor Detector")
