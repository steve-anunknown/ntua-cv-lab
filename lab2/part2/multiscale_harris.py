import os
import sys
import random
from detector_utils import HarrisDetector
from detector_utils import MultiscaleDetector
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import show_detection

RESULTS_FOLDER = "results"
DATA_FOLDER = "SpatioTemporal"
CATEGORIES = ["walking", "running", "handwaving"]

if __name__ == "__main__":
    num_frames = 100
    # pick a random video from every category
    # find the interest points and save them
    for category in CATEGORIES:
        # get the video names
        video_names = [os.path.join(DATA_FOLDER, category, video_name)
                       for video_name in os.listdir(os.path.join(DATA_FOLDER, category))]
        # pick a random video
        video_name = random.choice(video_names)
        # read the video
        video = read_video(video_name, num_frames, 0)
        # get the interest points

        harris_points = MultiscaleDetector(lambda video, sigma, tau:
                                           HarrisDetector(video, 2, sigma, tau, kappa=0.005, threshold=0.25, num_points=500),
                                           video, [3*(1.1**i) for i in range(8)], tau=1.5, num_points=500)
        # save the detection
        # check that folder exists
        if not os.path.exists(os.path.join(RESULTS_FOLDER, "multiscale_harris", category)):
            os.makedirs(os.path.join(RESULTS_FOLDER, "multiscale_harris", category))
        show_detection(video, harris_points)
