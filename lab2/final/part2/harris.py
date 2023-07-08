import os
import sys
import random
from itertools import product
from detector_utils import HarrisDetector
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import show_detection


RESULTS_FOLDER = "results_trial/harris_{sigma}_{tau}_{threshold}_{num_points}"
DATA_FOLDER = "SpatioTemporal"
CATEGORIES = ["walking", "running", "handwaving"]
PARAMETERS = {
    "sigma": [2, 3, 4, 5],
    "tau": [1.5, 2, 2.5, 3],
    "kappa": [0.005],
    "threshold": [0.15],
    "num_points": [500]
}

if __name__ == "__main__":
    num_frames = 100
    combinations = product(*PARAMETERS.values())
    for combination in combinations:
        sigma, tau, kappa, threshold, num_points = combination
        results_folder = RESULTS_FOLDER.format(sigma=sigma, tau=tau, threshold=threshold, num_points=num_points)    
        for category in CATEGORIES:
            # get the video names
            video_names = [os.path.join(DATA_FOLDER, category, video_name)
                        for video_name in os.listdir(os.path.join(DATA_FOLDER, category))]
            # pick a random video
            video_name = random.choice(video_names)
            # read the video
            video = read_video(video_name, num_frames, 0)
            # get the interest points
            harris_points = HarrisDetector(video, s=2, sigma=sigma, tau=tau, kappa=0.005,
                                           threshold=threshold, num_points=num_points)
            # save the interest points
            if not os.path.exists(os.path.join(results_folder, category)):
                os.makedirs(os.path.join(results_folder, category))
            show_detection(video, harris_points, save_path=os.path.join(results_folder, category))