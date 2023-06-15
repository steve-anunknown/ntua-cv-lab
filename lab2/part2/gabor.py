import os
import sys
import random
from itertools import product
from detector_utils import GaborDetector
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import show_detection

RESULTS_FOLDER = "results_trial/gabor_{sigma}_{tau}_{threshold}_{num_points}"
DATA_FOLDER = "SpatioTemporal"
CATEGORIES = ["walking", "running", "handwaving"]
PARAMETERS = {
    "sigma": [2, 3, 4, 5],
    "tau": [1.5, 2, 2.5, 3],
    "threshold": [0.1, 0.15, 0.2, 0.25, 0.3],
    "num_points": [500]
}

if __name__ == "__main__":
    num_frames = {"walking": 70, "running": 50, "handwaving": 50}
    # pick a random video from every category
    # find the interest points and save them
    combinations = product(*PARAMETERS.values())
    for combination in combinations:
        sigma, tau, threshold, num_points = combination
        results_folder = RESULTS_FOLDER.format(sigma=sigma, tau=tau, threshold=threshold, num_points=num_points)    
        for category in CATEGORIES:
            frames = num_frames[category]
            # get the video names
            video_names = [os.path.join(DATA_FOLDER, category, video_name)
                        for video_name in os.listdir(os.path.join(DATA_FOLDER, category))]
            # pick a random video
            video_name = random.choice(video_names)
            # read the video
            video = read_video(video_name, frames, 0)
            # get the interest points
            gabor_points = GaborDetector(video, sigma, tau, threshold, num_points)
            # save the interest points
            if not os.path.exists(os.path.join(results_folder, category)):
                os.makedirs(os.path.join(results_folder, category))
            show_detection(video, gabor_points, save_path=os.path.join(results_folder, category))
            

