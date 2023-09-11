import os
import sys
import time
import random
from detector_utils import GaborDetector, GaborDetectorTrial, GaborDetectorTrial2, time_function
from detector_utils import MultiscaleDetector
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import show_detection

RESULTS_FOLDER = "results"
DATA_FOLDER = "SpatioTemporal"
CATEGORIES = ["walking", "running", "handwaving"]
TEST = False
if __name__ == "__main__":
    num_frames = 100
    # pick a random video from every category
    # find the interest points and save them
    if TEST:
        # get name from command line
        if len(sys.argv) != 2:
            print("Usage: python3 multiscale_gabor.py <video_name>")
            sys.exit(1)
        video_name = sys.argv[1]
        # read the video
        video = read_video(video_name, num_frames, 0)
        # get the interest points
        scales = [3*(1.1**i) for i in range(9)]
        start = time.time()
        h_ev, h_od = time_function(tau=1.5)
        detector = lambda video, sigma: GaborDetectorTrial2(video, sigma, h_ev, h_od, threshold=0.3, num_points=500)
        gabor_points = MultiscaleDetector(detector, video, scales, tau=1.5, num_points=500)
        end = time.time()
        print("Time: {}".format(end-start))
        start = time.time()
        detector = lambda video, sigma: GaborDetectorTrial(video, sigma, tau=1.5, threshold=0.3, num_points=500)
        gabor_points = MultiscaleDetector(detector, video, scales, tau=1.5, num_points=500)
        end = time.time()
        print("Time: {}".format(end-start))

        start = time.time()
        detector = lambda video, sigma: GaborDetector(video, sigma, tau=1.5, threshold=0.3, num_points=500)
        gabor_points = MultiscaleDetector(detector, video, scales, tau=1.5, num_points=500)
        end = time.time()
        print("Time: {}".format(end-start))

    else:
        for category in CATEGORIES:
            # get the video names
            video_names = [os.path.join(DATA_FOLDER, category, video_name)
                           for video_name in os.listdir(os.path.join(DATA_FOLDER, category))]
            # pick a random video
            video_name = random.choice(video_names)
            # read the video
            video = read_video(video_name, num_frames, 0)
            # get the interest points
            gabor_points = MultiscaleDetector(lambda video, sigma, tau:
                                              GaborDetector(video, sigma, tau, threshold=0.3, num_points=500),
                                              video, sigmas=[3*(1.1**i) for i in range(6)], tau=1.5, num_points=500)
            # save the detection
            # check that folder exists
            if not os.path.exists(os.path.join(RESULTS_FOLDER, "multiscale_gabor", category)):
                os.makedirs(os.path.join(RESULTS_FOLDER, "multiscale_gabor", category))
            show_detection(video, gabor_points, save_path=os.path.join(RESULTS_FOLDER, "multiscale_gabor", category))
