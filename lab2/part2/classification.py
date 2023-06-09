import os
import numpy as np
import random
import pickle
from detector_utils import get_hog_hof, GaborDetector, HarrisDetector, MultiscaleDetector, NUM_POINTS
from cv23_lab2_2_utils import bag_of_words, read_video, svm_train_test

NUM_FRAMES = 200
NUM_POINTS = 600
SCALE = "uniscale"
TRAIN_NAMES_FILE = "train_names.txt"
TEST_NAMES_FILE = "test_names.txt"
DESCRIPTORS_FILE = f"classification_descriptors/{SCALE}_{NUM_POINTS}.pickle"
RESULTS_FILE = "classification_results/{scale}_{method}_{points}.txt"
VIDEO_FOLDER = "SpatioTemporal/{label}"

if __name__ == "__main__":
    if SCALE == "uniscale":
        harris = lambda video: HarrisDetector(video, 2, 3, 1.5, kappa=0.005, threshold=0.05)
        gabor = lambda video: GaborDetector(video, 3, 1.5, threshold=0.3)
    elif SCALE == "multiscale":
        harris = lambda video: MultiscaleDetector(lambda video, sigma, tau: HarrisDetector(video, 2, sigma, tau, kappa=0.005, threshold=0.05),
                                                  video, [3*(1.1**i) for i in range(6)], tau=1.5)
        gabor = lambda video: MultiscaleDetector(lambda video, sigma, tau: GaborDetector(video, sigma, tau, threshold=0.3),
                                                 video, sigmas=[3*(1.1**i) for i in range(6)], tau=1.5)
    else:
        raise ValueError("SCALE must be either 'uniscale' or 'multiscale'")
    # check if train_names.txt and test_names.txt exist
    # if they exist, load them:
    if os.path.exists(TRAIN_NAMES_FILE) and os.path.exists(TEST_NAMES_FILE):
        # read them
        with open(TRAIN_NAMES_FILE, "r") as f:
            train_names = f.read().splitlines()
        with open(TEST_NAMES_FILE, "r") as f:
            test_names = f.read().splitlines()
    # otherwise, create them:
    else:
        # get all the video names with full path
        # from current directory
        video_names = []
        for label in ["handwaving", "running", "walking"]:
            video_names += [os.path.join(VIDEO_FOLDER.format(label=label), video_name)
                            for video_name in os.listdir(VIDEO_FOLDER.format(label=label))]
        
        # shuffle them
        random.shuffle(video_names)
        
        # split them into train and test
        train_names = video_names[:int(0.7*len(video_names))]
        test_names = video_names[int(0.7*len(video_names)):]
        
        # save them
        with open(TRAIN_NAMES_FILE, "w") as f:
            f.write("\n".join(train_names))
        with open(TEST_NAMES_FILE, "w") as f:
            f.write("\n".join(test_names))

    # Define a dictionary for label mappings
    label_mappings = {
        "handwaving": 0,
        "running": 1,
        "walking": 2
    }

    # Generate train_labels using list comprehension
    train_labels = [label_mappings[name.split("_")[1]] for name in train_names]

    # Generate test_labels using list comprehension
    test_labels = [label_mappings[name.split("_")[1]] for name in test_names]

    # if descriptors.pickle exists, load it
    if os.path.exists(DESCRIPTORS_FILE):
        with open(DESCRIPTORS_FILE, "rb") as f:
            descriptors = pickle.load(f)
        hogs_train_gabor  = descriptors["hogs_train_gabor" ]
        hofs_train_gabor  = descriptors["hofs_train_gabor" ]
        hogs_train_harris = descriptors["hogs_train_harris"]
        hofs_train_harris = descriptors["hofs_train_harris"]
        hogs_test_gabor   = descriptors["hogs_test_gabor"  ]
        hofs_test_gabor   = descriptors["hofs_test_gabor"  ]
        hogs_test_harris  = descriptors["hogs_test_harris" ]
        hofs_test_harris  = descriptors["hofs_test_harris" ]
    # otherwise, create it
    else:
        # create the descriptors
        hogs_train_gabor,  hofs_train_gabor    = [], []
        hogs_train_harris, hofs_train_harris   = [], []
        hogs_test_gabor,   hofs_test_gabor     = [], []
        hogs_test_harris,  hofs_test_harris    = [], []
        print("Computing train descriptors...")
        for index, video_name in enumerate(train_names):
            print(f"\t{index} / {len(train_names)}")
            video = read_video(video_name, NUM_FRAMES, 0)
            
            print("\tfinding interest points with gabor...")
            gabor_points = gabor(video)
            print("\tfinished")
            print("\tfinding descriptors...")
            hogs, hofs = get_hog_hof(video, gabor_points, nbins=10)
            print("\tfinished")
            hogs_train_gabor.append(hogs)
            hofs_train_gabor.append(hofs)

            print("\tfinding interest points with harris...")
            harris_points = harris(video)
            print("\tfinished")
            print("\tfinding descriptors...")
            hogs, hofs = get_hog_hof(video, harris_points, nbins=10)
            print("\tfinished")
            hogs_train_harris.append(hogs)
            hofs_train_harris.append(hofs)

        print("Computing test descriptors...")
        for index, video_name in enumerate(test_names):
            print(f"\t{index} / {len(test_names)}")
            video = read_video(video_name, NUM_FRAMES, 0)
            print("\tfinding interest points with gabor...")
            gabor_points = gabor(video)
            print("\tfinished")
            print("\tfinding descriptors...")
            hogs, hofs = get_hog_hof(video, gabor_points, nbins=10)
            print("\tfinished")
            hogs_test_gabor.append(hogs)
            hofs_test_gabor.append(hofs)
            print("\tfinding interest points with harris...")
            harris_points = harris(video)
            print("\tfinished")
            print("\tfinding descriptors...")
            hogs, hofs = get_hog_hof(video, harris_points, nbins=10)
            print("\tfinished")
            hogs_test_harris.append(hogs)
            hofs_test_harris.append(hofs)
        # save the descriptors
        print("Saving descriptors...")
        with open(DESCRIPTORS_FILE, "wb") as f:
            pickle.dump({"hogs_train_gabor": hogs_train_gabor,
                         "hofs_train_gabor": hofs_train_gabor,
                         "hogs_train_harris": hogs_train_harris,
                         "hofs_train_harris": hofs_train_harris,
                         "hogs_test_gabor": hogs_test_gabor,
                         "hofs_test_gabor": hofs_test_gabor,
                         "hogs_test_harris": hogs_test_harris,
                         "hofs_test_harris": hofs_test_harris}, f)

    # for each method (gabor, harris)
    # try with different combinations
    # (hog, hof, hog+hof)
    # and print the results in a file
    print("Training and testing...")
    for train, test, method in zip([hogs_train_gabor, hofs_train_gabor,
                                    [np.concatenate((hogs, hofs)) for hogs, hofs in zip(hogs_train_gabor, hofs_train_gabor)],
                                    hogs_train_harris, hofs_train_harris,
                                    [np.concatenate((hogs, hofs)) for hogs, hofs in zip(hogs_train_harris, hofs_train_harris)]],
                                   [hogs_test_gabor, hofs_test_gabor,
                                    [np.concatenate((hogs, hofs)) for hogs, hofs in zip(hogs_test_gabor, hofs_test_gabor)],
                                    hogs_test_harris, hofs_test_harris,
                                    [np.concatenate((hogs, hofs)) for hogs, hofs in zip(hogs_test_harris, hofs_test_harris)]],
                                   ["gabor_hog", "gabor_hof", "gabor_hog_hof",
                                    "harris_hog", "harris_hof", "harris_hog_hof"]):
        # print method, train and test
        # for debugging purposes
        print(method)
        print(len(train), len(test))
        print(f"train shape: {train[0].shape}")
        print(f"test shape: {test[0].shape}")
        bow_train, bow_test = bag_of_words(train, test, num_centers=50)
        accuracy, pred = svm_train_test(bow_train, train_labels, bow_test, test_labels)
        # print results in file
        with open(RESULTS_FILE.format(scale=SCALE, method=method), "a") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Predictions: {pred}\n")
            f.write(f"True labels: {test_labels}\n")
            f.write("\n")

