import os
import numpy as np
import random
import pickle
from detector_utils import get_hog_hof, GaborDetector, HarrisDetector
from cv23_lab2_2_utils import bag_of_words, read_video, svm_train_test


NUM_FRAMES = 200
TRAIN_NAMES_FILE = "train_names.txt"
TEST_NAMES_FILE = "test_names.txt"
DESCRIPTORS_FILE = "descriptors.pickle"
RESULTS_FILE = "results_{method}_hoghof.txt"
VIDEO_FOLDER = "SpatioTemporal/{label}"



if __name__ == "__main__":
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
            video_names += [os.path.join(VIDEO_FOLDER.format(label=label), video_name) for video_name in os.listdir(VIDEO_FOLDER.format(label=label))]
        
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
        hogs_train_gabor = descriptors["hogs_train_gabor"]
        hofs_train_gabor = descriptors["hofs_train_gabor"]
        hogs_train_harris = descriptors["hogs_train_harris"]
        hofs_train_harris = descriptors["hofs_train_harris"]
        hogs_test_gabor = descriptors["hogs_test_gabor"]
        hofs_test_gabor = descriptors["hofs_test_gabor"]
        hogs_test_harris = descriptors["hogs_test_harris"]
        hofs_test_harris = descriptors["hofs_test_harris"]
    # otherwise, create it
    else:
        # create the descriptors
        # TODO: the get_hog_hof function produces an error
        # it's either due to the parameters or the function itself
        hogs_train_gabor, hofs_train_gabor = [], []
        hogs_train_harris, hofs_train_harris = [], []
        hogs_test_gabor, hofs_test_gabor = [], []
        hogs_test_harris, hofs_test_harris = [], []
        for video_name in train_names:
            video = read_video(video_name, NUM_FRAMES, 0)
            gabor_points = GaborDetector(video, sigma=4, tau=1.5, threshold=0.2)
            hogs, hofs = get_hog_hof(video, gabor_points, sigma=4, nbins=10)
            hogs_train_gabor.append(hogs)
            hofs_train_gabor.append(hofs)

            harris_points = HarrisDetector(video, 2, sigma=4, tau=1.5, kappa=0.005, threshold=0.05)
            hogs, hofs = get_hog_hof(video, harris_points, sigma=4, nbins=10)
            hogs_train_harris.append(hogs)
            hofs_train_harris.append(hofs)

        for video_name in test_names:
            video = read_video(os.path.join(VIDEO_FOLDER, video_name), NUM_FRAMES, 0)
            gabor_points = GaborDetector(video, sigma=4, tau=1.5, threshold=0.2)
            hogs, hofs = get_hog_hof(video, gabor_points, sigma=4, nbins=10)
            hogs_test_gabor.append(hogs)
            hofs_test_gabor.append(hofs)

            harris_points = HarrisDetector(video, 2, sigma=4, tau=1.5, kappa=0.005, threshold=0.05)
            hogs, hofs = get_hog_hof(video, harris_points, sigma=4, nbins=10)
            hogs_test_harris.append(hogs)
            hofs_test_harris.append(hofs)

        # save the descriptors
        with open(DESCRIPTORS_FILE, "wb") as f:
            pickle.dump({"hogs_train_gabor": hogs_train_gabor,
                         "hofs_train_gabor": hofs_train_gabor,
                         "hogs_train_harris": hogs_train_harris,
                         "hofs_train_harris": hofs_train_harris,
                         "hogs_test_gabor": hogs_test_gabor,
                         "hofs_test_gabor": hofs_test_gabor,
                         "hogs_test_harris": hogs_test_harris,
                         "hofs_test_harris": hofs_test_harris}, f)

    # merge hogs and hofs
    descriptors_gabor = np.concatenate((hogs_train_gabor, hofs_train_gabor), axis=1)
    descriptors_harris = np.concatenate((hogs_train_harris, hofs_train_harris), axis=1)
    descriptors_test_gabor = np.concatenate((hogs_test_gabor, hofs_test_gabor), axis=1)
    descriptors_test_harris = np.concatenate((hogs_test_harris, hofs_test_harris), axis=1)
    train_descriptors = np.concatenate((descriptors_gabor, descriptors_harris), axis=1)
    test_descriptors = np.concatenate((descriptors_test_gabor, descriptors_test_harris), axis=1)
    
    for train, test, method in zip(train_descriptors, test_descriptors, ["gabor", "harris"]):
        bow_train, bow_test = bag_of_words(train, test, 50)
        accuracy, pred = svm_train_test(bow_train, train_labels, bow_test, test_labels)
        # print results in file
        with open(RESULTS_FILE.format(method=method), "a") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Predictions: {pred}\n")
            f.write(f"True labels: {test_labels}\n")
            f.write("\n")
        
        




