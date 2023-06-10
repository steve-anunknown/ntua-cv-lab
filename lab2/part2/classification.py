import os
import numpy as np
import random
import pickle
from memory_profiler import profile
from detector_utils import get_hog_hof, get_hof_descriptors, get_hog_descriptors, GaborDetector, HarrisDetector, MultiscaleDetector
from cv23_lab2_2_utils import bag_of_words, read_video, svm_train_test

NBINS = 10
NUM_FRAMES = 200

TRAIN_NAMES_FILE = "train_names.txt"
TEST_NAMES_FILE = "test_names.txt"

VIDEO_FOLDER = "SpatioTemporal/{label}"

# define dictionary for multiple tests
# each test is a dictionary with keys:
# scale, detector, descriptor, points
PARAMETERS = {
    "scale": ["uniscale", "multiscale"],
    "detector": ["gabor", "harris"],
    "descriptor": ["hog_hof", "hog", "hof"],
    "points": [500]
}
DESCRIPTORS_FILE = "classification_descriptors/{scale}_{detector}_{descriptor}_{points}.pickle"
RESULTS_FILE = "classification_results/{scale}_{detector}_{descriptor}_{points}.txt"

def init_detectors(scale, detector):
    """
    Returns the detector function.

    Keyword arguments:
    scale -- the scale of the detector
    detector -- the detector to use
    """
    print("\n\tInitializing detector...")
    if scale == "multiscale":
        if detector == "harris":
            detector = lambda video, points: MultiscaleDetector(lambda video, sigma, tau:
                                                                HarrisDetector(video, 2, sigma, tau, kappa=0.005, threshold=0.05, num_points=points),
                                                                video, [3*(1.1**i) for i in range(6)], tau=1.5, num_points=points)
        elif detector == "gabor":
            detector = lambda video, points: MultiscaleDetector(lambda video, sigma, tau:
                                                                GaborDetector(video, sigma, tau, threshold=0.5, num_points=points),
                                                                video, [3*(1.1**i) for i in range(6)], tau=1.5, num_points=points)
        else:
            raise ValueError("DETECTOR must be either 'harris' or 'gabor'")
    elif scale == "uniscale":
        if detector == "harris":
            detector = lambda video, points: HarrisDetector(video, s=2, sigma=4, tau=1.5,
                                                            kappa=0.005, threshold=0.05, num_points=points)
        elif detector == "gabor":
            detector = lambda video, points: GaborDetector(video, sigma=4, tau=1.5,
                                                           threshold=0.25, num_points=points)
        else:
            raise ValueError("DETECTOR must be either 'harris' or 'gabor'")
    print("\tFinished ...")
    return detector

def init_descriptors(descriptor):
    """
    Returns the get_descriptors function.
    
    Keyword arguments:
    descriptor -- the descriptor to use
    """
    print("\n\tInitializing descriptor...")
    if descriptor == "hog":
        get_descriptors = lambda video, points: get_hog_descriptors(video, points, nbins=NBINS)
    elif descriptor == "hof":
        get_descriptors = lambda video, points: get_hof_descriptors(video, points, nbins=NBINS)
    elif descriptor == "hog_hof":
        get_descriptors = lambda video, points: get_hog_hof(video, points, nbins=NBINS)
    else:
        raise ValueError("DESCRIPTOR must be either 'hog', 'hof' or 'hog_hof'")
    print("\tFinished ...")
    return get_descriptors

def get_names():
    """
    Returns the train and test names.
    If train_names.txt and test_names.txt exist,
    it loads them.
    Otherwise, it creates them.
    """
    print("\n\tGetting names...")
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
    print("\tFinished ...")
    return train_names, test_names

def extract_descriptors(names, detector, get_descriptors, points):
    """
    Extracts the descriptors from the videos.
    
    Keyword arguments:
    names -- the names of the videos
    detector -- the detector to use
    get_descriptors -- the descriptor to use
    points -- the number of points to use
    """
    print("\n\tExtracting descriptors...")
    descriptors = []
    for name in names:
        video = read_video(name, NUM_FRAMES, 0)
        interest = detector(video, points)
        descs = get_descriptors(video, interest)
        descriptors.append(descs)
    print("\tChecking if all descriptors have the same shape...")
    original_shape = descriptors[0].shape
    print(f"\toriginal_shape: {original_shape}")
    for index, descriptor in enumerate(descriptors):
        if descriptor.shape != original_shape:
            raise ValueError(f"All descriptors must have the same shape. Descriptor {index} has shape {descriptor.shape}")
    print("\tFinished ...")
    return descriptors

def run_test(scale, detector, descriptor, points):
    """
    Runs the test for the current parameters.
    
    Saves the descriptors in a pickle file.
    Saves the results in a text file.

    Keyword arguments:
    scale -- the scale of the detector
    detector -- the detector to use
    descriptor -- the descriptor to use
    points -- the number of points to use
    """
    fun_detector = init_detectors(scale, detector)
    get_descriptors = init_descriptors(descriptor)
    train_names, test_names = get_names()
    
    # Define a dictionary for label mappings
    label_mappings = { "handwaving": 0, "running": 1, "walking": 2 }

    # Generate test_labels
    test_labels = [label_mappings[name.split("_")[1]] for name in test_names]
    # Generate train_labels
    train_labels = [label_mappings[name.split("_")[1]] for name in train_names]
    
    # check if descriptors have already been computed
    if os.path.exists(DESCRIPTORS_FILE.format(scale=scale, detector=detector, descriptor=descriptor, points=points)):
        print("\n\tLoading descriptors...")
        with open(DESCRIPTORS_FILE.format(scale=scale, detector=detector, descriptor=descriptor, points=points), "rb") as f:
            descriptors = pickle.load(f)
        test_descriptors = descriptors["test"]
        train_descriptors = descriptors["train"]
    else:
        # Extract descriptors
        test_descriptors = extract_descriptors(test_names, fun_detector, get_descriptors, points)
        train_descriptors = extract_descriptors(train_names, fun_detector, get_descriptors, points)

    # train and test
    bow_train, bow_test = bag_of_words(train_descriptors, test_descriptors, num_centers=50)
    accuracy, pred = svm_train_test(bow_train, train_labels, bow_test, test_labels)
    with open(RESULTS_FILE.format(scale=scale, detector=detector, descriptor=descriptor, points=points), "w") as f:
        f.write("Accuracy: {accuracy}\n".format(accuracy=accuracy))
        f.write("Predictions: {pred}\n".format(pred=pred))
        f.write("Test labels: {test_labels}\n".format(test_labels=test_labels))

    with open(DESCRIPTORS_FILE.format(scale=scale, detector=detector, descriptor=descriptor, points=points), "wb") as f:
        pickle.dump({"test": test_descriptors, "train": train_descriptors}, f)

def main():
    # perform the test for every combination of parameters
    for scale in PARAMETERS["scale"]:
        for detector in PARAMETERS["detector"]:
            for descriptor in PARAMETERS["descriptor"]:
                for points in PARAMETERS["points"]:                    
                    # run the test
                    print(f"Test: scale={scale}, detector={detector}, descriptor={descriptor}, points={points}")
                    run_test(scale, detector, descriptor, points)


if __name__ == "__main__":
    main()