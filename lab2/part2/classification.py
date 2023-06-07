import os
import sys
import random
from detector_utils import hof_descriptors
from detector_utils import hog_descriptors
from detector_utils import GaborDetector
from detector_utils import MultiscaleDetector
from cv23_lab2_2_utils import bag_of_words
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import svm_train_test

if __name__ == "__main__":
    num_frames = 200
    # get video name from the command line
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
    video = read_video(video_name, num_frames, 0)
    gabor_points = MultiscaleDetector(lambda video, sigma, tau: GaborDetector(video, sigma, tau),
                                        video, [3*(1.1**i) for i in range(6)], 1.5)
    
    # first get all the video names from the SpatioTemporal folder
    categories = os.listdir("SpatioTemporal")
    names = []
    for category in categories:
        names += os.listdir(category)
    # then split them into training and testing sets
    random.shuffle(names)
    test_names = names[:int(0.3*len(names))]
    train_names = names[int(0.3*len(names)):]
    # then get the descriptors for each video
    train_descriptors = []
    train_labels = []
    for name in train_names:
        category = name.split("_")[1]
        video = read_video("SpatioTemporal/"+category+"/"+name, num_frames, 0)
        hof = hof_descriptors(video, gabor_points, 10, 10)
        train_descriptors.append(hof)
        train_labels.append(category)
    test_descriptors = []
    test_labels = []
    for name in test_names:
        category = name.split("_")[1]
        video = read_video("SpatioTemporal/"+category+"/"+name, num_frames, 0)
        hof = hof_descriptors(video, gabor_points, 10, 10)
        test_descriptors.append(hof)
        test_labels.append(category)
    # then train a bag of words model
    bow_train, bow_test = bag_of_words(train_descriptors, test_descriptors, num_centers=100)
    # TODO: train the svm classifier
    # TODO: report the accuracy
    # TODO: experiment with different detectors and descriptors

