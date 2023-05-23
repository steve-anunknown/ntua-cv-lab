import skin_detection_utils
import numpy as np
import cv2

if __name__ == "__main__":
    # load some image from the dataset in
    # the "part1-GreekSignLanguage" folder.

    image = cv2.imread("part1-GreekSignLanguage/1/1_1.jpg")

    # load the skin samples that are stored in the
    # same folder in .mat format in order to fit
    # a gaussian curve on them.

    skin_samples = np.load("part1-GreekSignLanguage/skinSamplesRGB.mat")
    