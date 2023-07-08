from skin_detection_utils import FitSkinGaussian
from skin_detection_utils import fd
import matplotlib.pyplot as plt
import scipy.io
import cv2
import os

NUM_IMAGES = 2

if __name__ == "__main__":
    # load the skin samples that are stored in the
    # same folder in .mat format in order to fit
    # a gaussian curve on them.
    skin_samples = scipy.io.loadmat("part1-GreekSignLanguage/skinSamplesRGB.mat")
    skin_samples = skin_samples["skinSamplesRGB"]
    # first convert the rgb values to ycbcr
    skin_samples = cv2.cvtColor(skin_samples, cv2.COLOR_RGB2YCR_CB)
    # fit a gaussian curve on the skin samples
    mu, cov = FitSkinGaussian(skin_samples)
    print(f"Mean: {mu}")
    print(f"Covariance: {cov}")
    for i in range(1, 40, 5):
        # load some image from the dataset in
        # the "part1-GreekSignLanguage" folder.
        image = cv2.imread(f"part1-GreekSignLanguage/{i}.png")

        # convert the image to YCbCr color space
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        image_plot = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boundaries = fd(image_ycrcb, mu, cov)
        # draw the bounding boxes on the image
        for boundary in boundaries:
            # x, y, w, h = boundary
            y, x, h, w = boundary
            # cv2.rectangle(image, (y, x), (y+h, x+w), (0, 0, 255), 2)
            cv2.rectangle(image_plot, (x, y), (x+w, y+h), (0, 255, 255), 2)
        # show the image
        fig, axs = plt.subplots(1, 1)
        axs.imshow(image_plot)
        axs.set_title(f"Skin detection on frame {i}")
        # check if the results folder exists
        # if not, create it
        if not os.path.exists("results"):
            os.makedirs("results")
        plt.savefig(f"results/skin_detection_{i}.png")
        
    