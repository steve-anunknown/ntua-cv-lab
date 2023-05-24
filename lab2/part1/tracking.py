from tracking_utils import lk
from tracking_utils import displ
from skin_detection_utils import FitSkinGaussian
from skin_detection_utils import fd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import cv2

if __name__ == "__main__":
    initial = cv2.imread("part1-GreekSignLanguage/1.png")
    initial = cv2.cvtColor(initial, cv2.COLOR_BGR2YCR_CB)
    skin_samples = scipy.io.loadmat("part1-GreekSignLanguage/skinSamplesRGB.mat")
    skin_samples = skin_samples["skinSamplesRGB"]
    skin_samples = cv2.cvtColor(skin_samples, cv2.COLOR_RGB2YCR_CB)
    mu, cov = FitSkinGaussian(skin_samples)
    boundaries = fd(initial, mu, cov)
    dx0, dy0 = 0, 0
    for i in range(1, 20):
        # load some image from the dataset in
        # the "part1-GreekSignLanguage" folder.
        image1 = cv2.imread(f"part1-GreekSignLanguage/{i}.png")
        image2 = cv2.imread(f"part1-GreekSignLanguage/{i+1}.png")

        # convert the image to YCbCr color space
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2YCR_CB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2YCR_CB)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        for boundary in boundaries:
            x, y, w, h = boundary
            # crop the image to retain only the pixels
            # inside of the boundaries
            cropped1 = gray1[x:x+w, y:y+h]
            cropped2 = gray2[x:x+w, y:y+h]
            features = cv2.goodFeaturesToTrack(cropped2, 25, 0.01, 10)
            # use squeeze to remove the redundant dimension
            features = np.squeeze(features.astype(np.int32))
            [dx, dy] = lk(cropped1, cropped2, features, 1, 0.001, dx0, dy0)
            [nextx, nexty] = displ(dx, dy, "energy")
            # readjust the boundaries
            x += nextx
            y += nexty
            # show the new boundaries
            cv2.rectangle(image2, (y, x), (y+h, x+w), (0, 0, 255), 2)
        # show the image
        fig, axs = plt.subplots(1, 1)
        axs.imshow(image2)
        axs.set_title(f"Frame {i+1}")
        plt.show()


