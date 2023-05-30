from tracking_utils import lk
from tracking_utils import displ
from skin_detection_utils import FitSkinGaussian
from skin_detection_utils import fd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import cv2

if __name__ == "__main__":
    original = cv2.imread("part1-GreekSignLanguage/1.png")
    initial = cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB)
    skin_samples = scipy.io.loadmat("part1-GreekSignLanguage/skinSamplesRGB.mat")
    skin_samples = skin_samples["skinSamplesRGB"]
    skin_samples = cv2.cvtColor(skin_samples, cv2.COLOR_RGB2YCR_CB)
    mu, cov = FitSkinGaussian(skin_samples)
    boundaries = fd(initial, mu, cov)
    # find good features to track from the first image
    # in each boundary
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    features = []
    initial_guesses = []
    for boundary in boundaries:
        x, y, w, h = boundary
        cropped = gray[x:x+w, y:y+h]
        feats = cv2.goodFeaturesToTrack(cropped, 25, 0.01, 10)
        feats = np.squeeze(feats.astype(np.int32))
        features.append(feats)
        initial_guesses.append(np.array([np.zeros(len(feats)), np.zeros(len(feats))]))
    
    for i in range(1, 20):
        # load some image from the dataset in
        # the "part1-GreekSignLanguage" folder.
        image1 = cv2.imread(f"part1-GreekSignLanguage/{i}.png")
        image2 = cv2.imread(f"part1-GreekSignLanguage/{i+1}.png")
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        for index, boundary in enumerate(boundaries):
            x, y, w, h = boundary
            # crop the image to retain only the pixels
            # inside of the boundaries
            cropped1 = gray1[x:x+w, y:y+h]
            cropped2 = gray2[x:x+w, y:y+h]


            dx0, dy0 = initial_guesses[index]
            print(initial_guesses[index].shape)
            [dx, dy] = lk(cropped1, cropped2, features[index], 2, 0.04, dx0, dy0)
            # update the guesses
            initial_guesses[index] = np.array([dx, dy])
            # update the features
            features[index] = features[index] + np.array(list(zip(dx, dy)))


            # compute the flow
            [flowx, flowy] = displ(dx, dy, 0.1) 
            # update the boundaries
            x = int(x + flowx)
            y = int(y + flowy)
            boundaries[boundaries.index(boundary)] = (x, y, w, h)
            # show the new boundaries
            cv2.rectangle(image2, (y, x), (y+h, x+w), (0, 0, 255), 2)
            plt.quiver(-dx, -dy, angles='xy', scale=100)

        # show the image
        fig, axs = plt.subplots(1, 1)
        axs.imshow(image2)
        axs.set_title(f"Frame {i+1}")
        plt.show()


