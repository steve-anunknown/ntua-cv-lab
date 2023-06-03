from tracking_utils import lk
from tracking_utils import displ
from tracking_utils import makegif
from skin_detection_utils import FitSkinGaussian
from skin_detection_utils import fd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import cv2

if __name__ == "__main__":
    feats = 20
    padding = 20
    names = ["face", "right hand", "left hand"]
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
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    print(f"image.shape: {gray.shape}")
    print(f"face: {boundaries[0]}")
    print(f"right hand: {boundaries[1]}")
    print(f"left hand: {boundaries[2]}")
    dx0, dy0 = np.zeros(feats), np.zeros(feats)
    for i in range(1, 70):
        # load some image from the dataset in
        # the "part1-GreekSignLanguage" folder.
        image1 = cv2.imread(f"part1-GreekSignLanguage/{i}.png")
        image2 = cv2.imread(f"part1-GreekSignLanguage/{i+1}.png")
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


        fig, axs = plt.subplots(2, 3)
        axindices = {0: (0, 1), 1: (1, 2), 2: (1, 0)}
        for j in range(2):
            for k in range(3):
                if (j, k) not in axindices.values():
                    axs[j, k].axis("off")
        for index, boundary in enumerate(boundaries):
            y, x, h, w = boundary
            # crop the image to retain only the pixels
            # inside of the boundaries
            cropped1 = gray1[y-padding:y+h+padding, x-padding:x+w+padding]
            cropped2 = gray2[y-padding:y+h+padding, x-padding:x+w+padding]
            features = np.squeeze(cv2.goodFeaturesToTrack(cropped2, feats, 0.05, 5).astype(int))
            [dx, dy] = lk(cropped1, cropped2, features, 2, 0.01, dx0, dy0)
            
            # compute the flow
            [flowx, flowy] = displ(dx, dy, 0.7) 
            
            # update the boundaries
            x = int(round(x - flowx))
            y = int(round(y - flowy))
            boundaries[index] = (y, x, h, w)
            
            # show the new boundaries
            cv2.rectangle(image2, (x, y), (x + w, y + h), colours[index], 2)
            
            # plot the flow like a gradient field
            axs[axindices[index]].quiver(features[:, 0], features[:, 1], -dx, -dy, angles='xy')
            axs[axindices[index]].set_title(f"Optical flow for {names[index]}")

        axs[1, 1].imshow(image2)
        axs[1, 1].set_title(f"Frame {i+1}")
        plt.tight_layout()
        # save the figure
        plt.savefig(f"flow/{i+1}.png")
    makegif("flow/", "flow.gif")


