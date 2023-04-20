import matplotlib.pyplot as plt
import numpy as np
import cv2
from intro_utils import getstd
from edge_detection_utils import EdgeDetect
from edge_detection_utils import QualityMetric

# TESTED AND WORKING

def edgedetectintro():
    # read the image, convert to gray scale and normalize it
    image = cv2.imread("cv23_lab1_part12_material/edgetest_23.png", cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float64)/image.max()

    fig, axs = plt.subplots(1,1)
    axs.imshow(image, cmap='gray')
    axs.set_title("Original Image")
    plt.show(block=False)
    plt.pause(0.01)

    # add noise to the images. 10db and 20db
    # in the psnr context, less dBs equals more noise
    image10db = image + np.random.normal(0, getstd(image, 10), image.shape)
    image20db = image + np.random.normal(0, getstd(image, 20), image.shape)

    # Play around with sigma and theta in order to
    # obtain the best results. 
    noised_images = [image10db, image20db]
    sigma = [3, 1.5]
    theta = [0.2, 0.2]
    thetareal = 0.08

    for index, img in enumerate(noised_images):
        N1 = EdgeDetect(img, sigma[index], theta[index], "linear")
        N2 = EdgeDetect(img, sigma[index], theta[index], "nonlinear")

        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        M = cv2.dilate(image, cross) - cv2.erode(image, cross)
        T = ( M > thetareal ).astype(np.uint8)

        fig, axs = plt.subplots(2,2)
        axs[0, 0].imshow(img, cmap='gray')
        axs[0, 0].set_title("Noised Image")
        axs[0, 1].imshow(N1, cmap='gray')
        axs[0, 1].set_title("Linear edge detection")
        axs[1, 0].imshow(N2, cmap='gray')
        axs[1, 0].set_title("Non linear edge detection")
        axs[1, 1].imshow(T, cmap='gray')
        axs[1, 1].set_title("Actual Edges")
        plt.show(block=False)
        plt.pause(0.01)
        plt.savefig(f"image-plots/edges-intro{index}.jpg")

        C = QualityMetric(T, N1)
        print(f"Linear method: C[{index}] = {C}")
        C = QualityMetric(T, N2)
        print(f"Non linear method: C[{index}] = {C}")

edgedetectintro()
plt.show()
