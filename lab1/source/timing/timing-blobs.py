import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from box_detection_utils import BoxFilters
from box_detection_utils import BoxLaplacian
from blob_detection_utils import BlobDetection
from blob_detection_utils import HessianLaplacian

def timing():
    up = cv2.imread("cv23_lab1_part12_material/up.png")
    up = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
    gray = cv2.imread("cv23_lab1_part12_material/up.png", cv2.IMREAD_GRAYSCALE)
    gray = gray.astype(np.float64)/gray.max()

    # play around with the parameters
    sigma = 3.5
    theta = 0.5
    scale = 1.1
    N = 8

    testrange = list(range(2, 10))
    testnum = len(testrange)
    normal = np.zeros((testnum, 1))
    fast = np.zeros((testnum, 1))

    for index, sigma in enumerate(testrange):
        start = time.time()
        _ = BlobDetection(gray, sigma, theta)
        end = time.time()
        #print(f"sigma = {sigma} : Uniscale Blob Detection (No Integral Image) = {end-start}")
        normal[index] = end - start

        start = time.time()
        _ = BoxFilters(gray, sigma, theta)
        end = time.time()
        #print(f"sigma = {sigma} : Uniscale Blob Detection (Integral Image) = {end-start}")
        fast[index] = end - start

       # start = time.time()
       # blobs = HessianLaplacian(gray, sigma, theta, scale, N)
       # end = time.time()
       # print(f"sigma = {sigma} : Multiscale Blob Detection (No Integral Image) = {end-start}")
       # start = time.time()
       # blobs = BoxLaplacian(gray, sigma, theta, scale, N)
       # end = time.time()
       # print(f"sigma = {sigma} : Multiscale Blob Detection (Integral Image) = {end-start}")

    fig = plt.figure()
    plt.plot(testrange, normal, label="convolution")
    plt.plot(testrange, fast, label="integral images")
    plt.legend(loc="upper left")
    fig.suptitle("Timing of blob detection methods")
    plt.xlabel("sigma")
    plt.ylabel("time (s)")
    fig.savefig("image-plots/blob-comparison.png")
    plt.show()

timing()
