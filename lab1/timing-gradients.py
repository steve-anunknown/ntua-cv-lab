import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from intro_utils import smooth_gradient
from intro_utils import smooth_gradient2
from box_detection_utils import BoxDerivative

def timing():
    up = cv2.imread("cv23_lab1_part12_material/up.png")
    up = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
    gray = cv2.imread("cv23_lab1_part12_material/up.png", cv2.IMREAD_GRAYSCALE)
    gray = gray.astype(np.float64)/gray.max()

    # play around with the parameters

    testrange = list(range(2, 10))
    testnum = len(testrange)
    normal = np.zeros((testnum, 1))
    manual = np.zeros((testnum, 1))
    fast = np.zeros((testnum, 1))

    for index, sigma in enumerate(testrange):
        start = time.time()
        _, _, _ = smooth_gradient(gray, sigma, 2)
        end = time.time()
        normal[index] = end - start
        #print(f"manual convolution is {end - start}")
        start = time.time()
        _, _, _ = smooth_gradient2(gray, sigma, 2)
        end = time.time()
        manual[index] = end - start
        #print(f"cv2 filter2D is      {end - start}")
        start = time.time()
        _, _, _ = BoxDerivative(gray, sigma)
        end = time.time()
        fast[index] = end - start
        #print(f"box derivative is    {end - start}")

    fig = plt.figure()
    plt.plot(testrange, normal, label="manual convolution")
    plt.plot(testrange, manual, label="filter2D convolution")
    plt.plot(testrange, fast, label="box derivative")
    plt.legend(loc="upper left")
    fig.suptitle("Timing of gradient computation methods")
    plt.xlabel("sigma")
    plt.ylabel("time (s)")
    fig.savefig("image-plots/grad-comparison.png")
    plt.show()

timing()