import numpy as np
import matplotlib.pyplot as plt
import cv2
from edge_detection_utils import EdgeDetect
from edge_detection_utils import QualityMetric

# TESTED AND WORKING

def edgedetectreal():
    # read image and normalize it
    kyoto = cv2.imread("cv23_lab1_part12_material/kyoto_edges.jpg", cv2.IMREAD_GRAYSCALE)
    kyoto = kyoto.astype(np.float64)/kyoto.max()

    # play around with sigma and theta
    # big sigma => much smoothing => not fine details
    # big theta => less edges, small theta => many edges

    sigma = 0.3
    theta = 0.2
    thetareal = 0.23

    N1 = EdgeDetect(kyoto, sigma, theta, "linear")
    N2 = EdgeDetect(kyoto, sigma, theta, "nonlinear")

    # the non linear method gives the best results,
    # therefore we name it D and continue our evaluation
    D = N2
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    M = cv2.dilate(kyoto, cross) - cv2.erode(kyoto, cross)
    T = ( M > thetareal ).astype(np.uint8)

    fig, axs = plt.subplots(2,2)
    axs[0, 0].imshow(kyoto, cmap='gray')
    axs[0, 0].set_title("Noised Image")
    axs[0, 1].imshow(T, cmap='gray')
    axs[0, 1].set_title("Actual Edges")
    axs[1, 0].imshow(N1, cmap='gray')
    axs[1, 0].set_title("Linear edge detection")
    axs[1, 1].imshow(N2, cmap='gray')
    axs[1, 1].set_title("Non Linear edge detection")
    plt.show(block=False)
    plt.pause(0.01)

    C = QualityMetric(T, D)
    print(f"The quality criterion of the linear edge detection is C = {C}")
    C = QualityMetric(T, N1)
    print(f"The quality criterion of the non linear edge detection is C = {C}")

edgedetectreal()
plt.show()