import cv2
import numpy as np
import matplotlib.pyplot as plt
from blob_detection_utils import BlobDetection
from blob_detection_utils import HessianLaplacian
from cv23_lab1_part2_utils import interest_points_visualization

# TESTED AND WORKING

def blobtest():
    up = cv2.imread("cv23_lab1_part12_material/up.png")
    up = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
    gray = cv2.imread("cv23_lab1_part12_material/up.png", cv2.IMREAD_GRAYSCALE)
    gray = gray.astype(np.float64)/gray.max()

    # play around with the parameters
    sigma = 2.5
    theta = 0.005
    scale = 1.1
    N = 8

    blobs = BlobDetection(gray, sigma, theta)
    interest_points_visualization(up, blobs, None)

    # play around with the parameters
    blobs = HessianLaplacian(gray, sigma, theta, scale, N)
    interest_points_visualization(up, blobs, None)

    cells = cv2.imread("cv23_lab1_part12_material/cells.jpg")
    cells = cv2.cvtColor(cells, cv2.COLOR_BGR2RGB)
    gray = cv2.imread("cv23_lab1_part12_material/cells.jpg", cv2.IMREAD_GRAYSCALE)
    gray = gray.astype(np.float64)/gray.max()

    # play around with the parameters
    sigma = 3
    theta = 0.005
    scale = 1.1
    N = 8

    blobs = BlobDetection(gray, sigma, theta)
    interest_points_visualization(cells, blobs, None)

    # play around with the parameters
    blobs = HessianLaplacian(gray, sigma, theta, scale, N)
    interest_points_visualization(cells, blobs, None)

blobtest()
plt.show()