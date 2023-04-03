import cv2
import numpy as np
import matplotlib.pyplot as plt
from box_detection_utils import BoxFilters
from box_detection_utils import BoxLaplacian
from cv23_lab1_part2_utils import interest_points_visualization

# Either the implementation of the function is not good enough
# or is is good enough but the proposed approximation is bad.

# For the same set of parameters, the blob detection is 
# accurate but the box filter method detects some non existent
# blobs. Perhaps it needs a greater theta.

def boxtest():
    up = cv2.imread("cv23_lab1_part12_material/up.png")
    up = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
    gray = cv2.imread("cv23_lab1_part12_material/up.png", cv2.IMREAD_GRAYSCALE)
    gray = gray.astype(np.float64)/gray.max()

    # play around with the parameters
    sigma = 2.5
    theta = 0.03

    blobs = BoxFilters(gray, sigma, theta)
    interest_points_visualization(up, blobs, None)

    # play around with the parameters
    sigma = 2.5
    theta = 0.3
    scale = 1.1
    N = 8

    blobs = BoxLaplacian(gray, sigma, theta, scale, N)
    interest_points_visualization(up, blobs, None)

    cells = cv2.imread("cv23_lab1_part12_material/cells.jpg")
    cells = cv2.cvtColor(cells, cv2.COLOR_BGR2RGB)
    gray = cv2.imread("cv23_lab1_part12_material/cells.jpg", cv2.IMREAD_GRAYSCALE)
    gray = gray.astype(np.float64)/gray.max()

    # play around with the parameters
    sigma = 3
    theta = 0.025

    blobs = BoxFilters(gray, sigma, theta)
    interest_points_visualization(cells, blobs, None)
    
    # play around with the parameters
    sigma = 3
    theta = 0.1
    scale = 1.1
    N = 8

    blobs = BoxLaplacian(gray, sigma, theta, scale, N)
    interest_points_visualization(cells, blobs, None)

boxtest()
plt.show()