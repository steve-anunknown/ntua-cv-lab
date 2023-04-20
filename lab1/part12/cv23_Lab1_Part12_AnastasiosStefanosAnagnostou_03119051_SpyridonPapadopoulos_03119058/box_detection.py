import cv2
import numpy as np
import matplotlib.pyplot as plt
from box_detection_utils import BoxFilters
from box_detection_utils import BoxLaplacian
from cv23_lab1_part2_utils import interest_points_visualization

# THIS WORKS NOW, HOWEVER IT IS NOT SPED UP.
# THE SPEED UP ACTUALLY DEPENDS ON HOW THE
# SIMPLE BLOB DETECTION FUNCTION WORKS.

# IF IT IS IMPLEMENTED USING THE BUILT IN
# FILTER2D AND GAUSSIAN FUNCTIONS, THEN 
# IT IS ALREADY REALLY FAST AND ITS TIME
# COMPLEXITY DOES NOT DEPEND ON THE SIGMA
# PARAMETER. IF A MANUAL CONVOLUTION HAS
# BEEN USED, THEN A HUGE SPEEDUP (OR ACTUALLY SLOW DOWN)
# WILL BE NOTICEABLE.

def boxtest():
    up = cv2.imread("cv23_lab1_part12_material/up.png")
    up = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
    gray = cv2.imread("cv23_lab1_part12_material/up.png", cv2.IMREAD_GRAYSCALE)
    gray = gray.astype(np.float64)/gray.max()

    # play around with the parameters
    sigma = 2.5
    theta = 0.05

    blobs = BoxFilters(gray, sigma, theta)
    interest_points_visualization(up, blobs, None)
    plt.savefig(f"image-plots/blob-detection-ii-up.jpg")

    # play around with the parameters
    sigma = 2.5
    theta = 0.05
    scale = 1.1
    N = 8

    blobs = BoxLaplacian(gray, sigma, theta, scale, N)
    interest_points_visualization(up, blobs, None)
    plt.savefig(f"image-plots/blob-detection-multiscale-ii-up.jpg")

    cells = cv2.imread("cv23_lab1_part12_material/cells.jpg")
    cells = cv2.cvtColor(cells, cv2.COLOR_BGR2RGB)
    gray = cv2.imread("cv23_lab1_part12_material/cells.jpg", cv2.IMREAD_GRAYSCALE)
    gray = gray.astype(np.float64)/gray.max()

    # play around with the parameters
    sigma = 3
    theta = 0.05

    blobs = BoxFilters(gray, sigma, theta)
    interest_points_visualization(cells, blobs, None)
    plt.savefig(f"image-plots/blob-detection-ii-cells.jpg")
    
    # play around with the parameters
    sigma = 3
    theta = 0.05
    scale = 1.1
    N = 8

    blobs = BoxLaplacian(gray, sigma, theta, scale, N)
    interest_points_visualization(cells, blobs, None)
    plt.savefig(f"image-plots/blob-detection-multiscale-ii-cells.jpg")

boxtest()
plt.show()
