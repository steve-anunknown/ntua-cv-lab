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

    # play around with the parameters
    gray = gray.astype(np.float64)/gray.max()
    blobs = BlobDetection(gray, 2, 0.005)
    interest_points_visualization(up, blobs, None)

    # play around with the parameters
    blobs = HessianLaplacian(gray, 1.5, 0.05, 2, 4)
    interest_points_visualization(up, blobs, None)

    cells = cv2.imread("cv23_lab1_part12_material/cells.jpg")
    cells = cv2.cvtColor(cells, cv2.COLOR_BGR2RGB)
    gray = cv2.imread("cv23_lab1_part12_material/cells.jpg", cv2.IMREAD_GRAYSCALE)

    # play around with the parameters
    blobs = BlobDetection(gray, 4.5, 0.55)
    interest_points_visualization(cells, blobs, None)

    # play around with the parameters
    blobs = HessianLaplacian(gray, 4, 0.35, 1.1, 5)
    interest_points_visualization(cells, blobs, None)

blobtest()
plt.show()