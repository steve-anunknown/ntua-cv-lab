import cv2
import numpy as np
import matplotlib.pyplot as plt
from corner_detection_utils import CornerDetection
from corner_detection_utils import HarrisLaplacian
from cv23_lab1_part2_utils import interest_points_visualization

# TESTED AND WORKING

def cornertest():
    kyoto = cv2.imread("cv23_lab1_part12_material/kyoto_edges.jpg")
    kyoto = cv2.cvtColor(kyoto, cv2.COLOR_BGR2RGB)
    gray = cv2.imread("cv23_lab1_part12_material/kyoto_edges.jpg", cv2.IMREAD_GRAYSCALE)
    gray = gray.astype(np.float64)/gray.max()

    # play around with the parameters
    corners = CornerDetection(gray, 2, 2.5, 0.005, 0.1)
    interest_points_visualization(kyoto, corners, None)
    
    # play around with the parameters
    corners = HarrisLaplacian(gray, 2, 2.5, 0.05, 0.1, 1.1, 8)
    interest_points_visualization(kyoto, corners, None)

cornertest()
plt.show()