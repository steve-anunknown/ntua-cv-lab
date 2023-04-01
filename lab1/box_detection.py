import cv2
import numpy as np
import matplotlib.pyplot as plt
from box_detection_utils import BoxFilters
from box_detection_utils import BoxLaplacian
from cv23_lab1_part2_utils import interest_points_visualization

# NOT TESTED

def boxtest():
    up = cv2.imread("cv23_lab1_part12_material/up.png")
    up = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
    gray = cv2.imread("cv23_lab1_part12_material/up.png", cv2.IMREAD_GRAYSCALE)

    # play around with the parameters
    gray = gray.astype(np.float64)/gray.max()
    blobs = BoxFilters(gray, 2.5, 0.25)
    interest_points_visualization(up, blobs, None)


boxtest()
plt.show()