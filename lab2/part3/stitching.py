import os
import cv2
from stitching_utils import stitchImages


IMAGE_PATH = 'ImageStitching'
IMAGE_NAME = 'img{num}_ratio01.jpg'
RESULT_PATH = 'StichingResults'
RESULT_NAME = 'img{num1}_{num2}_stitched.jpg'
NUM_IMAGES = 6

def main():
    # check if result path exists
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    # stitch images
    for i in range(1, NUM_IMAGES):
        img1 = cv2.imread(IMAGE_PATH + '/' + IMAGE_NAME.format(num=i))
        img2 = cv2.imread(IMAGE_PATH + '/' + IMAGE_NAME.format(num=i+1))
        stitched_img = stitchImages(img1, img2, match='bf')
        cv2.imwrite(RESULT_PATH + '/' + RESULT_NAME.format(num1=i, num2=i+1), stitched_img)
    # stich results together


if __name__ == '__main__':
    main()