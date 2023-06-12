import cv2
import numpy as np


def projectionImage(H, img):
    """
    Apply inverse warping to img1.

    Keyword arguments:
    H -- homography matrix
    img -- image to be warped
    """

    # Step1: Compute the warped coordinates of the four corners of the image
    # to obtain the coordinate of the top-left corner and bottom-right corner
    # of the warped image
    h, w = img.shape[:2]
    corners = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)
    corners = np.array([corners])
    warped_corners = cv2.perspectiveTransform(corners, H)[0]
    warped_corners = warped_corners.astype(np.int32)
    warped_topleft_coords = warped_corners.min(axis=0)
    warped_bottomright_coords = warped_corners.max(axis=0)

    # Step2: Compute the translation matrix that will shift the image
    # such that the top-left corner of the image will move to the origin
    # (0, 0) of the coordinate system
    tx = -warped_topleft_coords[0]
    ty = -warped_topleft_coords[1]
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    # Step3: Compute the warped image by applying the inverse warping matrix
    # to the image
    warped_img = cv2.warpPerspective(img, T.dot(H), (warped_bottomright_coords[0], warped_bottomright_coords[1]))

    return warped_img, warped_topleft_coords

def mergeWarpedImages(img1_warped, img2, img1_topleft_coords):
    """
    Merge the warped version of img1 with img2 under the same coordinate system of img2.
    
    Keyword arguments:
    img1_warped -- warped version of img1
    img2 -- second image
    img1_topleft_coords -- top-left corner coordinates of img1
    """
    
    # Step1: Compute the translation matrix that will shift the image
    # such that the top-left corner of the image will move to the origin
    # (0, 0) of the coordinate system
    tx = -img1_topleft_coords[0]
    ty = -img1_topleft_coords[1]
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    # Step2: Compute the merged image by applying the translation matrix
    # to the warped image and then merging it with the second image
    merged_img = cv2.warpPerspective(img1_warped, T, (img2.shape[1], img2.shape[0]))
    merged_img[0:img2.shape[0], 0:img2.shape[1]] = img2

    return merged_img

def stitchImages(img1, img2, match = 'bf'):
    """
    Stitch two images together.
    
    Keyword arguments:
    img1 -- first image
    img2 -- second image
    """

    # Step1: Extract SIFT features and descriptors from both images
    sift1 = cv2.xfeatures2d.SIFT_create()
    sift2 = cv2.xfeatures2d.SIFT_create()
    points1, descriptors1 = sift1.detectAndCompute(img1, None)
    points2, descriptors2 = sift2.detectAndCompute(img2, None)

    # Step2: Match features by applying Brute-Force-based or FLANN-based matching
    if match == 'bf':
        bf = cv2.BFMatcher()
        matches = bf.match(descriptors1, descriptors2)
    elif match == 'flann':
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        k = 2
        matches = flann.knnMatch(descriptors1, descriptors2, k)
    
    # Step3: Apply Lowe's criterion to speedily reject most outliers
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    matches = good_matches

    # Step4: Compute RANSAC-based Homography H
    # given the matches between image 1 and image 2
    # we compute the homography H21 that maps
    # points from image 1 to image 2
    # x2 = H21 * x1

    H = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)[0]

    # Step5: Apply inverse warping to img1

    img1_warped, img1_topleft_coords = projectionImage(H, img1)

    # Step6: Merge the warped version of img1 with img2 under the same coordinate system of img2

    stiched_img = mergeWarpedImages(img1_warped, img2, img1_topleft_coords)

    return stiched_img