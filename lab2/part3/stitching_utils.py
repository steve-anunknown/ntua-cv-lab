import cv2
import numpy as np
from matplotlib import pyplot as plt



def combine_images(img1, img2, H):
    """
    Combine two images together.
    
    Keyword arguments:
    img1 -- first image
    img2 -- second image
    H -- homography matrix
    """
    points1 = np.array([[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]], dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))

    points2 = np.array([[0, 0], [0, img2.shape[0]], [img2.shape[1], img2.shape[0]], [img2.shape[1], 0]], dtype=np.float32)
    points2 = points2.reshape((-1, 1, 2))

    warped_points1 = cv2.perspectiveTransform(points1, H)[0]
    warped_points1 = warped_points1.astype(np.int32)
    warped_points1 = warped_points1.reshape((-1, 1, 2))

    points = np.concatenate((points2, warped_points1), axis=0)

    [x_min, y_min] = np.min(points, axis=0).ravel().astype(np.int32)
    [x_max, y_max] = np.max(points, axis=0).ravel().astype(np.int32)
    
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    
    output = cv2.warpPerspective(img1, T.dot(H), (x_max-x_min, y_max-y_min))
    # show warped image
    plt.imshow(output)
    plt.show()
    cv2.imwrite('StichingResults/warped.jpg', output)
    exit()
    output[-y_min:img2.shape[0]-y_min, -x_min:img2.shape[1]-x_min] = img2
    
    return output

def projectionImage(H, img):
    """
    Apply inverse warping to img.

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
    [x_min, y_min] = warped_topleft_coords
    [x_max, y_max] = warped_bottomright_coords
    tx = -warped_topleft_coords[0]
    ty = -warped_topleft_coords[1]
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    
    # Step3: Compute the warped image by applying the inverse warping matrix
    # to the image
    warped_img = cv2.warpPerspective(img, T.dot(H),
                                     (x_max-x_min, y_max-y_min))
    warped_img[-y_min:h-y_min, -x_min:w-x_min] = img
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
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

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
    keypoints1, descriptors1 = sift1.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift2.detectAndCompute(img2, None)
    # Convert keypoints to numpy arrays
    points1 = np.float32([kp.pt for kp in keypoints1]).reshape(-1, 1, 2)
    points2 = np.float32([kp.pt for kp in keypoints2]).reshape(-1, 1, 2)

    # Step2: Match features by applying Brute-Force-based or FLANN-based matching
    k = 2
    if match == 'bf':
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k)
    elif match == 'flann':
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k)
    
    # Step3: Apply Lowe's criterion to speedily reject most outliers
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    matches = good

    points1 = np.float32([points1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([points2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

    # Step4: Compute RANSAC-based Homography H
    # given the matches between image 1 and image 2
    # we compute the homography H21 that maps
    # points from image 1 to image 2
    # x2 = H21 * x1
    H = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)[0]
    
    # Step5: Apply inverse warping to img1
    # img1_warped, img1_topleft_coords = projectionImage(H, img1)
    
    # Step6: Merge the warped version of img1 with img2 under the same coordinate system of img2

    # stiched_img = mergeWarpedImages(img1_warped, img2, img1_topleft_coords)
    
    # the combination of the above do not work
    # this works
    stiched_img = combine_images(img1, img2, H)
    return stiched_img