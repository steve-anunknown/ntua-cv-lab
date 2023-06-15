import cv2
import numpy as np

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
    output[-y_min:img2.shape[0]-y_min, -x_min:img2.shape[1]-x_min] = img2
    
    return output


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
    
    # Step6: Merge the warped version of img1 with img2 under the same coordinate system of img2
    stiched_img = combine_images(img1, img2, H)

    return stiched_img