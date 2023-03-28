import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv23_lab1_part2_utils import interest_points_visualization
from cv23_lab1_part2_utils import disk_strel

# ================= BEG FUNCTIONS ================= #
def getpsnr(image, noisestd):
    return 20*log10((np.max(image)-np.min(image))/noisestd)

def getstd(image, psnr):
    return (np.max(image)-np.min(image))/(10**(psnr/20))

# this function seems to be unused
def gaussian2d(x, y, x0, y0, sigmax, sigmay, a):
    return a*np.exp(-((x-x0)**2/(2*sigmax**2)+(y-y0)**2/(2*sigmay**2)))

def my2dconv(image, kernel):
    ix, iy = image.shape
    nx, ny = kernel.shape
    result = np.zeros((ix + nx - 1, iy + ny - 1))
    padded = np.pad(image, [(nx//2, nx//2),(ny//2,ny//2)], mode='constant')
    for i in range(nx//2, ix + nx//2):
        for j in range(ny//2, iy + ny//2):
            result[i,j] = np.sum(padded[i-nx//2:i+nx//2+1,j-ny//2:j+ny//2+1] * kernel)
    return result[nx//2:ix+nx//2, ny//2:iy+ny//2]

def myfilter(sigma, method):
    if (not (method == "gaussian" or method == "log")):
        print("Error: method has to be either \"gaussian\" or \"log\"")
        exit(2)
    n = int(np.ceil(3*sigma)*2 + 1)
    # generating the kernels using meshgrid is
    # said to be more accurate than multiplying
    # two 1d gaussian kernels together. That is
    # because the product of two gaussian functions
    # is not neccessarily a gaussian function, so
    # there may be a loss of symmetry between the
    # x, y axis.
    x, y = np.meshgrid(np.arange(-n//2+1, n//2+1),
            np.arange(-n//2+1, n//2+1))
    kernel = np.exp(-(x**2+y**2)/(2*sigma**2))
    # this isn't really necessary, it preserves brightness
    kernel = kernel/np.sum(kernel) 
    if (method == "gaussian"):
        return kernel
    laplacian = np.array([[0,1,0],
        [1,-4,1],
        [0,1,0]])
    # perform the convolution between the gaussian kernel
    # and the laplacian, in order to create the log kernel
    logkernel = my2dconv(kernel, laplacian)
    return logkernel

def EdgeDetect(image, sigma, theta, method):
    if (not (method == "linear" or method == "nonlinear")):
        print("Error: method has to be either \"linear\" or \"nonlinear\"")
        exit(2)
    gaussf = myfilter(sigma, "gaussian")
    smooth = cv2.filter2D(image, -1, gaussf)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    if (method == "linear"):
        # construct the laplacian of gaussian kernel
        # and use it to filter the image
        logfilter = myfilter(sigma, "log")
        imgloged = cv2.filter2D(image, -1, logfilter)
    elif (method == "nonlinear"):
        # Perform morphological operations using
        # the cross structuring element
        imgloged = cv2.dilate(smooth, cross) + cv2.erode(smooth, cross) - 2*smooth
    # the imgloged variable is visible only if one of the if blocks
    # get executed. hopefully, this always happens
    L = imgloged
    # type uin8 is needed for compatibility with the
    # dilate and erode functions. otherwise, the matrix's
    # elements would have boolean type.
    X = (L > 0).astype(np.uint8)
    Y = (cv2.dilate(X, cross)) - (cv2.erode(X, cross))

    gradx, grady = np.gradient(smooth)
    grad = np.abs(gradx + 1j * grady)
    D = ((Y == 1) & (grad > (theta * np.max(grad))))
    return D

def qualitycriterion(real, computed):
    # use the following names for compatibility
    # with the project's guide.
    T = real
    D = computed
    DT = (D & T)
    # the matrices are supposed to be boolean
    # therefore the sum() functions counts the
    # elements that are true / 1.
    cardT = T.sum()
    cardD = D.sum()
    cardDT = DT.sum()

    prTD = cardDT/cardT
    prDT = cardDT/cardD

    C = (prDT + prTD)/2
    return C

def CornerCriterion(image, sigma, rho, k):
    # define the filters according to the arguments
    Gs = myfilter(sigma, "gaussian")
    Gr = myfilter(rho, "gaussian")
    # smoothen the image
    smooth = cv2.filter2D(image, -1, Gs)
    # calculate the gradient on both directions 
    gradx, grady = np.gradient(smooth)
    # calculate whatevere these elements are 
    j1 = cv2.filter2D(gradx * gradx, -1, Gr)
    j2 = cv2.filter2D(gradx * grady, -1, Gr)
    j3 = cv2.filter2D(grady * grady, -1, Gr)
    # calculate the eigenvalues of J = [j1 j2 j3]
    # lplus = 1/2*(j1 + j3 + np.sqrt( (j1 - j3)**2 + 4*j2**2))
    # lminus = 1/2*(j1 + j3 - np.sqrt( (j1 - j3)**2 + 4*j2**2))
    # store the sum so as not to recalculate it
    temp = j1 + j3 
    lplus = 1/2*(temp + np.sqrt( (j1 - j3)**2 + 4*j2**2))
    # trick not to recalculate nor store the square root
    lminus = temp - lplus 
    # calculate the cornerness criterion
    r = lplus * lminus - k*((lplus + lminus)**2)
    return r

def CornerDetection(image, sigma, rho, theta, k):
    # Keep in mind that the image is smoothed
    # by default in the criterion function.
    r = CornerCriterion(image, sigma, rho, k)
    # evaluate the following 2 conditions 
    # condition 1
    ns = np.ceil(3*sigma)*2 + 1
    bsq = disk_strel(ns)
    cond1 = ( r == cv2.dilate(r, bsq) )
    # condition 2
    maxr = np.max(r)
    cond2 = ( r > theta * maxr )
    # choose the pixels that satisfy both of them
    # return their coordinates and their scale
    x, y = np.where(cond1 & cond2)
    # for compatibility with the utility function
    # provided by the lab staff, the y coordinate
    # has to come before the x coordinate
    indices = np.column_stack((y,x))
    scale = sigma*np.ones((indices.shape[0], 1))
    corners = np.concatenate((indices, scale), axis=1)
    return corners

def img_grad2(img,s):
    Gs = myfilter(s, "gaussian")
    smooth = cv2.filter2D(img, -1, Gs)
    gradx,  grady = np.gradient(img)
    gradxx, gradxy = np.gradient(gradx)
    gradyx, gradyy = np.gradient(grady)
    return (gradxx, gradxy, gradyy)

def logmetric(image, params, itemsperscale, scales):
    # log((x,y), s) = (s^2)|Lxx((x,y),s) + Lyy((x,y),s)|
    N = len(params)
    gradsxx = [img_grad2(image, s)[0] for (s, _) in params]
    gradsyy = [img_grad2(image, s)[2] for (s, _) in params]
    grads = list(zip(scales, gradsxx, gradsyy))
    logs = [(s**2)*np.abs(xx + yy) for (s, xx, yy) in grads]
    # now we iterate through the points and compare each scale
    # with its previous and its next. if the log metric is not
    # maximized, we reject it. 
    final = []
    for index, items in enumerate(itemsperscale):
        logp = logs[max(index-1,0)]
        logc = logs[index]
        logn = logs[min(index+1,N-1)] 
        for triplet in items:
            x = int(triplet[1])
            y = int(triplet[0])
            prev = logp[x][y]
            curr = logc[x][y]
            next = logn[x][y]
            if (curr >= prev) and (curr >= next):
                final.append(triplet)
    return np.array(final)

def HarrisLaplacian(image, sigma, rho, theta, k, scale, N):
    # Perhaps the code can be a little cleaner
    # construct the lists of sigma and rho parameters
    # using the given scales and zip them into a comfy list.
    scales = [scale**i for i in list(range(N))]
    sigmas = [scale * sigma for scale in scales]
    rhos = [scale * rho for scale in scales]
    params = list(zip(sigmas, rhos))
    # call the edge detection function for each pair of parameters
    # if the image is MxM, then the resulting array is
    # M x (3*N), because the CornerDetection method returns a M x 3 array
    # and the iterations happens N times, once for every scale.
    corners_per_scale = [CornerDetection(image, s, r, theta, k) for (s, r) in params]
    # now we calculate the LoG for the pixels of every scale
    return logmetric(image, params, corners_per_scale, scales)

def Hessian(image, sigma):
    Gs = myfilter(sigma, "gaussian")
    smooth = cv2.filter2D(image, -1, Gs)
    gradx, grady = np.gradient(smooth)
    gradxx, gradxy = np.gradient(gradx)
    gradxy, gradyy = np.gradient(grady)
    return np.array([[gradxx, gradxy],
        [gradxy, gradyy]])
def BlobCriterion(image, sigma):
    H = Hessian(image, sigma)
    return H[0,0] * H[1,1] - H[0,1] * H[1,0]

def BlobDetection(image, sigma, theta):
    # calculate the blobness criterion
    # keep in mind that the image is smoothed
    # by default in the BlobCriterion function
    r = BlobCriterion(image, sigma) 
    # evaluate the following 2 conditions 
    # condition 1
    ns = np.ceil(3*sigma)*2 + 1
    bsq = disk_strel(ns)
    cond1 = ( r == cv2.dilate(r, bsq) )
    # condition 2
    maxr = np.max(r)
    cond2 = ( r > theta * maxr )
    x, y = np.where(cond1 & cond2)
    # for compatibility with the utility function
    # provided by the lab staff, the y coordinate
    # has to come before the x coordinate
    indices = np.column_stack((y,x))
    scale = sigma*np.ones((indices.shape[0], 1))
    blobs = np.concatenate((indices, scale), axis=1)
    return blobs

# this is terribly inefficient. the gradients are computed
# many more times than needed.

def HessianLaplacian(image, sigma, rho, scale, N):
    # construct the lists of sigma and rho parameters
    # using the given scales and zip them into a comfy list.
    scales = [scale**i for i in list(range(N))]
    sigmas = [scale * sigma for scale in scales]
    rhos = [scale * rho for scale in scales]
    params = list(zip(sigmas, rhos))
    # call the blob detection function for each pair of parameters
    # if the image is MxM, then the resulting array is
    # M x (3*N), because the BlobDetection method returns a M x 3 array
    # and the iterations happens N times, once for every scale.
    blobs_per_scale = [BlobDetection(image, s, r) for (s, r) in params]
    # now we calculate the LoG for the pixels of every scale
    # log((x,y), s) = (s^2)|Lxx((x,y),s) + Lyy((x,y),s)|
    return logmetric(image, params, blobs_per_scale, scales)


# ================= END FUNCTIONS ================= #

# read the image, convert to gray scale and normalize it
image = cv2.imread("cv23_lab1_part12_material/edgetest_23.png", cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float64)/image.max()

fig, axs = plt.subplots(1,1)
axs.imshow(image, cmap='gray')
axs.set_title("Original Image")
plt.show(block=False)
plt.pause(0.01)

# add noise to the images. 10db and 20db
# in the psnr context, less dBs equals more noise
image10db = image + np.random.normal(0, getstd(image, 10), image.shape)
image20db = image + np.random.normal(0, getstd(image, 20), image.shape)

# Play around with sigma and theta in order to
# obtain the best results. 
noised_images = [image10db, image20db]
sigma = [1.5, 3]
theta = [0.2, 0.2]
thetareal = 0.01

for index, img in enumerate(noised_images):
    N1 = EdgeDetect(img, sigma[index], theta[index], "linear")
    N2 = EdgeDetect(img, sigma[index], theta[index], "nonlinear")

    # the non linear method gives the best results,
    # therefore we name it D and continue our evaluation
    D = N2
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    M = cv2.dilate(image, cross) - cv2.erode(image, cross)
    T = ( M > thetareal ).astype(np.uint8)
    print(T.shape)

    fig, axs = plt.subplots(2,2)
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title("Noised Image")
    axs[0, 1].imshow(N1, cmap='gray')
    axs[0, 1].set_title("Linear edge detection")
    axs[1, 0].imshow(N2, cmap='gray')
    axs[1, 0].set_title("Non linear edge detection")
    axs[1, 1].imshow(T, cmap='gray')
    axs[1, 1].set_title("Actual Edges")
    plt.show(block=False)
    plt.pause(0.01)

    C = qualitycriterion(T, D)
    print(f"The quality criterion is C[{index}] = {C}")


# ================= BEG REAL IMAGE PROCESSING ================= #

# read image and normalize it
kyoto = cv2.imread("cv23_lab1_part12_material/kyoto_edges.jpg", cv2.IMREAD_GRAYSCALE)
kyoto = kyoto.astype(np.float64)/kyoto.max()

# play around with sigma and theta
# big sigma => much smoothing => not fine details
# big theta => less edges, small theta => many edges

sigma = 1.5
theta = 0.2
thetareal = 0.14

N1 = EdgeDetect(kyoto, sigma, theta, "linear")
N2 = EdgeDetect(kyoto, sigma, theta, "nonlinear")

# the non linear method gives the best results,
# therefore we name it D and continue our evaluation
D = N2
cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
M = cv2.dilate(kyoto, cross) - cv2.erode(kyoto, cross)
T = ( M > thetareal ).astype(np.uint8)

fig, axs = plt.subplots(2,2)
axs[0, 0].imshow(kyoto, cmap='gray')
axs[0, 0].set_title("Noised Image")
axs[0, 1].imshow(T, cmap='gray')
axs[0, 1].set_title("Actual Edges")
axs[1, 0].imshow(N1, cmap='gray')
axs[1, 0].set_title("Linear edge detection")
axs[1, 1].imshow(N2, cmap='gray')
axs[1, 1].set_title("Non Linear edge detection")
plt.show(block=False)
plt.pause(0.01)

C = qualitycriterion(T, D)
print(f"The quality criterion is C[{index}] = {C}")

# ================= END REAL IMAGE PROCESSING ================= #
# =================   BEG CORNERS DETECTION   ================= #

kyoto = cv2.imread("cv23_lab1_part12_material/kyoto_edges.jpg")
kyoto = kyoto.astype(np.float64)/kyoto.max()
gray = cv2.imread("cv23_lab1_part12_material/kyoto_edges.jpg", cv2.IMREAD_GRAYSCALE)
# play around with the parameters
corners = CornerDetection(gray, 1.2, 2.5, 0.09, 0.1)
interest_points_visualization(kyoto, corners, None)
# play around with the parameters
corners = HarrisLaplacian(gray, 1.2, 2.5, 0.05, 0.1, 1.1, 8)
interest_points_visualization(kyoto, corners, None)

# =================   END CORNERS DETECTION   ================= #
# =================    BEG BLOB DETECTION     ================= #

up = cv2.imread("cv23_lab1_part12_material/up.png")
up = up.astype(np.float64)/up.max()
gray = cv2.imread("cv23_lab1_part12_material/up.png", cv2.IMREAD_GRAYSCALE)
# play around with the parameters
blobs = BlobDetection(gray, 2.5, 0.25)
interest_points_visualization(up, blobs, None)
# play around with the parameters
blobs = HessianLaplacian(gray, 1.5, 0.05, 1.1, 8)
interest_points_visualization(up, blobs, None)

# =================    END BLOB DETECTION     ================= #

plt.show()



