import cv2
import matplotlib.pyplot as plt
import numpy as np

# ================= BEG FUNCTIONS ================= #
def getpsnr(image, noisestd):
    return 20*log10((np.max(image)-np.min(image))/noisestd)

def getstd(image, psnr):
    return (np.max(image)-np.min(image))/(10**(psnr/20))

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
    if (method == "gaussian"):
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
        return kernel
    if (method == "log"):
        n = int(np.ceil(3*sigma)*2 + 1)
        x, y = np.meshgrid(np.arange(-n//2+1, n//2+1),
                np.arange(-n//2+1, n//2+1))
        kernel = np.exp(-(x**2+y**2)/(2*sigma**2))
        # this isn't really necessary, it preserves brightness
        kernel = kernel/np.sum(kernel) 
        laplacian = np.array([[0,1,0],
            [1,-4,1],
            [0,1,0]])
        # perform the convolution between the gaussian kernel
        # and the laplacian, in order to create the log kernel
        logkernel = my2dconv(kernel, laplacian)
        return logkernel
    print("Error: method has to be either \"gaussian\" or \"log\"")

def EdgeDetect(image, sigma, theta, method):
    if (method == "linear"):
        # construct the laplacian of gaussian kernel
        # and use it to filter the image
        logfilter = myfilter(sigma, "log")
        imgloged = cv2.filter2D(image, -1, logfilter)

        L = imgloged
        X = (L >= 0).astype(np.uint8)
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        Y = (cv2.dilate(X, cross)) - (cv2.erode(X, cross))

        gaussf = myfilter(sigma, "gaussian")
        smooth = cv2.filter2D(image, -1, gaussf)
        gradx, grady = np.gradient(smooth)
        grad = np.abs(gradx + 1j * grady)
        #gradx = cv2.Sobel(img_gaussed, cv2.CV_64F, 1, 0)
        #grady = cv2.Sobel(img_gaussed, cv2.CV_64F, 0, 1)
        #grad = np.abs(gradx + 1j * grady)
        D = ((Y == 1) & (grad > (theta * np.max(grad))))

        return D
    elif (method == "nonlinear"):
        # construct a gaussian kernel and use it
        # to smoothen the image. Then, perform
        # morphological operations on the smoothed image
        gaussf = myfilter(sigma, "gaussian")
        smooth = cv2.filter2D(image, -1, gaussf)
       
        # USED FOR DEBUGGING
        fig, axs = plt.subplots(1,1)
        axs.imshow(smooth, cmap='gray')
        axs.set_title("Smoothed Image")
        plt.show(block=False)
        plt.pause(0.01)
        
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        imgloged = cv2.dilate(smooth, cross) + cv2.erode(smooth, cross) - 2*smooth

        L = imgloged
        # USED FOR DEBUGGING
        print(np.min(L), np.max(L))
        X = (L >= 0).astype(np.uint8)
        print(np.min(X), np.max(X))

        Y = (cv2.dilate(X, cross)) - (cv2.erode(X, cross))

        # USED FOR DEBUGGING
        fig, axs = plt.subplots(1,1)
        axs.imshow(L, cmap='gray')
        axs.set_title("Almost Done Image")
        plt.show(block=False)
        plt.pause(0.01)

        gradx, grady = np.gradient(smooth)
        grad = np.abs(gradx + 1j * grady)
        #gradx = cv2.Sobel(img_gaussed, cv2.CV_64F, 1, 0)
        #grady = cv2.Sobel(img_gaussed, cv2.CV_64F, 0, 1)
        #grad = np.abs(gradx + 1j * grady)
        D = ((Y == 1) & (grad > (theta * np.max(grad))))

        return D
    else:
        print("Error: method has to be either \"linear\" or \"nonlinear\"")

def qualitycriterion(real, computed):
    T = real
    D = computed
    DT = (D & T)

    cardT = T.sum()
    cardD = D.sum()
    cardDT = DT.sum()

    prTD = cardDT/cardT
    prDT = cardDT/cardD

    C = (prDT + prTD)/2
    return C

# ================= END FUNCTIONS ================= #

# read the image, convert to gray scale and normalize it
image = cv2.imread("cv23_lab1_part12_material/edgetest_23.png", cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float)/image.max()

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
kyoto = kyoto.astype(np.float)/kyoto.max()

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

plt.show()



def BoxCriterion(ii, sigma):
    lxx, lxy, lyy = BoxDerivative(ii, sigma)
    r = lxx*lyy - (0.9*lxy)**2
    return r

def BoxFilters(ii, sigma, theta):
    r = BoxCriterion(ii, sigma)
    indices = InterestPointCoord(r, sigma, theta)
    scale = sigma*np.ones((indices.shape[0], 1))
    blobs = np.concatenate((indices, scale), axis=1)
    return blobs

def BoxFiltersLaplacian(image, sigma, theta, scale, N):
    # Multiscale Blob Detection
    scales = [scale**i for i in list(range(N))]
    sigmas = [scale*sigma for scale in scales]
    ii = IntegralImage(image)
    blobs_per_scale = [BoxFilters(ii, s, theta) for s in sigmas]
    return LogMetric(image, sigmas, blobs_per_scale, scales)



def LogMetric2(image, sigmas, itemsperscale, scales):
    # log((x,y), s) = (s^2)|Lxx((x,y),s) + Lyy((x,y),s)|
    # returns the coordinates of the points that maximize
    # the log metric in a neighborhood of 3 scales
    # (prev scale), (curr scale), (next scale)
    def img_grad2(img,s):
        Gs = myfilter(s, "gaussian")
        smooth = cv2.filter2D(img, -1, Gs)
        gradx,  grady = np.gradient(smooth)
        gradxx, gradxy = np.gradient(gradx)
        _ , gradyy = np.gradient(grady)
        return (gradxx, gradxy, gradyy)
    N = len(sigmas)
    gradsxx = [img_grad2(image, s)[0] for s in sigmas]
    gradsyy = [img_grad2(image, s)[2] for s in sigmas]
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
            prev = logp[x,y]
            curr = logc[x,y]
            next = logn[x,y]
            if (curr >= prev) and (curr >= next):
                final.append(triplet)
    return np.array(final)



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
    temp = j1 + j3 
    lplus = 1/2*(temp + np.sqrt( (j1 - j3)**2 + 4*j2**2))
    # trick not to recalculate nor store the square root
    lminus = temp - lplus 
    # calculate the cornerness criterion
    r = lplus * lminus - k*((lplus + lminus)**2)
    return r


def BlobDetection(image, sigma, theta):
    # calculate the blobness criterion
    # keep in mind that the image is smoothed
    # by default in the BlobCriterion function
    def BlobCriterion(image, sigma):
        def Hessian(image, sigma):
            # this is the same as the img_grad2 function
            # but it returns an array, not a triplet
            Gs = myfilter(sigma, "gaussian")
            smooth = cv2.filter2D(image, -1, Gs)
            gradx, grady = np.gradient(smooth)
            gradxx, gradxy = np.gradient(gradx)
            gradxy, gradyy = np.gradient(grady)
            return np.array([[gradxx, gradxy],
                [gradxy, gradyy]])
        H = Hessian(image, sigma)
        return H[0,0] * H[1,1] - H[0,1] * H[1,0]
    r = BlobCriterion(image, sigma)
    indices = InterestPointCoord(r, sigma, theta)
    scale = sigma*np.ones((indices.shape[0], 1))
    blobs = np.concatenate((indices, scale), axis=1)
    return blobs