import cv2
import matplotlib.pyplot as plt
import numpy as np

# enable interactive mode
# plt.ion()

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
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        imgloged = cv2.dilate(smooth, cross) + cv2.erode(smooth, cross) - 2*smooth

        L = imgloged
        X = (L >= 0).astype(np.uint8)
        Y = (cv2.dilate(X, cross)) - (cv2.erode(X, cross))

        gradx, grady = np.gradient(smooth)
        grad = np.abs(gradx + 1j * grady)
        #gradx = cv2.Sobel(img_gaussed, cv2.CV_64F, 1, 0)
        #grady = cv2.Sobel(img_gaussed, cv2.CV_64F, 0, 1)
        #grad = np.abs(gradx + 1j * grady)
        D = ((Y == 1) & (grad > (theta * np.max(grad))))

        return D
    else:
        print("Error: method has to be either \"linear\" or \"nonlinear\"")


# ================= END FUNCTIONS ================= #

# read the image and convert to gray scale
image = cv2.imread("cv23_lab1_part12_material/edgetest_23.png", cv2.IMREAD_GRAYSCALE)

# add noise to the images. 10db and 20db
# in the psnr context, less dBs equals more noise
image10db = image + np.random.normal(0, getstd(image, 10), image.shape)
image20db = image + np.random.normal(0, getstd(image, 20), image.shape)

noised_images = [image10db, image20db]
sigma = [1.5, 3]
theta = [0.2, 0.2]

fig, axs = plt.subplots(1,1)
axs.imshow(image, cmap='gray')
axs.set_title("Original Image")
plt.show()


for index, img in enumerate(noised_images):
    N1 = EdgeDetect(img, sigma[index], theta[index], "linear")
    N2 = EdgeDetect(img, sigma[index], theta[index], "nonlinear")
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Noised Image")
    axs[1].imshow(N1, cmap='gray')
    axs[1].set_title("Linear edge detection")
    axs[2].imshow(N2, cmap='gray')
    axs[2].set_title("Non linear edge detection")
    plt.show()



