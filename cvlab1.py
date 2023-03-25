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

# ================= END FUNCTIONS ================= #

# read the image and convert to gray scale
image = cv2.imread("cv23_lab1_part12_material/edgetest_23.png", cv2.IMREAD_GRAYSCALE)

# add noise to the images. 10db and 20db
image10db = image + np.random.normal(0, getstd(image, 10), image.shape)
image20db = image + np.random.normal(0, getstd(image, 20), image.shape)

noised_images = [image10db, image20db]
for nsd_img in noised_images:
    # plot the images
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image, cmap='gray')
    axs[1].imshow(nsd_img, cmap='gray')
    axs[0].set_title("Original Image")
    axs[1].set_title("Image with gaussian noise")
    plt.show()

    # It can be seen that, indeed, the 10db PSNR gaussian noise
    # is more intense than the 20db PSNR one.
    sigma = 1.5 # 1.5 for 10db, 3 for 20db
    gaussianfilter = myfilter(sigma, "gaussian")
    img_gaussed = cv2.filter2D(nsd_img, -1, gaussianfilter)

    # laplacian on gaussian
    logfilter1 = myfilter(sigma, "log")
    img_loged1 = cv2.filter2D(nsd_img, -1, logfilter1)

    # non linear approximation of laplacian on gaussian
    # using morphological operators
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    img_loged2 = cv2.dilate(img_gaussed, cross) + cv2.erode(img_gaussed, cross) - 2*img_gaussed

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img_gaussed, cmap='gray')
    axs[0].set_title("Gaussianly Smoothed Image")
    axs[1].imshow(img_loged1, cmap='gray')
    axs[1].set_title("LoGed Image (Linear Approximation)")
    axs[2].imshow(img_loged2, cmap='gray')
    axs[2].set_title("LoGed Image (Non Linear Approximation)")
    plt.show()
