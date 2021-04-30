from ex2_utils import *
import matplotlib.pyplot as plt
import numpy as np
import cv2

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def conv1Demo():

    signals = [np.array([1, 2, 3, 4]),
               np.array([0, 0, 0]),
               np.array([1, 3, 5, 4, 5, 3])]

    kernels = [np.array([0, -1, 0.5, -4]),
               np.array([0, 0, 0, 0, 0]),
               np.array([1/2, 1/3, 1/4])]

    correct_counter = 0
    for i in range(3):
        for j in range(3):
            np_conv = np.convolve(signals[i], kernels[j])
            my_conv = conv1D(signals[i], kernels[j])
            if np_conv.all() == my_conv.all():
                correct_counter += 1

    if correct_counter == len(signals)*len(kernels):
        print("All conv1Demo() tests were ran and passed! :)")
    else:
        print("Not all conv1Demo() tests passed! :(")


def conv2Demo():
    img = imReadAndConvert('boxman.jpeg', LOAD_GRAY_SCALE)
    kernels = [np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
               np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
               np.array([[-4.5, 2.5], [3.5, 5.5]]),
               np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8],
                         [7, 8, 9, 10, 11], [10, 11, 12, 13, 14]])]

    kernels[1] = kernels[1]/kernels[1].sum()
    kernels[2] = kernels[2]/kernels[2].sum()
    kernels[3] = kernels[3]/kernels[3].sum()

    correct_counter = 0
    for kernel in kernels:
        np_conv = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        my_conv = conv2D(img, kernel)
        if np_conv.all() == my_conv.all():
            correct_counter += 1

    if correct_counter == len(kernels):
        print("All conv2Demo() tests were ran and passed! :)")
    else:
        print("Not all conv2Demo() tests passed! :(")


def derivDemo():
    img = imReadAndConvert("coins.jpeg", LOAD_GRAY_SCALE)
    direction, magnitude, x_der, y_der = convDerivative(img)
    plots = [direction, magnitude, x_der, y_der]
    titles = ["Direction", "Magnitude", "X Derivative", "Y Derivative"]
    fig = plt.figure(figsize=(8, 7))
    plt.gray()
    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i)
        ax.imshow(plots[i-1])
        ax.title.set_text(titles[i-1])

    plt.tight_layout
    plt.show()


def testSobel():

    img = imReadAndConvert('codeMonkey.jpeg', LOAD_GRAY_SCALE)
    original_monkey = imReadAndConvert('codeMonkey.jpeg', LOAD_RGB)
    cv_monkey, my_monkey = edgeDetectionSobel(img, thresh=0.1)
    fig, ax = plt.subplots(1, 3, figsize=(9, 5))
    titles = ['Original Image', 'CV2 Sobel', 'My Sobel']
    plots = [original_monkey, cv_monkey, my_monkey]
    for i in range(3):
        ax[i].set_title(titles[i])
        ax[i].imshow(plots[i], cmap='gray')
        plt.tight_layout
    plt.show()
    plt.show()


def zeroCrossing():
    img = imReadAndConvert("dog.jpeg", LOAD_GRAY_SCALE)
    result = edgeDetectionZeroCrossingLOG(img)
    plt.imshow(result, cmap='gray')
    plt.title("Laplacian of Gaussian Zero Crossing Edge Detection")
    plt.show()


def cannyEdge():
    img = cv2.imread('pool_balls.jpeg', cv2.IMREAD_GRAYSCALE)
    cv2_canny, my_canny = edgeDetectionCanny(img, 50, 100)
    fig, ax = plt.subplots(1, 3, figsize=(9, 5))
    titles = ['Original Image', 'CV2 Canny Edge Detection', 'My Canny Edge Detection']
    plots = [img, cv2_canny, my_canny]
    for i in range(3):
        ax[i].set_title(titles[i])
        ax[i].imshow(plots[i], cmap='gray')
        plt.tight_layout
    plt.show()


def edgeDemo():
    testSobel()
    zeroCrossing()
    cannyEdge()


def blurDemo():
    img = cv2.imread("dog.jpeg", cv2.IMREAD_GRAYSCALE)
    kernel_size = 20
    fig, ax = plt.subplots(1, 3, figsize=(9, 5))
    titles = ['Original Image', 'CV2 Blur', 'My Blur']
    plots = [img, blurImage2(img, kernel_size), blurImage1(img, kernel_size)]
    for i in range(3):
        ax[i].set_title(titles[i])
        ax[i].imshow(plots[i], cmap='gray')
        plt.tight_layout

    plt.show()


def houghDemo():
    img = cv2.imread("coins.jpeg", cv2.IMREAD_GRAYSCALE)
    circles = houghCircle(img, 30, 100)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    for x, y, radius in circles:
        circle_plot = plt.Circle((x, y), radius, color='r', fill=False)
        ax.add_artist(circle_plot)
    plt.title("Circle Hough Transform Implementation")
    plt.show()


def main():
    conv1Demo()   # FINISHED
    conv2Demo()   # FINISHED
    derivDemo()   # FINISHED
    blurDemo()    # FINISHED
    edgeDemo()    # FINISHED
    houghDemo()   # FINISHED


if __name__ == '__main__':
    main()
