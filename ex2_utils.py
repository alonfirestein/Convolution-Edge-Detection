import numpy as np
import cv2 as cv

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2



def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE(1) or RGB(2)
    :return: The image object
    """
    # Loading an image and converting it according the the representation input
    img = cv.imread(filename)
    if img is not None:
        if representation == LOAD_GRAY_SCALE:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        elif representation == LOAD_RGB:
            # We weren't asked to convert a grayscale image to RGB so this will suffice
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        else:  # Any other value was entered as the second parameter
            raise ValueError("Please enter [1] for Grayscale, or [2] for RGB representation of the image.")
    else:
        raise Exception("Could not read the image! Please try again.")
    return img / 255.0


def conv1D(inSignal:np.ndarray, kernel1:np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """
    # Getting size of signal/image and kernel
    signal_len, kernel_len = np.size(inSignal), np.size(kernel1)
    # Size of output vector is: signal_len + kernel_len - 1
    conv_arr = np.zeros(signal_len + kernel_len - 1)
    for i in np.arange(signal_len):
        for j in np.arange(kernel_len):
            conv_arr[i+j] = conv_arr[i+j] + (inSignal[i] * kernel1[j])

    return conv_arr


def conv2D(inImage:np.ndarray, kernel2:np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """
    k_height, k_width = kernel2.shape
    img_height, img_width = inImage.shape
    padded_mat = np.pad(inImage, ((k_height, k_height), (k_width, k_width)), 'mean')
    convolved_mat = np.zeros((img_height, img_width))
    for i in range(img_height):
        for j in range(img_width):
            x_head = j + 1 + k_width
            y_head = i + 1 + k_height
            convolved_mat[i, j] = (padded_mat[y_head:y_head + k_height, x_head:x_head + k_width] * kernel2).sum()

    return convolved_mat


def convDerivative(inImage:np.ndarray) -> (np.ndarray, np.ndarray,  np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """
    kernel = np.array([[1, 0, -1]])
    kernel_transposed = kernel.T

    x_der = conv2D(inImage, kernel)
    y_der = conv2D(inImage, kernel_transposed)

    # Calculating magnitude => sqrt(iX**2 + iY**2)
    mag = np.sqrt((np.power(x_der, 2) + np.power(y_der, 2)))

    # Basic rule of math: tan^-1(x) == arctan(x)
    direction = np.arctan2(y_der, x_der)

    return direction, mag, x_der, y_der


def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    # Sigma of kernel i,j = 1.0
    sigma = 1.0
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            ker_diff = np.sqrt(np.power(i-center, 2) + np.power(j-center, 2))
            kernel[i, j] = np.exp(-(np.power(ker_diff, 2)) / (2 * np.power(center, 2)))

    gaussian_kernel = kernel/sigma
    blur = conv2D(in_image, gaussian_kernel)
    return blur


def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    # NOTE: Use gaussian kernel and filter2D functions only AND NOT blur/GaussianBlur function
    # Creating a Gaussian kernel using the OpenCV library
    gaussian_kernel = cv.getGaussianKernel(kernel_size, -1)
    # Applying the Gaussian kernel to the image
    blurred_img = cv.sepFilter2D(in_image, -1, gaussian_kernel, gaussian_kernel)
    return blurred_img


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    # CV implementation
    cv_sobel_x = cv.Sobel(img, -1, 1, 0, ksize=5)
    cv_sobel_y = cv.Sobel(img, -1, 0, 1, ksize=5)
    cv_sobel_magnitude = np.sqrt(np.square(cv_sobel_x) + np.square(cv_sobel_y))
    cv_sobel = np.zeros(cv_sobel_magnitude.shape)
    cv_sobel[cv_sobel_magnitude > thresh] = 1

    # My implementation:
    # More info taken from here: https://en.wikipedia.org/wiki/Sobel_operator
    ker_sobel_x = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])

    ker_sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

    sobel_x = conv2D(img, np.flip(ker_sobel_x))
    sobel_y = conv2D(img, np.flip(ker_sobel_y))
    my_sobel_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    my_sobel = np.zeros(my_sobel_magnitude.shape)
    my_sobel[my_sobel_magnitude > thresh] = 1

    return cv_sobel, my_sobel


def edgeDetectionZeroCrossingLOG(img:np.ndarray)->(np.ndarray):
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    # Smoothing the image with 2D Gaussian
    blur = cv.GaussianBlur(img, (3, 3), 0)
    # Applying the Laplacian filter
    img = cv.Laplacian(blur, cv.CV_64F)
    img_crossing = img / img.max()
    zero_crossing_img = np.zeros(img.shape)
    # Initializing the pixel value counter (positive/negative values)
    neg_pixel_count = 0
    pos_pixel_count = 0
    img_height, img_width = img.shape
    # Looking for zero crossing patterns: such as {+,0,-} or {+,-}
    # Meaning we check the sign (positive or negative) of all the pixels around each pixel
    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            # 3x3 kernel
            pixel_neighbours = [img_crossing[i + 1, j - 1], img_crossing[i + 1, j],
                                img_crossing[i + 1, j + 1], img_crossing[i, j - 1],
                                img_crossing[i, j + 1],     img_crossing[i - 1, j - 1],
                                img_crossing[i - 1, j],     img_crossing[i - 1, j + 1]]

            for pixel_value in pixel_neighbours:
                if isPositive(pixel_value):
                    pos_pixel_count += 1
                elif not isPositive(pixel_value):
                    neg_pixel_count += 1

            # Checking if both the positive and negative value counts are positive,
            # then zero crossing potentially exists for that pixel
            zero_crossing = isPositive(pos_pixel_count) and isPositive(neg_pixel_count)

            # Finding the maximum neighbour pixel difference and changing the pixel value
            min_value_diff = img_crossing[i, j] + np.abs(min(pixel_neighbours))
            max_value_diff = np.abs(img_crossing[i, j]) + max(pixel_neighbours)
            if zero_crossing:
                if isPositive(img_crossing[i, j]):
                    zero_crossing_img[i, j] = min_value_diff
                elif not isPositive(img_crossing[i, j]):
                    zero_crossing_img[i, j] = max_value_diff

    return zero_crossing_img


def isPositive(value):
    return value > 0


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    # CV implementation:
    cv_canny = cv.Canny(img, thrs_1, thrs_2)

    # My implementation:
    # Extra info from here: https://en.wikipedia.org/wiki/Canny_edge_detector
    # Smoothing the image with a Gaussian
    img = cv.GaussianBlur(img, (3, 3), 0)
    # Getting the magnitude and direction of the gradient:
    cv_sobel_x, cv_sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, thrs_1), cv.Sobel(img, cv.CV_64F, 1, 0, thrs_2)
    magnitude = cv.magnitude(cv_sobel_x, cv_sobel_y)
    direction = np.arctan2(cv_sobel_x, cv_sobel_y) * 180 / np.pi

    # Performing non-maximum suppression(any gradient value that is not a local peak is set to zero):
    post_suppression = non_maximum_suppression(magnitude, direction)
    # Performing Hysteresis and finding false edges
    my_canny = hysteresis(post_suppression, thrs_2, thrs_1)

    return cv_canny, my_canny


def non_maximum_suppression(img: np.ndarray, direction):
    img_height, img_width = img.shape
    nms = np.zeros((img_height, img_width))
    angle = direction
    # Normalizing the angle values
    angle[angle < 0] += 180

    # We iterate through each pixel in the image matrix and for each pixel (x,y) we compare to pixels along its
    # gradient direction
    for x in range(1, img_height - 1):
        for y in range(1, img_width - 1):
            q = 255
            r = 255

            # 0 Degree Angle
            if (0 <= angle[x, y] < 22.5) or (157.5 <= angle[x, y] <= 180):
                q = img[x, y + 1]
                r = img[x, y - 1]

            # 45 Degree Angle
            elif 22.5 <= angle[x, y] < 67.5:
                q = img[x - 1, y - 1]
                r = img[x + 1, y + 1]
            # 90 Degree Angle
            elif 67.5 <= angle[x, y] < 112.5:
                q = img[x + 1, y]
                r = img[x - 1, y]

            # 135 Degree Angle
            elif 112.5 <= angle[x, y] < 157.5:
                q = img[x + 1, y - 1]
                r = img[x - 1, y + 1]

            # We check if the pixel is bigger than its neighbours, and if it is then we keep it,
            # otherwise we change the pixel value to zero.
            if (img[x, y] >= q) and (img[x, y] >= r):
                nms[x, y] = img[x, y]
            else:
                nms[x, y] = 0

    return nms


def hysteresis(img, low_threshold, high_threshold):
    img_height, img_width = img.shape
    weak = 75
    strong = 255

    # Any edge that is above high is a true edge => Then we keep it
    strong_i, strong_j = np.where(img >= high_threshold)

    # Any edge that is below low is a false edge => Then we remove it
    zeros_i, zeros_j = np.where(img < low_threshold)

    # For any edge pixel that is in between => Then we keep it only if it is connected to a strong edge
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    img = np.zeros((img_height, img_width))
    img[zeros_i, zeros_j] = 0
    img[strong_i, strong_j] = strong
    # We mark all the edges that are in between, and we will check if one of its neighbours is a strong edge
    img[weak_i, weak_j] = weak

    # Here we iterate and check each "weak" pixel if one of its neighbours is a strong intensity
    # meaning it's a strong edge, and if it is, therefore we set that pixel to strong.
    for i in range(1, img_height-1):
        for j in range(1, img_width-1):
            if img[i, j] == weak:
                pixel_neighbours = img[i-1:i+1, j-1:j+1]
                row, col = np.where(pixel_neighbours == strong)

                if len(row) > 0:
                    img[i, j] = strong
                else:
                    img[i, j] = 0

    return img


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension :param I: Input image
    :param minRadius: Minimum circle radius
    :param maxRadius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """
    circles = list()
    thresh = 0.7
    sobel_x = cv.Sobel(img, cv.CV_64F, 0, 1, thresh)
    sobel_y = cv.Sobel(img, cv.CV_64F, 1, 0, thresh)
    direction = np.radians(np.arctan2(sobel_x, sobel_y) * 180 / np.pi)
    accumulator = np.zeros((len(img), len(img[0]), max_radius+1))
    edges = cv.Canny(img, 0.1, 0.45)
    height = len(edges)
    width = len(edges[0])
    for x in range(0, height):
        for y in range(0, width):
            if edges[x][y] == 255:
                for radius in range(min_radius, max_radius + 1):
                    angle = direction[x, y] - np.pi / 2
                    # x1, y1 => value + radius
                    # x2, y2 => value - radius
                    x1, x2 = np.int32(x - radius * np.cos(angle)), np.int32(x + radius * np.cos(angle))
                    y1, y2 = np.int32(y + radius * np.sin(angle)), np.int32(y - radius * np.sin(angle))
                    if 0 < x1 < len(accumulator) and 0 < y1 < len(accumulator[0]):
                        accumulator[x1, y1, radius] += 1
                    if 0 < x2 < len(accumulator) and 0 < y2 < len(accumulator[0]):
                        accumulator[x2, y2, radius] += 1

    thresh = np.multiply(np.max(accumulator), 1/2)
    x, y, radius = np.where(accumulator >= thresh)
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 0 and radius[i] == 0:
            continue
        circles.append((y[i], x[i], radius[i]))

    return circles
