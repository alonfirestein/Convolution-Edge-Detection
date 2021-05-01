# Convolution-Edge-Detection  
  
### Image Processing and Computer Vision Course Assignment 2:  


*In this assignment the following tasks were implemented using Python and the OpenCV library:*
- Implementing convolution on a 1D array
- Implementing convolution on a 2D array
- Performing image derivative
- Performing image blurring
- Different methods of Edge detection (Sobel, Canny, Zero Crossing...
- Hough Circles Transform
  
  
##### All tasks and functions were written in the ex2_utils.py file and they were all tested in the ex2_main.py file.
*NOTE:* It is recommended to run the functions one at a time in the main file and not all at once as it may take a while!  
  
  
## Image Outputs of the tasks listed above using the OpenCV and Matplotlib libraries:
  
  
## Performing Image Derivative:  
An image derivative is defined as the change in the pixel value of an image.  
  
![Derivative](https://user-images.githubusercontent.com/57404551/116781182-0e79bc80-aa8a-11eb-8333-c398aa133fa2.png)
  
  
## Image Blurring:  
Image blurring is achieved by convolving the image with a low-pass filter kernel. It is useful for removing noise. It actually removes high frequency content (eg: noise, edges) from the image. So edges are blurred a little bit in this operation (there are also blurring techniques which don't blur the edges).  
  
In the image below, we can see the difference between the original photo, to the blurred image using the OpenCV library, and the blurred image using my implementation:  
  
![blur](https://user-images.githubusercontent.com/57404551/116781420-62d16c00-aa8b-11eb-96c8-a5f5794add5d.png)

  
    
    
## Edge Detection:  
In edge detection, we find the boundaries or edges of objects in an image, by determining where the brightness of the image changes dramatically. Edge detection can be used to extract the structure of objects in an image. If we are interested in the number, size, shape, or relative location of objects in an image, edge detection allows us to focus on the parts of the image most helpful, while ignoring parts of the image that will not help us.  
  
In this assignment I implemented 3 different types of edge detectors:  
- Using the Sobel Operator  
- Zero Crossing Detector Using the Laplacian of Gaussian (LoG) Filter
- Using the Canny Edge Detection method
  
Below are images of the results of my edge detection implementations, using different images as well:  
  
##### Sobel:  
![sobel_monkey](https://user-images.githubusercontent.com/57404551/116781533-24887c80-aa8c-11eb-886d-8a3835ebdd09.png)  
  
##### Zero Crossing (LoG):  
![zeroCross](https://user-images.githubusercontent.com/57404551/116781576-65809100-aa8c-11eb-9ea0-7fabbb5f77ba.png)  
  
##### Canny:  
![canny](https://user-images.githubusercontent.com/57404551/116781585-77623400-aa8c-11eb-9009-c6c6afc73de5.png)  
  
  
  
## Hough Circles Transform:  
The circle Hough Transform (CHT) is a basic feature extraction technique used in digital image processing for detecting circles in imperfect images. The circle candidates are produced by “voting” in the Hough parameter space and then selecting local maxima in an accumulator matrix. 
  
I used an image containing coins to find the circles in the image, unfortunatley I didn't have enough time to finish it quite as I wanted, I'm planning on returning to it to fix the issue that the function marks circles multiple times even though they were already marked in the voting process. Orginally it is supposed to be marked only once.  
  
    
![circles](https://user-images.githubusercontent.com/57404551/116781596-88ab4080-aa8c-11eb-905f-24cfc7997134.png)  
    

