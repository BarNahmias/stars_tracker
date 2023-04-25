# stars_tracker
## Image Processing and RANSAC
This Python code applies various image processing techniques and the RANSAC algorithm to find the best line that fits a set of keypoints detected in an image.

Dependencies
The following libraries are required to run this code:

OpenCV (cv2)
NumPy
Random
Installation
To install the required dependencies, you can use the following command:


pip install opencv-python numpy
Usage
The code can be run from the command line by providing an image file as an argument. For example:

python main.py image.jpg
This will process the image and display the resulting image with keypoints and the best line found using RANSAC.

Image Processing
The image processing pipeline consists of the following steps:

Load the image and convert to grayscale.
Apply a Gaussian blur with a 5x5 kernel to reduce noise.
Apply Canny edge detection with threshold values of 100 and 200 to detect edges.
Detect keypoints using the SIFT algorithm with custom parameters.
Filter out keypoints that are too close to each other.
Remove keypoints that are too close to the edges of the image.
RANSAC Algorithm
The RANSAC algorithm is used to find the best line that fits a set of keypoints in the image. The algorithm consists of the following steps:

Select two random keypoints.
Compute the slope and intercept of the line connecting the two keypoints.
Find the inliers that lie within a certain range of the line and between the two keypoints.
Repeat steps 1-3 for a specified number of iterations.
Choose the line with the most inliers as the best line.
Parameters
The following parameters can be adjusted to customize the behavior of the code:

margin: The fraction of the image width and height to exclude from the edges when filtering keypoints. Default is 0.1 (10%).
min_distance: The minimum distance between keypoints to be considered separate. Default is 20 pixels.
sigma: The standard deviation of the Gaussian filter used in SIFT. Default is 1.6.
edgeThreshold: The minimum edge threshold value in SIFT. Default is 20.
contrastThreshold: The minimum contrast threshold value in SIFT. Default is 0.07.
nOctaveLayers: The number of octave layers in SIFT. Default is 15.
range_value: The maximum distance from a point to the line to be considered an inlier in RANSAC. Default is 5 pixels.
max_iterations: The maximum number of iterations to run RANSAC. Default is 1000.
inlier_count: The minimum number of inliers required for a line to be considered valid. Default is 10.
delta: The minimum change in the number of inliers required for a new line to be considered better than the previous best line. Default is 5.

![תמונה של WhatsApp‏ 2023-04-24 בשעה 20 47 10](https://user-images.githubusercontent.com/92825016/234222731-9350952e-ddb5-4939-8554-19819b2f2cb0.jpg)
