# stars_tracker
## Image Processing and Similar Triangles
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


Image Processing
The image processing pipeline consists of the following steps:

star tracker
Algorithm description:

An algorithm for solving the problem of matching the stars between two images and finding the transition matrix between them

Step A: Image processing, noise filtering and finding the coordinates of the constellation in the two images

Step B: Finding a group T1,T2 triangles whose vertices are the stars for each image.

Step C: Finding the group of similar triangles from T1,T2 according to sides.

Step D: Performing a linear transformation between two similar triangles t1,t2, in 6 points and finding the common stars

Step E: Repeat step D until finding the best transformation

https://benedikt-bitterli.me/astro/


![image](https://user-images.githubusercontent.com/92825016/235878094-5249e9c6-4c30-4fa8-a24e-28343926f917.png)

