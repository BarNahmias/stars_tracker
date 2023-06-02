import random

import cv2
import numpy as np
import os
"""
https://benedikt-bitterli.me/astro/
"""


"""
The function receives an image and performs noise filtering and image processing
"""
def load_image(image):
    # Load image
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # Apply binary thresholding to convert to black and white
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # define structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # perform morphological opening to remove small spots
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening

"""
The function receives a processed image using the load_image function.
Outputs coordinates of stars in SIFT means.
 Creates patterns of triangles using three stars
"""
def create_triangle(image):
    img = cv2.imread(image)
    filtered_img = load_image(image)

    #The cv2.SIFT_create() function creates a SIFT object that can be used to detect keypoints
    # and extract descriptors from images using the SIFT algorithm.
    # The sigma parameter sets the scale of the difference of Gaussians used to detect keypoints

    sift = cv2.SIFT_create(sigma=0.9)
    kp = sift.detect(filtered_img, None)
    # sort the stars by response
    kp_filter = sorted(kp, key=lambda x: x.size, reverse=True)

    # Get keypoints coordinates as a NumPy array
    pts = np.array([k.pt for k in kp[:]], dtype=np.float32)

    # Compute Delaunay triangulation of the keypoints
    tri = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))
    tri.insert(pts)
    triangle_list = tri.getTriangleList()
    return triangle_list, kp, img

"""
The function receives a list of triangles
calculates their angles and returns a dictionary.
Key=angles , value=vertices
"""
def angle(triangle_list):
    # Create empty dictionary to store triangles by angles
    triangle_dict = {}
    # Loop through each triangle in the list
    for i in range(triangle_list.shape[0]):
        angle=[]
        # Get vertices of current triangle
        tri_pts = triangle_list[i].reshape(3, 2)

        # Get edges of triangle
        a = np.linalg.norm(tri_pts[0] - tri_pts[1])
        b = np.linalg.norm(tri_pts[1] - tri_pts[2])
        c = np.linalg.norm(tri_pts[2] - tri_pts[0])

        # Calculate angles using law of cosines
        alpha = np.degrees(np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
        beta = np.degrees(np.arccos((c ** 2 + a ** 2 - b ** 2) / (2 * c * a)))
        gamma = np.degrees(np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))
        angle.append(round(alpha,0))
        angle.append(round(beta,0))
        angle.append(round(gamma,0))

        # Add triangle to dictionary using angles as key
        triangle_dict[tuple(angle)] = tri_pts
    return triangle_dict

"""
Get two dictionaries of triangle vertices and its angles
returns the list of similar
"""
def get_shared_angle(map1, map2):
    common_keys = set(map1.keys()) & set(map2.keys())
    shared_vertices = []
    for key in common_keys:
        triangle1 = map1[key]
        triangle2 = map2[key]
        shared_vertices.append((triangle1,triangle2))
    return shared_vertices

"""
 This function takes in two triangles, triangle1 and triangle2, and a set of points points.
 It extracts the (x, y) coordinates from the cv2.KeyPoint objects in points, creates a transformation matrix using cv2.getAffineTransform(),
 applies this transformation matrix to the points using cv2.transform(), converts the transformed points back to cv2.KeyPoint objects,
 and returns these transformed cv2.KeyPoint objects.
"""

def transform_points(triangle1, triangle2, points):
    # Extract (x, y) coordinates from cv2.KeyPoint objects
    points = [(kp.pt[0], kp.pt[1]) for kp in points]

    # Create transformation matrix
    M = cv2.getAffineTransform(triangle1, triangle2)

    # Apply transformation to points
    transformed_points = cv2.transform(np.array([points], dtype='float32'), M)
    transformed_points = transformed_points[0].tolist()

    # Convert back to cv2.KeyPoint objects
    transformed_kps = [cv2.KeyPoint(x, y, 1) for (x, y) in transformed_points]

    return transformed_kps

"""
This function find_shared_keypoints takes two lists of keypoints (keypoints1 and keypoints2)
and a delta_range as input,
and returns a list of keypoints that are close to each other within the specified delta_range
"""
def find_shared_keypoints(keypoints1, keypoints2, delta_range):
    shared_keypoints = []
    for kp1 in keypoints1:
        for kp2 in keypoints2:
            delta = np.sqrt((kp1.pt[0] - kp2.pt[0]) ** 2 + (kp1.pt[1] - kp2.pt[1]) ** 2)
            if delta <= delta_range:
                shared_keypoints.append(kp1)
                break
    return shared_keypoints

"""
The function goes over all similar triangles and performs a transformation.
Returns the transformation that gives the most common starsdef"
"""
def best_tra_transform(similar_triangle, points1,points2,delta):
    transform = []
    original = []
    shared_keypoint = []
    max = 500000
    if(max>len(similar_triangle)):
        max =len(similar_triangle)
    for i in range(max):
        curr_transform =transform_points(similar_triangle[i][0],similar_triangle[i][1],points1)
        curr_shared_keypoints =find_shared_keypoints(curr_transform,points2,delta)
        if len(curr_shared_keypoints) > len(shared_keypoint):
            shared_keypoint = curr_shared_keypoints
            transform = curr_transform
            # find the original shared keypoint from point1 (Reverse transformation)
            curr_original = transform_points(similar_triangle[i][1], similar_triangle[i][0], shared_keypoint)
            curr_shared_original = find_shared_keypoints(curr_original,points1,delta)
            original = curr_shared_original

    return shared_keypoint, transform, original

"""
The draw_keypoints_triangle function takes an image, a triangle, 
and a list of keypoints, and draws the triangle and the keypoints on the image
"""
def draw_keypoints_triangle(image,triangle,keypoints):
    triangle = np.array(triangle)
    # Draw the triangle on the image
    # img_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
    img_combined= cv2.drawContours(image, [triangle.astype(int)], 0, (0, 0, 255), 2)
    image_with_lines_resized = cv2.resize(img_combined, (0, 0), fx=0.30, fy=0.30)

    return image_with_lines_resized

"""
 Display image
"""
def draw_image(image, group_pts):
    # Make a copy of the input image so we don't modify the original
    img = image.copy()

    box_size = 15

    # Draw group points in blue
    for pt in group_pts:
        # cv2.circle(img, (int(pt.pt[0]), int(pt.pt[1])), 3, (255, 0, 0), -1)
        cv2.rectangle(img, (int(pt.pt[0] - 10), int(pt.pt[1] - box_size)),
                      (int(pt.pt[0] + box_size), int(pt.pt[1] + box_size)), (0, 255, 0), 2)
    image_with_lines_resized = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    # Save image to "match photo" directory with filename
    match_photo_dir = "match photo"
    os.makedirs(match_photo_dir, exist_ok=True)  # create directory if it doesn't exist
    index = random.randint(1, 100)
    filepath = os.path.join(match_photo_dir, f'{index} match.jpg')
    cv2.imwrite(filepath, image_with_lines_resized)

    return image_with_lines_resized

"""
function that takes a cv2.KeyPoint object and prints its X and Y coordinates, 
as well as the values of its size (size) and response (response) attributes
"""
def print_keypoint_info(keypoints):
   for keypoint in keypoints:
        x, y = keypoint.pt
        size = keypoint.size
        response = keypoint.response
        print(f"X: {x}, Y: {y}, Size: {size}, Response: {response}")

"""
get an image and a list of triangles and draws triangles on the image based on those keypoints:
"""
def draw_triangles(image, triangles):
    # Convert image to grayscale

    # Create a copy of the original image to draw on
    img_copy = image.copy()


    # Draw the triangles on the copy of the image
    for triangle in triangles:
        pt1 = (int(triangle[0]), int(triangle[1]))
        pt2 = (int(triangle[2]), int(triangle[3]))
        pt3 = (int(triangle[4]), int(triangle[5]))
        cv2.line(img_copy, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img_copy, pt2, pt3, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img_copy, pt3, pt1, (0, 0, 255), 1, cv2.LINE_AA)
    image_with_lines_resized = cv2.resize(img_copy, (0, 0), fx=0.50, fy=0.50)
    cv2.imshow('Match', image_with_lines_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Return the copy of the image with the triangles drawn on it
    return image_with_lines_resized

"""

"""
def find_match_stras(image1,image2):
    triangle_list1 ,kp1 ,img1 = create_triangle(image1)
    triangle_list2 ,kp2 ,img2 = create_triangle(image2)
    angle_list1 = angle(triangle_list1)
    angle_list2 = angle(triangle_list2)
    similar_triangle = get_shared_angle(angle_list1,angle_list2)
    shared_keypoints ,transformed_points, original =best_tra_transform(similar_triangle,kp1,kp2,25)
    img3=draw_image(img2,shared_keypoints)
    img4=draw_image(img1,original)
    cv2.imshow('D.B', img4)
    cv2.imshow('Match', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print_keypoint_info(shared_keypoints)

find_match_stras('ST_db1.png','ST_db2.png')
find_match_stras('ST_db1.png','fr2.jpg')
find_match_stras('ST_db2.png','fr2.jpg')
find_match_stras('ST_db2.png','ST_db1.png')
find_match_stras('fr1.jpg','fr2.jpg')
find_match_stras('fr2.jpg','fr1.jpg')
find_match_stras('fr2.jpg','ST_db1.png')
find_match_stras('fr2.jpg','ST_db2.png')
