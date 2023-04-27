import random
import cv2
import numpy as np


# margin range 0.0-0.499..
def remove_edge_stars(keypoints, width, height, margin=0.1):
    # Define the range of x and y coordinates to include
    range_x = (int(width * margin), int(width * (1 - margin)))
    range_y = (int(height * margin), int(height * (1 - margin)))

    # Filter out keypoints that are outside the defined range
    filtered = [kp for kp in keypoints if range_x[0] <= kp.pt[0] <= range_x[1] and range_y[0] <= kp.pt[1] <= range_y[1]]

    return filtered


def process_image(image_path):
    # Load the image and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply a 5x5 Gaussian blur to the image
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Canny edge detection with threshold values of 100 and 200
    filtered_img = cv2.Canny(img_blur, 100, 200)

    # Detect keypoints using SIFT
    sift = cv2.SIFT_create(sigma=1.6, edgeThreshold=20, contrastThreshold=0.07, nOctaveLayers=15)
    keypoints = sift.detect(filtered_img, None)

    # Remove keypoints that are too close to each other
    min_distance = 20  # minimum distance between keypoints
    filtered_keypoints = []
    for i, kp1 in enumerate(keypoints):
        is_close = False
        for kp2 in filtered_keypoints:
            distance = ((kp1.pt[0] - kp2.pt[0]) ** 2 + (kp1.pt[1] - kp2.pt[1]) ** 2) ** 0.5
            if distance < min_distance:
                is_close = True
                break
        if not is_close:
            filtered_keypoints.append(kp1)

    # Draw keypoints on the filtered image
    img_with_keypoints = cv2.drawKeypoints(filtered_img, filtered_keypoints, None, color=(255, 255, 255))

    return img_with_keypoints, filtered_keypoints


def ransac(keypoints, range_value, max_iterations):
    best_inliers = []
    best_p1 = None
    best_p2 = None

    for i in range(max_iterations):
        # Select 2 points at random
        p1, p2 = random.sample(keypoints, 2)

        # Compute slope and intercept of the line connecting the two points
        x1, y1 = p1.pt
        x2, y2 = p2.pt
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Find the inliers that lie within the range and between the two points
        inliers = []
        for kp in keypoints:
            x, y = kp.pt
            if abs(y - (slope * x + intercept)) < range_value:
                if x1 <= x <= x2 or x2 <= x <= x1:
                    inliers.append(kp)

        # Update the best set of inliers found so far
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_p1 = p1
            best_p2 = p2

    return len(best_inliers), best_inliers, best_p1, best_p2


def ransac_match(keypoints, range_value, max_iterations, inlier_count, delta):
    best_inliers = []
    best_p1 = None
    best_p2 = None

    for i in range(max_iterations):
        # Select 2 points at random
        p1, p2 = random.sample(keypoints, 2)

        # Compute slope and intercept of the line connecting the two points
        x1, y1 = p1.pt
        x2, y2 = p2.pt
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Find the inliers that lie within the range and between the two points
        inliers = []
        for kp in keypoints:
            x, y = kp.pt
            if abs(y - (slope * x + intercept)) < range_value:
                if x1 <= x <= x2 or x2 <= x <= x1:
                    inliers.append(kp)

        # Update the best set of inliers found so far
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_p1 = p1
            best_p2 = p2

        if (len(best_inliers) >= inlier_count - delta) and (len(best_inliers) >= inlier_count + delta):
            break

    return len(best_inliers), best_inliers, best_p1, best_p2


def draw_image(image, group_a_pts, group_b_pts, p1, p2):
    # Make a copy of the input image so we don't modify the original
    img = image.copy()

    # Draw group A points in yellow
    for pt in group_a_pts:
        cv2.circle(img, (int(pt.pt[0]), int(pt.pt[1])), 6, (0, 255, 255), -1)
    box_size = 10

    # Draw group B points in blue
    for pt in group_b_pts:
        cv2.circle(img, (int(pt.pt[0]), int(pt.pt[1])), 3, (255, 0, 0), -1)
        cv2.rectangle(img, (int(pt.pt[0] - 10), int(pt.pt[1] - box_size)),
                      (int(pt.pt[0] + box_size), int(pt.pt[1] + box_size)), (0, 255, 0), 2)
    # Draw a green square around points p1 and p2
    cv2.rectangle(img, (int(p1.pt[0] - box_size), int(p1.pt[1] - box_size)),
                  (int(p1.pt[0] + box_size), int(p1.pt[1] + box_size)), (0, 255, 0), 2)
    cv2.rectangle(img, (int(p2.pt[0] - box_size), int(p2.pt[1] - box_size)),
                  (int(p2.pt[0] + box_size), int(p2.pt[1] + box_size)), (0, 255, 0), 2)

    # Draw a red line between points p1 and p2
    cv2.line(img, (int(p1.pt[0]), int(p1.pt[1])), (int(p2.pt[0]), int(p2.pt[1])), (0, 0, 255), 2)
    image_with_lines_resized = cv2.resize(img, (0, 0), fx=0.30, fy=0.30)

    return image_with_lines_resized


def draw_image_match(image, group_a_pts, group_b_pts):
    # Make a copy of the input image so we don't modify the original
    img = image.copy()

    box_size = 15

    # Draw group B points in blue
    for pt in group_b_pts:
        cv2.rectangle(img, (int(pt.pt[0] - 10), int(pt.pt[1] - box_size)),
                      (int(pt.pt[0] + box_size), int(pt.pt[1] + box_size)), (0, 255, 0), 2)

    image_with_lines_resized = cv2.resize(img, (0, 0), fx=0.30, fy=0.30)

    return image_with_lines_resized


def transform_keypoints(input_pts, src_pt1, src_pt2, dst_pt1, dst_pt2):
    # Compute the transformation matrix using the two pairs of corresponding points
    src_pts = np.array([[src_pt1.pt[0], src_pt1.pt[1]], [src_pt2.pt[0], src_pt2.pt[1]]])
    dst_pts = np.array([[dst_pt1.pt[0], dst_pt1.pt[1]], [dst_pt2.pt[0], dst_pt2.pt[1]]])
    T = np.linalg.solve(src_pts, dst_pts)

    # Transform the input keypoints using the computed matrix
    transformed_pts = []
    for kp in input_pts:
        x, y = T @ np.array([kp.pt[0], kp.pt[1]])
        transformed_kp = cv2.KeyPoint(x, y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        transformed_pts.append(transformed_kp)

    return transformed_pts


def find_shared_keypoints(keypoints1, keypoints2, delta_range):
    shared_keypoints = []
    for kp1 in keypoints1:
        for kp2 in keypoints2:
            delta = np.sqrt((kp1.pt[0] - kp2.pt[0]) ** 2 + (kp1.pt[1] - kp2.pt[1]) ** 2)
            if delta <= delta_range:
                shared_keypoints.append(kp1)
                break
    return shared_keypoints


def main():
    # Load the image and convert to grayscale
    image_path = 'ST_db2.png'
    img, keypoints = process_image(image_path)

    # Filter out keypoints on the edges of the image
    keypoints = remove_edge_stars(keypoints, img.shape[1], img.shape[0])

    # Fit a line to the remaining keypoints using RANSAC
    # inliers_count, slope, intercept, inliers, p1, p2= ransac(keypoints, 100, 10000)
    inliers_count, inliers, p1, p2 = ransac(keypoints, 100, 1000000)
    image_with_line_resized = draw_image(img, keypoints, inliers, p1, p2)

    # Load the image and convert to grayscale
    image_path1 = 'ST_db1.png'
    img1, keypoints1 = process_image(image_path1)
    # Filter out keypoints on the edges of the image
    keypoints1 = remove_edge_stars(keypoints1, img1.shape[1], img1.shape[0])

    # Fit a line to the remaining keypoints using RANSAC
    inliers_count1, inliers1, p11, p21 = ransac_match(keypoints1, 100, 1000000, inliers_count, 0)
    image_with_line_resized1 = draw_image(img1, keypoints1, inliers1, p11, p21)

    transformed_pts = transform_keypoints(keypoints, p1, p2, p11, p21)

    match_keypoints = find_shared_keypoints(transformed_pts, keypoints1, 500)

    image_with_line_resized_match = draw_image_match(img1, keypoints1, match_keypoints)

    print(match_keypoints)
    # Show the image
    cv2.imshow('Original', image_with_line_resized)
    cv2.imshow('Opened', image_with_line_resized1)
    cv2.imshow('match', image_with_line_resized_match)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
