"""
    Project: Calibration-intrinsic
        to get the calibrated images
    Author: He Zhanxin
    Time:7/8/2017
"""

import numpy as np
import cv2
import glob
import os
import yaml

# The paths where the sample images are and where the calibrated images go
sample_path = '/home/hora/PycharmProjects/calibration_undistortion/calibration/pictures/'
cali_path = '/home/hora/PycharmProjects/calibration_undistortion/calibration/calibrated/'

# Import images path
distor_images_path = sorted(glob.glob(os.path.join(sample_path, '*.png')))


def calibration_circle_grid(sample_img_path, pattern_size):
    """

    :param pattern_size:
    :param sample_img_path: distorted image path
    :return: the list of intrinsic camera matrix and write into calibration_intrinsic_mtx.yaml
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Size of pattern
    row_size = pattern_size[0]
    col_size = pattern_size[1]

    # Blob detector
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1200
    params.maxArea = 4000

    params.minDistBetweenBlobs = 100
    params.filterByColor = False

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.maxCircularity = 1.2
    detector = cv2.SimpleBlobDetector_create(params)

    # prepare object points
    objp = np.zeros((row_size * col_size, 3), np.float32)
    objp[:, :2] = np.mgrid[0:col_size, 0:row_size].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    # Start to find circles in each photo
    img_index = 1

    for sample in sample_img_path:
        img_name = cv2.imread(sample)  # Import image
        img_gray = cv2.cvtColor(img_name, cv2.COLOR_BGR2GRAY)

        key_points = detector.detect(img_gray)  # Detect blob

        img_with_circle = cv2.drawKeypoints(img_name, key_points, np.array([]), (255, 0, 0),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_with_circle_gray = cv2.cvtColor(img_with_circle, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findCirclesGrid(img_with_circle, (7, 7), flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                                           blobDetector=detector)

        # Draw and display the corners
        if ret:
            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(img_with_circle_gray, corners, (11, 11), (-1, -1),
                                        criteria)  # Refines the corner locations.
            img_points.append(corners2)

            # Draw and display the corners
            img_with_circle = cv2.drawChessboardCorners(img_name, (col_size, row_size), corners2, ret)
        cv2.imshow("img{}".format(img_index), img_with_circle[650:1800, 1300:2400])

        img_index += 1

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ret, intrinsic_mtx, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_gray.shape[::-1],
                                                                           None, None)
        # Transform the matrix to list.
        # data = {'camera_matrix': np.asarray(intrinsic_mtx).tolist(), 'dist_coeff': np.asarray(dist_coeff).tolist()}
        # with open("calibration_intrinsic_mtx.yaml", "w") as f:
        #     yaml.dump(data, f)
        print(intrinsic_mtx)
# def calibration(sample_img, row_size, col_size):
#     # termination criteria
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#     # Check board size
#
#     # prepare object points
#     objp = np.zeros((row_size * col_size, 3), np.float32)
#     objp[:, :2] = np.mgrid[0:col_size, 0:row_size].T.reshape(-1, 2)
#
#     # Arrays to store object points and image points from all the images.
#     obj_points = []  # 3d point in real world space
#     img_points = []  # 2d points in image plane.
#
#     # import image
#     gray_img_circled = sample_img
#     h, w = gray_img_circled.shape[:2]
#
#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray_img_circled, (col_size, row_size), None)
#
#     # If found, add object points, image points (after refining them)
#     if ret:
#         obj_points.append(objp)
#
#         corners2 = cv2.cornerSubPix(gray_img_circled, corners, (11, 11), (-1, -1), criteria)
#         img_points.append(corners2)
#
#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(gray_img_circled, (col_size, row_size), corners2, ret)
#
#     ret, mtx, dist_coe, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_img_circled.shape[::-1], None,
#                                                            None)
#     new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_coe, (w, h), 1, (w, h))
#     new_camera_mtx = np.hstack((new_camera_mtx, np.zeros((3, 1))))
#     # un-distort
#     dst = cv2.undistort(img, mtx, dist_coe, new_camera_mtx)
#
#     # crop the image
#     x, y, w, h = roi
#     return dst[y:y + h, x:x + w], new_camera_mtx, roi, (w, h)


pattern = (7, 7)
calibration_circle_grid(distor_images_path, pattern)
