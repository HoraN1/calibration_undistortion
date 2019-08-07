"""
    Project: Calibration-intrinsic
        to get the calibrated images
    Author: He Zhanxin
    Time:4/8/2017
"""

import numpy as np
import cv2
import glob
import os


def calibration_chessboard(sample_img):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Check board size
    cbrow = 6
    cbcol = 9

    # prepare object points
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    # import image
    img = sample_img
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray_img, (cbcol, cbrow), None)

    # If found, add object points, image points (after refining them)
    if ret:
        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(gray_img, (cbcol, cbrow), corners2, ret)

    ret, mtx, dist_coe, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None)
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_coe, (w, h), 1, (w, h))
    new_camera_mtx = np.hstack((new_camera_mtx, np.zeros((3, 1))))
    # un-distort
    dst = cv2.undistort(img, mtx, dist_coe, new_camera_mtx)

    # crop the image
    x, y, w, h = roi
    return dst[y:y + h, x:x + w], new_camera_mtx, roi, (w, h)


# The paths where the sample images are and where the calibrated images go
sample_path = '/home/hora/PycharmProjects/calibration_undistortion/calibration/pictures/'
cali_path = '/home/hora/PycharmProjects/calibration_undistortion/calibration/calibrated/'

# Import images

distor_images = sorted(glob.glob(os.path.join(sample_path, '*.jpg')))
img_index = 1

for sample in distor_images:
    img_name = cv2.imread(sample)
    undistorted_img, new_cam_mtx, roi_para, size = calibration_chessboard(img_name)
    cv2.imwrite(os.path.join(cali_path, 'calibrated-sample{}.jpg'.format(img_index)), undistorted_img)

    calibrated_img = cv2.imread(os.path.join(cali_path, 'calibrated-sample{}.jpg').format(img_index))
    cv2.imshow('img{}'.format(img_index), calibrated_img)
    print("sample{}".format(img_index) + '\n', new_cam_mtx, roi_para, size)

    img_index += 1

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
