import cv2 as cv
import cv2.aruco as aruco
import time
import os
from image_transforms import transforms_v1
import numpy as np

##########################################################################################################################################
# CAMERA CALIBRATION
##########################################################################################################################################

calib_file = os.path.join(os.getcwd(), "calib.json")
calibrationParams = cv.FileStorage(calib_file, cv.FILE_STORAGE_READ)
print("CALIBRATION PARAMS: {}".format(calibrationParams))
DEFAULT_CAMERA_MATRIX = calibrationParams.getNode("camera_matrix").mat()
DEFAULT_DISTORTION = calibrationParams.getNode("distortion_coefficients").mat()


def drawPointer(image, rvec_in, tvec_in, cameraMatrix, distCoeff, length=0.01):
    global initial
    z_axis = [[0, 0, 0], [0, 0, length]]
    y_axis = [[0, 0, 0], [0, length, 0]]
    x_axis = [[0, 0, 0], [length, 0, 0]]
    # print("AXES: {}".format(np.array([z_axis]).reshape(2, 3)))
    imagePoints, jacobian = cv.projectPoints(np.array([*z_axis, *y_axis, *x_axis]).reshape((6, 3)), rvec_in,
                                             tvec_in,
                                             cameraMatrix, distCoeff)
    imagePoints = np.rint(imagePoints.reshape(3, 2, 2))
    # print("IMAGE_POINTS: {}".format(imagePoints).replace("\n", ""))
    image_z_axis = imagePoints[0]
    image_y_axis = imagePoints[1]
    image_x_axis = imagePoints[2]
    cv.line(image, (round(image_z_axis[0][0]), round(image_z_axis[0][1])),
            (round(image_z_axis[1][0]), round(image_z_axis[1][1])),
            (255, 0, 0), 4)
    cv.line(image, (round(image_y_axis[0][0]), round(image_y_axis[0][1])),
            (round(image_y_axis[1][0]), round(image_y_axis[1][1])),
            (0, 255, 0), 4)
    cv.line(image, (round(image_x_axis[0][0]), round(image_x_axis[0][1])),
            (round(image_x_axis[1][0]), round(image_x_axis[1][1])),
            (0, 0, 255), 4)


def is_aruco(image, matrix_coefficients=DEFAULT_CAMERA_MATRIX, distortion_coefficients=DEFAULT_DISTORTION):
    """
    :param image: OpenCV image frame
    :param matrix_coefficients: camera distortion matrix defaults to globally declared DEFAULT DISTORTION
    :param distortion_coefficients: distortion coefficients defaults to globally declared DEFAULT CAMERA MATRIX
    :return: tuple(ARUCO Detected image frame, JSON metadata for each frame)
    """

    is_detected = False

    image = transforms_v1(image)
    aruco_dict, parameters = aruco.Dictionary_get(aruco.DICT_4X4_50), aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = aruco.detectMarkers(image, aruco_dict, parameters=parameters,
                                                            cameraMatrix=matrix_coefficients,
                                                            distCoeff=distortion_coefficients)
    if ids is None:
        corners, ids, rejected_img_points = aruco.detectMarkers(255 - image, aruco_dict,
                                                                parameters=parameters,
                                                                cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients)

    if np.all(ids is not None):
        is_detected = True
        for i in range(len(ids)):
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                       distortion_coefficients)
            corners_new = [corners[i] for i in range(len(corners))]
            aruco.drawDetectedMarkers(image, corners_new)
            drawPointer(image, rvec, tvec, matrix_coefficients, distortion_coefficients)

        return is_detected, corners
