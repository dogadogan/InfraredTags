import os
import random
import time

import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import pyzbar.pyzbar as zbar

from image_transforms import mask
from image_transforms import transforms_v1

##########################################################################################################################################
# GLOBAL VARIABLES
##########################################################################################################################################

VIDEO_STREAM = 1  # 0 for default cam 1 for usb and [url] for wireless stream

##########################################################################################################################################
# MUTABLES
##########################################################################################################################################
BLOCKSIZE_IN = 23  # Adaptive Threshold Params
SUBTRACTED_IN = 4  # Adaptive Threshold Params
CLIPLIMIT_IN = 20  # CLAHE PARAMS
GRIDSIZE_IN = 8  # CLAHE PARAMS
BLUR_IN = 3  # Gaussian Param
SCALE_IN = 40  # Scale Param
DETECTING_IN = 1  # 0 detect Aruco, 1 detect QR Code

##########################################################################################################################################
# WINDOWS
##########################################################################################################################################
title_window = 'InfraredTags Calibration'
cv.namedWindow("Original", cv.WINDOW_NORMAL)
cv.namedWindow(title_window, cv.WINDOW_NORMAL)
cv.namedWindow("Decoded", cv.WINDOW_NORMAL)

##########################################################################################################################################
# CAMERA CALIBRATION
##########################################################################################################################################
calib_file = os.path.join(os.getcwd(), "calib.json")
calibrationParams = cv.FileStorage(calib_file, cv.FILE_STORAGE_READ)
print("CALIBRATION PARAMS: {}".format(calibrationParams))
DEFAULT_CAMERA_MATRIX = calibrationParams.getNode("camera_matrix").mat()
DEFAULT_DISTORTION = calibrationParams.getNode("distortion_coefficients").mat()


##########################################################################################################################################
# UTILITY FUNCTIONS and SLIDER EVENT FUNCTIONS
##########################################################################################################################################

def printUserInputValues():
    print("Adaptive Threshold BlockSize: {}, Adaptive Threshold Subtracted Value: {}\n"
          "CLAHE ClipLimit: {}, CLAHE tileGridSize: {}\n"
          "SCALE: {}\n"
          "BLUR: {}\n"
          "DETECTING: {}".format(BLOCKSIZE_IN, SUBTRACTED_IN, CLIPLIMIT_IN, GRIDSIZE_IN, SCALE_IN, BLUR_IN,
                                 "ARUCO CODE" if DETECTING_IN == 0 else "QR CODE"))


def on_input1(val):
    global BLOCKSIZE_IN
    BLOCKSIZE_IN = 2 * int(val) + 3
    printUserInputValues()


def on_input2(val):
    global SUBTRACTED_IN
    SUBTRACTED_IN = int(val)
    printUserInputValues()


def on_input3(val):
    global CLIPLIMIT_IN
    CLIPLIMIT_IN = int(val)
    printUserInputValues()


def on_input4(val):
    global GRIDSIZE_IN
    GRIDSIZE_IN = int(val)
    printUserInputValues()


def on_input5(val):
    global SCALE_IN
    SCALE_IN = int(val)
    printUserInputValues()


def on_input6(val):
    global BLUR_IN
    BLUR_IN = 2 * int(val) + 1
    printUserInputValues()


def on_input7(val):
    global DETECTING_IN
    DETECTING_IN = int(val)
    printUserInputValues()


cv.createTrackbar("BlockSize", title_window, 0, 100, on_input1)
cv.createTrackbar("Subtracted", title_window, 0, 100, on_input2)
cv.createTrackbar("ClipLimit", title_window, 0, 255, on_input3)
cv.createTrackbar("GridSize", title_window, 0, 100, on_input4)
cv.createTrackbar("SCALE", title_window, 0, 100, on_input5)
cv.createTrackbar("BLUR", title_window, 0, 100, on_input6)
cv.createTrackbar("MODE", title_window, 0, 1, on_input7)


# ########################################################################################################################################
# DECODING METHODS
# #########################################################################################################################################
def ARUCO(image, matrix_coefficients=DEFAULT_CAMERA_MATRIX, distortion_coefficients=DEFAULT_DISTORTION):
    initial = time.time()
    """
    :param image: OpenCV image frame
    :param matrix_coefficients: camera distortion matrix defaults to globally declared DEFAULT DISTORTION
    :param distortion_coefficients: distortion coefficients defaults to globally declared DEFAULT CAMERA MATRIX
    :return: tuple(ARUCO Detected image frame, JSON metadata for each frame)
    """
    ARUCO_CODES_JSON = []

    def decode_and_display(image_to_decode, image_to_display=None, initial=initial):
        """
        :param initial:
        :param image_to_decode: image that you want to decode
        :param image_to_display: if not none will display this image instead of decoded one
        :return: image_to_display with code detected
        """
        is_detected = False

        def drawPointer(image_in, rvec_in, tvec_in, cameraMatrix, distCoeff, length=0.01):
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
            cv.line(image, (round(image_z_axis[0][0] * (100 / SCALE_IN)), round(image_z_axis[0][1] * (100 / SCALE_IN))),
                    (round(image_z_axis[1][0] * (100 / SCALE_IN)), round(image_z_axis[1][1] * (100 / SCALE_IN))),
                    (255, 0, 0), 4)
            cv.line(image, (round(image_y_axis[0][0] * (100 / SCALE_IN)), round(image_y_axis[0][1] * (100 / SCALE_IN))),
                    (round(image_y_axis[1][0] * (100 / SCALE_IN)), round(image_y_axis[1][1] * (100 / SCALE_IN))),
                    (0, 255, 0), 4)
            cv.line(image, (round(image_x_axis[0][0] * (100 / SCALE_IN)), round(image_x_axis[0][1] * (100 / SCALE_IN))),
                    (round(image_x_axis[1][0] * (100 / SCALE_IN)), round(image_x_axis[1][1] * (100 / SCALE_IN))),
                    (0, 0, 255), 4)

        image_to_display = image_to_decode if image_to_display is None else image_to_display
        aruco_dict, parameters = aruco.Dictionary_get(aruco.DICT_4X4_50), aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = aruco.detectMarkers(image_to_decode, aruco_dict, parameters=parameters,
                                                                cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients)
        detected_image = image_to_decode
        if ids is None:
            corners, ids, rejected_img_points = aruco.detectMarkers(255 - image_to_decode, aruco_dict,
                                                                    parameters=parameters,
                                                                    cameraMatrix=matrix_coefficients,
                                                                    distCoeff=distortion_coefficients)
            detected_image = 255 - image_to_decode

        if np.all(ids is not None):
            is_detected = True
            print("Time: {}".format((time.time() - initial)))
            for i in range(len(ids)):
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)
                corners_new = [corners[i] * (100 / SCALE_IN) for i in range(len(corners))]
                aruco.drawDetectedMarkers(image_to_display, corners_new)
                drawPointer(image_to_display, rvec, tvec, matrix_coefficients, distortion_coefficients)

        return image_to_display, ARUCO_CODES_JSON, is_detected, detected_image

    parmas = {"SCALE_IN": SCALE_IN,
              "CLIPLIMIT_IN": CLIPLIMIT_IN,
              "GRIDSIZE_IN": GRIDSIZE_IN,
              "BLUR_IN": BLUR_IN,
              "BLOCKSIZE_IN": BLOCKSIZE_IN,
              "SUBTRACTED_IN": SUBTRACTED_IN}
    imageToDecode = transforms_v1(image, parmas)
    result = decode_and_display(imageToDecode, image)
    return result[0], result[-1]


def QR(image, decoder):
    initial = time.time()
    """
    :param image: OpenCV image frame
    :param matrix_coefficients: camera distortion matrix defaults to globally declared DEFAULT DISTORTION
    :param distortion_coefficients: distortion coefficients defaults to globally declared DEFAULT CAMERA MATRIX
    :return: tuple(ARUCO Detected image frame, JSON metadata for each frame)
    """
    QR_CODES_JSON = []

    def decode_and_display(image_to_decode, image_to_display=None, initial=initial):
        """
        :param initial:
        :param image_to_decode: image that you want to decode
        :param image_to_display: if not none will display this image instead of decoded one
        :return: image_to_display with code detected
        """
        is_detected = False
        polygon = [[0, 0], [image_to_decode.shape[1], 0], [image_to_decode.shape[1], image_to_decode.shape[0]],
                   [0, image_to_decode.shape[0]]]

        def computePixelMidPoint(rect):
            (x_1, y_1, w_1, h_1) = rect
            return round(x_1 + (w_1 / 2)), round(y_1 + (h_1 / 2))

        image_to_display = image_to_decode if image_to_display is None else image_to_display
        decodedObjects = zbar.decode(image_to_decode)
        detected_image = image_to_decode
        if decodedObjects is None:
            decodedObjects = zbar.decode(255 - image_to_decode)
            detected_image = 255 - image_to_decode
        for code in decodedObjects:
            is_detected = True
            print("Time: {}".format((time.time() - initial)))
            (x, y, w, h) = code.rect
            (p1, p2, p3, p4) = code.polygon
            polygon = (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)
            polygon = [list(pt) for pt in polygon]
            # print("COORDINATE_OLD: ({},{},{},{})".format(x, y, w, h)) print("COORDINATE_NEW: ({},{},{},{})".format(
            # int(100*x/SCALE_IN), int(100*y/SCALE_IN), int(100*w/SCALE_IN), int(100*h/SCALE_IN)))
            (x, y, w, h) = (
                round(100 * x / SCALE_IN), round(100 * y / SCALE_IN), round(100 * w / SCALE_IN),
                round(100 * h / SCALE_IN))

            cv.rectangle(image_to_display, (x, y), (x + w, y + h), (100, 100, 255), 2)
            barcodeData, barcodeType = code.data.decode("utf-8"), code.type

            text = "{}_{} ({})".format(random.randint(0, 100), barcodeData, barcodeType)
            if barcodeType == "QRCODE":
                image_to_display = cv.circle(image_to_display, computePixelMidPoint((x, y, w, h)), radius=6,
                                             color=(0, 0, 255), thickness=-1)
                cv.putText(image_to_display, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
        return image_to_display, QR_CODES_JSON, is_detected, mask(detected_image, polygon)

    parmas = {"SCALE_IN": SCALE_IN,
              "CLIPLIMIT_IN": CLIPLIMIT_IN,
              "GRIDSIZE_IN": GRIDSIZE_IN,
              "BLUR_IN": BLUR_IN,
              "BLOCKSIZE_IN": BLOCKSIZE_IN,
              "SUBTRACTED_IN": SUBTRACTED_IN}
    imageToDecode = transforms_v1(image, parmas)
    result = decode_and_display(imageToDecode, image)
    return result[0], result[-1]


#

def main():
    printUserInputValues()
    cap = cv.VideoCapture(VIDEO_STREAM)

    while True:
        ret, frame = cap.read()
       #q print(frame.shape)
        if DETECTING_IN == 1:
            img, tra = QR(frame)
        elif DETECTING_IN == 0:
            img, tra = ARUCO(frame)
        cv.imshow("Original", frame)
        cv.imshow(title_window, tra)
        cv.imshow("Decoded", img)

        cv.moveWindow("Original", 0, 0)
        cv.moveWindow(title_window, 1000, 0)
        cv.moveWindow("Decoded", 500, 0)

        cv.resizeWindow("Original", 500, 300)
        cv.resizeWindow(title_window, 500, 300)
        cv.resizeWindow("Decoded", 500, 300)

        # ret, frame = cap.read()
        # cv.imshow('Video', frame)

        if (cv.waitKey(1) & 0xFF) == ord("q"):
            break
            exit(0)
    cap.release()
    cv.destroyAllWindows()


##########################################################################################################################################
# MAIN
##########################################################################################################################################

if __name__ == "__main__":
    main()
    
