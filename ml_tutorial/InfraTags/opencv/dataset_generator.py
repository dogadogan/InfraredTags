from matplotlib.font_manager import FontProperties
import splitfolders
import cv2 as cv
import sys
import getopt
import time
import itertools
import datetime
import csv
import os
import re
import pyzbar.pyzbar as zbar
import numpy as np
import cv2.aruco as aruco
from wechat_qrcode import get_wechat_qr_detector
from dbr_qrcode import get_dbr_detector
from image_transforms import mask

"""
VARY THE RANGES FOR EACH TRANSFORM IN ORDER TO GET BIGGER OR SMALLER DATASET
"""
TRANSFORMS = [
    (
        (lambda image, params=[]:  # params = [int(scale)]
         cv.resize(image, int(image.shape[1] * params[0] / 100), int(image.shape[0] * params[0] / 100))),
        [list(range(2678, 2678 + 1, 1))] #2678
    ),
    (
        (lambda image, params=[]:  # params = list
         cv.cvtColor(image, cv.COLOR_BGR2GRAY)),
        []
    ),
    (
        (lambda image, params=[]:  # params = [int(cliplimit), int(gridsize)]
         cv.createCLAHE(params[0], (params[1], params[1])).apply(image)),
        [list(range(4, 8 + 4, 4)), list(range(8, 16 + 8, 8))]
    ),
    (
        (lambda image, params=[]:  # params = [int(blur_in)]
         cv.GaussianBlur(image, (params[0], params[0]), 0)),
        [list(range(1, 7 + 2, 2))]
    ),
    (
        (lambda image, params=[]:  # params = [int(blocksize), int(subtracted)]
         cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, params[0], params[1])),
        [list(range(3, 101 + 2, 2)), list(range(0, 0 + 1, 1))]
    ),
    (
        (lambda image, params=[]:  # params = [int(detecting)]
         # ARUCO(image)[2] if params[0] else QR(image)[2]
         None
         ),
        [list(range(1, 1 + 1, 1))]
    )

]

ID = 0
OUTPUT_IMAGE = None


def is_QR(image, scale):
    is_detected = False
    result = [[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]]
    text = ""

    decodedObjects = zbar.decode(image, symbols=[zbar.ZBarSymbol.QRCODE])
    detected_image = image
    if not decodedObjects:
        decodedObjects = zbar.decode(255 - image,
                                     symbols=[zbar.ZBarSymbol.QRCODE])
        detected_image = 255 - image
    if decodedObjects:
        is_detected = True
        for code in decodedObjects:
            (p1, p2, p3, p4) = code.polygon
            text, _ = code.data.decode("utf-8"), code.type
            result = (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)
    return is_detected, text, [list(pt) for pt in result], detected_image


##########################################################################################################################################
# CAMERA CALIBRATION
##########################################################################################################################################
calib_file = os.path.join(os.getcwd(), "calib.json")
calibrationParams = cv.FileStorage(calib_file, cv.FILE_STORAGE_READ)
#print("CALIBRATION PARAMS: {}".format(calibrationParams))
DEFAULT_CAMERA_MATRIX = calibrationParams.getNode("camera_matrix").mat()
DEFAULT_DISTORTION = calibrationParams.getNode("distortion_coefficients").mat()


def is_ARUCO(image, scale, id, matrix_coefficients=DEFAULT_CAMERA_MATRIX, distortion_coefficients=DEFAULT_DISTORTION):
    is_detected = False
    detected_image = image
    aruco_dict, parameters = aruco.Dictionary_get(aruco.DICT_4X4_50), aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = aruco.detectMarkers(image, aruco_dict, parameters=parameters,
                                                            cameraMatrix=matrix_coefficients,
                                                            distCoeff=distortion_coefficients)

    if ids is None:
        corners, ids, rejected_img_points = aruco.detectMarkers(255 - image, aruco_dict,
                                                                parameters=parameters,
                                                                cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients)
        detected_image = 255 - image

    corners_new = [[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]]
    if np.all(ids is not None):
        for i in range(len(ids)):
            if ids[i][0] == id:
                is_detected = True
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)

                (rvec - tvec).any()
                corners_new = [corners[i] for i in range(len(corners))]
                corners_new = [[int(pt[0]), int(pt[1])] for pt in corners_new[0][0]]
    return is_detected, "ARUCO", corners_new, detected_image


def is_dbr_QR(image, detector, scale):
    is_detected = False
    result = [[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]]
    text = ""
    detected_image = image

    image_array = np.array(image)
    image_array = np.stack((image_array,) * 3, axis=-1)

    decoded = detector.decode_buffer(image_array)
    if decoded == None:
        decoded = detector.decode_buffer(255 - image_array)
        detected_image = 255 - image
    if decoded != None:
        is_detected = True
        text = decoded[0].barcode_text
        result = decoded[0].localization_result.localization_points
    return is_detected, text, [list(pt) for pt in result], detected_image


def is_wechat_QR(image, detector):
    is_detected = False
    result = None, None, None, None
    text = ""

    res, points = detector.detectAndDecode(np.array(image))
    if len(res) == 0:
        res, points = detector.detectAndDecode(255 - np.array(image))
    if len(res) != 0:
        is_detected = True
        text = res[0]
        coors = points[0]
        result = [tuple(coors[0]), tuple(coors[1]), tuple(coors[2]), tuple(coors[3])]
    return is_detected, text, result


def detected(image, params_list, mode='pyzbar', detector=None, id=None):
    global OUTPUT_IMAGE
    """
    :param image: OpenCV image frame
    :param params_lsit: [int(scale), int(cliplimit), int(gridsize), int(blur), int(blocksize), int(subtracted), int(detected)]
    :param invert: if true output image is inverted
    :return: transformed image
    """
    (width, height) = int(image.shape[1] * params_list[0] / 10000), int(image.shape[0] * params_list[0] / 10000)
    # print("SIZE: {}".format((width, height)))
    resize = cv.resize(image, (width, height))
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(params_list[1], (params_list[2], params_list[2])).apply(gray)
    blur = cv.GaussianBlur(clahe, (params_list[3], params_list[3]), 0)
    adaptive_threshold = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                              params_list[4],
                                              params_list[5])
    # integral_threshold = thresholdIntegral(blur, cv.integral(blur))
    output = adaptive_threshold  # change this value to output different transforms

    if mode == 'wechat':
        result = is_wechat_QR(output, detector, params_list[0])
    elif mode == 'dbr':
        result = is_dbr_QR(output, detector, params_list[0])
    elif mode == 'aruco':
        result = is_ARUCO(output, params_list[0], id)
    else:
        result = is_QR(output, params_list[0])
    OUTPUT_IMAGE = result[-1]
    return result


def get_COM(points):
    sums = [0 for j in range(len(points[0]))]
    length = len(points)
    for coordinate in points:
        for k, dimension in enumerate(coordinate):
            sums[k] += dimension / length
    return [round(s, 2) for s in sums]


def is_near_average(coordinate, COM, radius=20):
    for i, dimension in enumerate(coordinate):
        if not (dimension - COM[i]) ** 2 <= radius ** 2:
            return False
    return True


def generate_transforms_full_space(original_image, writer, parent_dir, transforms_list=TRANSFORMS, mode='pyzbar',
                                   detector=None, id=None):
    global ID
    initial = time.time()

    range_list = []
    for i in range(len(transforms_list)):
        transform, ranges = transforms_list[i][0], transforms_list[i][1]
        for r in ranges:
            if r:
                range_list.append(r)

    points = [item for item in itertools.product(*range_list)]
    com = get_COM(points)
    det = False
    print("WRITING CSV (MAY TAKE A WHILE) .....")
    count = 0
    detected_comb_list = []
    for comb in points:
        count += 1
        params_list = list(comb)
        output = detected(original_image, params_list, mode, detector, id)
        is_detected, text, polygon, output_image = output
        masked_image_output = mask(output_image, polygon)
        if is_detected:
            row_dict = {
                "IMAGE_ID": str("{}_input".format(ID)),
                "SCALE": params_list[0],
                "CLAHE_CLIPLIMIT": params_list[1],
                "CLAHE_GRIDSIZE": params_list[2],
                "BLUR": params_list[3],
                "AT_BLOCKSIZE": params_list[4],
                "AT_SUBTRACTED": params_list[5],
                "DETECTED": is_detected,
                "CODE_TEXT": text,
                "CODE_DIM": polygon,
                "OUT_IMAGE_DIM": output_image.shape,
                "MASKED_OUTPUT_ID": str("{}_output".format(ID)),
                "OUTPUT_IMAGE_ID": str("{}_masked_output".format(ID))
            }
            detected_comb_list.append((ID, original_image, output_image, row_dict, masked_image_output))
            ID += 1
            det = True
    if detected_comb_list:
        mid_point = len(detected_comb_list) // 2
        ls_id, ls_original_image, ls_output_image, ls_row_dict, ls_output_masked_image = detected_comb_list[mid_point]
        cv.imwrite(parent_dir + "/dataset_images/input/" + str(ls_id) + "_input.png", ls_original_image)
        cv.imwrite(parent_dir + "/dataset_images/output/" + str(ls_id) + "_output.png", ls_output_image)
        cv.imwrite(parent_dir + "/dataset_images/masked/" + str(ls_id) + "_masked_output.png", ls_output_masked_image)
        writer.writerow(ls_row_dict)

    print("COMPLETE (TIME: {}, {} Combinations Generated, Detected: {})".format(time.time() - initial, count, det))


##########################################################################################################################################
# MAIN
##########################################################################################################################################


def main(argv):
    i = -1
    inputfile = ''
    mode = ''
    print("STARTING DBR......")
    try:
        opts, args = getopt.getopt(argv, "hi:m:f:d:", ["ifile=", "mode=", "fpr=", "id="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -m <mode>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -m <mode>')
            sys.exit(2)
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-m", "--mode"):
            mode = arg
        elif opt in ("-f", "--fpr"):
            fpr = int(arg)
        elif opt in ("-d", "--id"):
            id = int(arg)


    detector = None
    if mode == 'wechat':
        detector = get_wechat_qr_detector()
    elif mode == 'dbr':
        detector = get_dbr_detector()
    print("DBR DONE.....")

    print("STARTING GENERATION........")

    file_base = os.path.basename(os.path.join(os.getcwd(), inputfile))
    parent_path = "datasets/dataset_{}_({})".format(os.path.splitext(file_base)[0] ,datetime.datetime.now().strftime("%Y-%m-%d_(%H_%M_%S)"))
    in_dir = 'dataset_images/input'
    out_dir = 'dataset_images/output'
    masked_dir = 'dataset_images/masked'

    parent_path = os.path.join(os.getcwd(), parent_path)
    in_dir = os.path.join(parent_path, in_dir)
    out_dir = os.path.join(parent_path, out_dir)
    masked_dir = os.path.join(parent_path, masked_dir)

    os.mkdir(parent_path)
    os.makedirs(in_dir)
    os.makedirs(out_dir)
    os.makedirs(masked_dir)

    video_file = inputfile
    cap = cv.VideoCapture(video_file)
    csv.field_size_limit(int(sys.maxsize / 100000000000))
    with open(parent_path + "/image_dataset.csv", "w") as csvFile:

        writer = csv.DictWriter(csvFile, fieldnames=["IMAGE_ID", "SCALE", "CLAHE_CLIPLIMIT", "CLAHE_GRIDSIZE", "BLUR",
                                                     "AT_BLOCKSIZE", "AT_SUBTRACTED", "OUTPUT_IMAGE_ID", "DETECTED",
                                                     "CODE_TEXT", "CODE_DIM", "OUT_IMAGE_DIM", "MASKED_OUTPUT_ID"])
        writer.writeheader()
        while cap.isOpened():
            ret, frame = cap.read()
            i += 1
            if i % fpr != 0:
                continue
            if ret:
                cv.imshow("Frame", OUTPUT_IMAGE if OUTPUT_IMAGE is not None else frame)
                cv.moveWindow("Frame", 0, 0)
                cv.resizeWindow("Frame", 432, 288)
                generate_transforms_full_space(frame, writer, parent_path, mode=mode, detector=detector, id=id)
            else:
                break
            if (cv.waitKey(1) & 0xFF) == ord("q"):
                exit(0)
                break

        cv.destroyAllWindows()
        cap.release()
    print("SPLITTING FOLDERS............")
    splitfolders.ratio(str(parent_path) + "/dataset_images", output=str(parent_path) + "/dataset_images_ml",
                       ratio=(.4, .3, .3))

    print("DATASET GENERATE NAVIGATE TO {}".format(parent_path))
    del detector


if __name__ == "__main__":
    main(sys.argv[1:])
