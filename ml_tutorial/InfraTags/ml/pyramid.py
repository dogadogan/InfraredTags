from gettext import find
from json import detect_encoding
from time import time
import cv2 
import os
from aruco import decode_aruco,  is_aruco_decodable
from qr import is_decodable, get_dbr_detector
from utils import draw_text
import numpy as np


detector = get_dbr_detector()


def pyramid(image_file, num):
    '''
    Generate image. Starting from the original image, the resolution of the image decrease by half each time.
    For example, 288x288 to 144x144.
    image_file: path to image file
    num: int. The number of images in the pyramid; the original one is included.
    '''

    image = cv2.imread(image_file)
    pyramid_image(image, num)

def pyramid_image(image, num):
    '''
    Generate image. Starting from the original image, the resolution of the image decrease by half each time.
    For example, 288x288 to 144x144.
    image: cv2 image
    num: int. The number of images in the pyramid; the original one is included.
    '''

    for _ in range(num):
        yield image
        image = cv2.pyrDown(image)

def decode_pyramid(image_file, params_list, num, aruco=True):
    '''
    Decode ArUco or QR code using image pyramid.
    image_file: path to image file
    params_list: List of parameters for filters. Have to be in order. [clipLimit, tileSize, kernelSize, blockSize, constant]
    num: int. The number of images in the pyramid; the original one is included.
    aruco: True for ArUco; False for QR.
    '''

    for image in pyramid(image_file, num):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(params_list[0], (params_list[1], params_list[1])).apply(gray)
        blur = cv2.GaussianBlur(clahe, (params_list[2], params_list[2]), 0)
        output = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                              params_list[3],
                                              params_list[4])
        if aruco:
            id, _ = decode_aruco(output)
            if id is None:
                id, _ = decode_aruco(255-output)
            if id is not None:
                return id
        else:
            return is_decodable(output, 'dbr', detector) or is_decodable(255-output, 'dbr', detector)

    return None


def test_pyramid_aruco(data_dir, params_lists, num):
    '''
    Test the pyramid method on an ArUco data directory.
    data_dir: path to data directory
    params_list: params_list: List of parameters for filters. Have to be in order. [clipLimit, tileSize, kernelSize, blockSize, constant]
    num: int. The number of images in the pyramid; the original one is included.
    '''

    files = os.listdir(data_dir)
    files = [f for f in files if not f.startswith('.')]
    files = sorted(files, key=lambda x: int(x[:-4]))
    count = 0
    start = time()
    for f in files:
        for params_list in params_lists:
            id = decode_pyramid(data_dir + "/" + f, params_list, num)
            if id is not None:
                count += 1
                # print(f)
                # print(id)
                # print('----')
                break
    end = time()
    print("time:", (end-start)/len(files))
    return count/len(files)

def test_pyramid_qr(data_dir, params_lists, num):
    '''
    Test the pyramid method on a QR data directory.
    data_dir: path to data directory
    params_list: params_list: List of parameters for filters. Have to be in order. [clipLimit, tileSize, kernelSize, blockSize, constant]
    num: int. The number of images in the pyramid; the original one is included.
    '''

    files = os.listdir(data_dir)
    files = sorted(files, key=lambda x: int(x[:-4]))
    count = 0
    start = time()
    for f in files:
        for params_list in params_lists:
            if decode_pyramid(data_dir + "/" + f, params_list, num, aruco=False):
                count += 1
                print(f)
                break
    end = time()
    print("time:", (end-start)/len(files))
    return count/len(files)

def test_param(video_file, params):

    cap = cv2.VideoCapture(video_file)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if i == 100:
            if ret:
                for param in params:
                    image = cv2.resize(frame, (432,288))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(param[0], (param[1], param[1])).apply(gray)
                    blur = cv2.GaussianBlur(clahe, (param[2], param[2]), 0)
                    output = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                              param[3], 0)
                    id = decode_aruco(output)
                    if id is None:
                        id = decode_aruco(255-output)
                    if id is not None:
                        print(param)
                break
        else:
            i += 1

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            exit(0)
            break

    cap.release()

def find_best_filter(data_dir, remaining_files=None, previous_best_params=[], n=1, filter_first=False):
    '''
    Find the n best filters that work well on a data directory using greedy algorithm. 
    Select the filter that detect most number of images, then exclude images detected by it. Then repeat.
    data_dir: path to directory
    remaining_files: reamining files that have not been detected. Always None for first call.
    previous_best_params: The best filters that have already been found. 
    n: number of filters to find
    filter_first: True if previous_best_param is not empty; otehrwise, False.
    '''

    if remaining_files == None:
        remaining_files = os.listdir(data_dir)
        remaining_files = sorted(remaining_files, key=lambda x: int(x[:-4]))

    if n == 0:
        return previous_best_params, len(remaining_files)
    
    if filter_first:
        detect = set()
        for f in remaining_files:
            for i in previous_best_params:
                id = decode_pyramid(data_dir + "/" + f, i)
                if id is not None:
                    detect.add(f)
                    break
        temp = []
        for f in remaining_files:
            if f not in detect:
                temp.append(f)
        remaining_files = temp

    params = []
    for i in [1,3,5,7]:
        for j in range(3,103, 2):
            if (4,8,i,j,0) not in previous_best_params:
                params.append([(4,8,i,j,0),0])
    
    for f in remaining_files:
        for i in range(len(params)):
            id = decode_pyramid(data_dir + "/" + f, params[i][0])
            if id is not None:
                params[i][1] += 1
    
    param_order = sorted(params, key=lambda x: x[1], reverse=True)
    best_params = None
    for i in param_order:
        if i[1] == param_order[0][1]:
            best_params = i[0]
    previous_best_params.append(best_params)

    detect = set()
    for f in remaining_files:
        id = decode_pyramid(data_dir + "/" + f, best_params)
        if id is not None:
            detect.add(f)

    temp_files = []
    for f in remaining_files:
        if f not in detect:
            temp_files.append(f)

    return find_best_filter(data_dir, temp_files, previous_best_params, n-1, filter_first=False)


cameraID = 0

def demo(filters, height, num, aruco=True):
    '''
    Demo.
    filters: List of List. The length of inner lists must be 5. 
    height: int. derised height of the image
    pyrdown_time: int. The number of times you scale down the image.
    num: int. The number of images in the pyramid; the original one is included.
    aruco: True if ArUco. False if QR.
    '''

    detector = get_dbr_detector()
    cap = cv2.VideoCapture('../opencv/movies_2/holo_AR3.mp4')
    # cap = cv2.VideoCapture(cameraID)
    while cap.isOpened():
        # read frame
        start = time()
        ret, frame = cap.read()

        width = int(frame.shape[1] * height / frame.shape[0])
        ori_image = cv2.resize(frame, (width, height))

        ori_image = ori_image[:, 115:403, :]
        # Select the middle square. You might have to change this line.
        
        cv2.imshow("Frame", ori_image)
        cv2.moveWindow("Frame", 250, 0)
        cv2.resizeWindow("Frame", width, height)

        if ret:
            for param in filters:
                for image in pyramid_image(ori_image, num):
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(param[0], (param[1], param[1])).apply(gray)
                    blur = cv2.GaussianBlur(clahe, (param[2], param[2]), 0)
                    output = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                              param[3], param[4])

                    readable = False
                    if aruco:
                        if is_aruco_decodable(255-output):
                            output = 255 - output
                            readable = True
                        elif is_aruco_decodable(output):
                            readable = True

                        if readable:
                            id, corner = decode_aruco(output)
                            print('ID:', id)
                            print('Corner:', corner)
                            end = time()
                            print('Time:', end-start)
                            print('-------------')
                            break

                    else:
                        if is_decodable(255-output, 'dbr', detector):
                            output = 255 - output
                            readable = True
                        elif is_decodable(output, 'dbr', detector):
                            readable = True

                        if readable:
                            decoded = detector.decode_buffer(output)
                            text = decoded[0].barcode_text
                            coors = decoded[0].localization_result.localization_points
                            cv2.polylines(output, [np.array(coors)], True, (51,153,255), 2)
                            draw_text(output, text, coors)
                            print('Text:', text)
                            break

                if readable:
                    break
 
            cv2.imshow("output", output)
            cv2.moveWindow("output", 250, 300)
            cv2.resizeWindow("output", width, height)

        if cv2.waitKey(10) & 0xFF == ord('q') :
            # break out of the while loop
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":

    param_holo = [(4, 8, 5, 5, 0), (4, 8, 3, 3, 0), (4, 8, 3, 13, 0), (4, 8, 7, 5, 0), (4, 8, 5, 3, 0)] #, (4, 8, 1, 3, 0), (4, 8, 7, 3, 0), (4, 8, 1, 7, 0)] #, (4, 8, 5, 37, 0), (4, 8, 1, 17, 0)])
    height = 288
    num = 1
    aruco_flag = True
    demo(param_holo, height, num, aruco_flag)

