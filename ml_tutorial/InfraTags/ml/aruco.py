import os
import cv2
import cv2.aruco as aruco
import numpy as np
from torchvision import transforms


def is_aruco_decodable(image):

     aruco_dict, parameters = aruco.Dictionary_get(aruco.DICT_4X4_50), aruco.DetectorParameters_create()
     _ , ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=parameters)

     return ids is not None

def decode_aruco(image):

    aruco_dict, parameters = aruco.Dictionary_get(aruco.DICT_4X4_50), aruco.DetectorParameters_create()
    _ , ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    return ids

def check_output_decode_rate(img_batch):

    decode = 0.0
    batch_size = img_batch.size(dim=0)
    t = transforms.ToPILImage(mode='L')
    
    for i in range(batch_size):
        img = t(img_batch[i,:,:,:])
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if is_aruco_decodable(img):
            decode += 1
    
    return decode

if __name__ == "__main__":
    
    pass

