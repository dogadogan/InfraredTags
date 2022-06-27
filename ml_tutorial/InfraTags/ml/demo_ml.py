from time import time
import dbr
import cv2
import math
from matplotlib.contour import ContourSet
import torch
import random
import numpy as np
import torch.nn as nn
from torchvision import transforms
from cnn import LocalizationModel, Unet
from dbr_qrcode import get_dbr_detector
from qr import is_decodable
from aruco import is_aruco_decodable, decode_aruco

def draw_text(img, text, coors,
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=1,
          font_thickness=1,
          text_color=(0, 0, 0),
          text_color_bg=(51,153,255)
          ):

    x = int((max(coors[0][0], coors[1][0], coors[2][0], coors[3][0]) + min(coors[0][0], coors[1][0], coors[2][0], coors[3][0]))/2)
    y = max(coors[0][1], coors[1][1], coors[2][1], coors[3][1]) + 5
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x,y), (x + text_w, y + text_h), text_color_bg, cv2.FILLED)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size
# define the IR camera port for OpenCV VideoCapture 
cameraID = 0;

def main():
    cap = cv2.VideoCapture(cameraID)
    margin = 15
    while cap.isOpened():
        # read frame
        ret, frame = cap.read()
        # resize to 432x288
        width, height = int(frame.shape[1] * 0.4), int(frame.shape[0] * 0.4)
        image = cv2.resize(frame, (width, height))
        # extract the middle 288x288
        image = image[:, 112:400,:]
        cv2.imshow("Frame", image)
        cv2.moveWindow("Frame", 250, 250)
        cv2.resizeWindow("Frame", 288, 288)
        # turn image to tensor with correct dimension
        image_tensor = t(image)
        image_tensor = torch.reshape(image_tensor, (1, image_tensor.size(dim=0), \
                        image_tensor.size(dim=1), image_tensor.size(dim=2)))
        if ret:
            # top left coordinate and width and height
            x, y, w, h = l_model(image_tensor)[0,:]
            x, y = math.floor(x), math.floor(y)
            w, h = math.ceil(w), math.ceil(h)
            l = max(w,h)
            # crop
            crop_tuple = (x-margin, y-margin, x+l+margin, y+l+margin)
            left, top, right, bottom = 0, 0, 0, 0
            if crop_tuple[0] < 0:
                left = -crop_tuple[0]
            if crop_tuple[1] < 0:
                top = -crop_tuple[1]
            if crop_tuple[2] > 288:
                right = crop_tuple[2] - 288
            if crop_tuple[3] > 288:
                bottom = crop_tuple[3] - 288
            padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255,255,255])
            image = padded_image[crop_tuple[0]+left:crop_tuple[2]+left+right, crop_tuple[1]+top:crop_tuple[3]+top+bottom]
            
            # resize to 244x244
            image = cv2.resize(image, (224, 224))
            cv2.imshow("local", image)
            cv2.moveWindow("local", 250, 250)
            cv2.resizeWindow("local", 224, 224)
            # turn image to tensor with correct dimension
            image_tensor = t(image)
            image_tensor = torch.reshape(image_tensor, (1, image_tensor.size(dim=0), \
                        image_tensor.size(dim=1), image_tensor.size(dim=2)))
            output_b = b_model(image_tensor)
            output_b = torch.round(sigmoid(output_b))
            output_b = to_pil(output_b[0,:,:,:])
            output_b = np.array(output_b)
            # dbr detector needs 3 dimensions.
            cv2.imshow("output", output_b)
            cv2.moveWindow("output", 0, 400)
            cv2.resizeWindow("output", 224, 224)
            if is_aruco_decodable(output_b):
                text_results = detector.decode_buffer(output_b)
                print(random.random(), ' Text:', text_results[0].barcode_text)

        if cv2.waitKey(10) & 0xFF == ord('q') :
            # break out of the while loop
            break

    cv2.destroyAllWindows()
    cap.release()

def main2():
    cap = cv2.VideoCapture(cameraID)
    while cap.isOpened():
        # read frame
        start = time()
        ret, frame = cap.read()
        height = 288
        width = int(frame.shape[1] * 288 / frame.shape[0])
        # resize to 432x288
        image = cv2.resize(frame, (width, height))
        # extract the middle 288x288
        image = image[:, 112:400, :]
        # print(image.shape)
        cv2.imshow("Frame", image)
        cv2.moveWindow("Frame", 0, 0)
        cv2.resizeWindow("Frame", 288, 288)
        # turn image to tensor with correct dimension
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = t(image)
        image_tensor = torch.reshape(image_tensor, (1, image_tensor.size(dim=0), \
                        image_tensor.size(dim=1), image_tensor.size(dim=2)))
        if ret:
            output_b = b_model(image_tensor)
            output_b = torch.round(sigmoid(output_b))
            output_b = to_pil(output_b[0,:,:,:])
            output_b = np.array(output_b)
            # dbr detector needs 3 dimensions.
            if len(output_b.shape) == 2:
                output_b = np.stack((output_b,)*3, axis=-1)
            if is_decodable(output_b, mode, detector):
                decoded = detector.decode_buffer(output_b)
                text = decoded[0].barcode_text
                coors = decoded[0].localization_result.localization_points
                cv2.polylines(output_b, [np.array(coors)], True, (51,153,255), 2)
                draw_text(output_b, text, coors)
                print('Text:', text)
                end = time()
                print('Time:', end-start)
                print('-------------')
            cv2.imshow("output", output_b)
            cv2.moveWindow("output", 0, 320)
            cv2.resizeWindow("output", 288, 288)

        if cv2.waitKey(10) & 0xFF == ord('q') :
            # break out of the while loop
            break

    cv2.destroyAllWindows()
    cap.release()

def main3():
    cap = cv2.VideoCapture(cameraID)
    while cap.isOpened():
        # read frame
        start = time()
        ret, frame = cap.read()
        height = 288
        width = int(frame.shape[1] * 288 / frame.shape[0])
        # resize to 432x288
        image = cv2.resize(frame, (width, height))
        # extract the middle 288x288
        image = image[:, 112:400, :]
        # print(image.shape)
        cv2.imshow("Frame", image)
        cv2.moveWindow("Frame", 250, 0)
        cv2.resizeWindow("Frame", 288, 288)
        # turn image to tensor with correct dimension
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = t(image)
        image_tensor = torch.reshape(image_tensor, (1, image_tensor.size(dim=0), \
                        image_tensor.size(dim=1), image_tensor.size(dim=2)))
        if ret:
            output_b = b_model(image_tensor)
            output_b = torch.round(sigmoid(output_b))
            output_b = to_pil(output_b[0,:,:,:])
            output_b = np.array(output_b)
            # dbr detector needs 3 dimensions.

            if is_aruco_decodable(output_b):
                id, corner = decode_aruco(output_b)
                print('ID:', id)
                print('Corner:', corner)
                end = time()
                print('Time:', end-start)
                print('-------------')
            cv2.imshow("output", output_b)
            cv2.moveWindow("output", 250, 500)
            cv2.resizeWindow("output", 288, 288)

        if cv2.waitKey(10) & 0xFF == ord('q') :
            # break out of the while loop
            break

    cv2.destroyAllWindows()
    cap.release()



if __name__ == "__main__":
    # batch_size = 16
    # model_path = './weights_localization/resnet18/bestnet_74_cr.pt'
    # l_model = LocalizationModel()
    # l_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # l_model.eval()

    # model_path = './weights_binarization/dbr_2_masked/bestnet_41_decode.pt'
    # in_channels = 3 # rgb
    # out_channels = 1 # binary
    # b_model = Unet(in_channels, out_channels)
    # b_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # b_model.eval()
    # mode = 'dbr'
    # detector = get_dbr_detector()

    # t = transforms.ToTensor()
    # to_pil = transforms.ToPILImage()
    # sigmoid = nn.Sigmoid()

    # main()

    model_path = './weights_binarization/ar_trial_288/bestnet_30_decode.pt'
    in_channels = 3 # rgb
    out_channels = 1 # binary
    b_model = Unet(in_channels, out_channels)
    b_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    b_model.eval()
    mode = 'dbr'
    detector = get_dbr_detector()

    t = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    sigmoid = nn.Sigmoid()

    main3()
 
