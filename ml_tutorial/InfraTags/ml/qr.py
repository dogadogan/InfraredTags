from pyzbar import pyzbar
from PIL import ImageOps 
from torchvision import transforms
import time
import numpy as np
from dbr_qrcode import get_dbr_detector
import os

def decode_qr(img):
    """
    Decode a QR code using pyzbar.
    """

    decoded = pyzbar.decode(img)

    if decoded:
        return decoded
    else:
        return pyzbar.decode(ImageOps.invert(img))

def is_decodable(img, mode, detector=None):
    """
    Check if the image is decodable.
    img: PIL Image object with QR code
    mode: library to be used. Either 'dbr' or 'pyzbar'
    detector: If mode is dbr, the detector should be passed to this function too.
    Return True if the image is decodable; otherwise, False.
    """
    
    if mode == 'dbr':
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
        return detector.decode_buffer(img) != None

    return len(pyzbar.decode(img)) != 0 or len(pyzbar.decode(ImageOps.invert(img))) != 0

def check_output_decodability(img_batch, mode='pyzbar', detector=None):
    """
    Find the number of output images that are decodable
    img_batch: tensor of images. 
    mode: library to be used. Either 'dbr' or 'pyzbar'
    detector: If mode is dbr, the detector should be passed to this function too.
    Return the the number of output images that are decodable.
    """

    decode = 0.0
    batch_size = img_batch.size(dim=0)
    t = transforms.ToPILImage(mode='L')
    
    for i in range(batch_size):
        img = t(img_batch[i,:,:,:])
        if is_decodable(img, mode, detector):
            decode += 1
    
    return decode

if __name__ == "__main__":
    
    # path = "../opencv/dataset_(2021-10-24_(15_06_15))/dataset_images_ml"
    # im_file = path + "/test/output/2_output.png"
    # im = Image.open(im_file)
    # print(decode_qr(im))

    # table_file = './dataset/train/image_dataset.csv'
    # df= pd.read_csv(table_file)
    detector = get_dbr_detector()
    # print(len(df))
    start = time.time()
    count = 0
    root = "../opencv/dataset_(2021-11-23_(14_23_24))/dataset_images_ml/train/output"
    files = os.listdir(root)
    # settings = detector.get_runtime_settings()
    # print(settings.__dict__)

    for file in files:
        # print(i)
        
        # img = Image.open(img_file)
        # img = cv2.imread(img_file)
        # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        # detector.detectAndDecode(img)
        # decode_qr(img)
        text_results = detector.decode_file(root+ '/' +file)
        if text_results != None:
            count += 1
        else:
            print(file)

    end = time.time()
    print(end-start)
    print(count/len(files))
    del detector
    

    