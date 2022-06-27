from distutils.command.install_egg_info import install_egg_info
from torch import _index_put_impl_
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import math
import os
import cv2
import pandas as pd
import numpy as np
from dbr_qrcode import get_dbr_detector
from aruco import is_aruco_decodable
from qr import is_decodable, decode_qr
from utils import find_bounding_box, draw_bounding_box, is_inside_box
from PIL import Image, ImageOps
from pathlib import Path
from aruco import is_aruco_decodable


def augment(input_img, output_img, coors, mode, qr, detector, size_wh, aruco):
    """
    Augment an image (Rotate, brightness, flipping, contrast, translation)
    input_img: PIL Image of input image (image from IR camera), expected size 432x288
    output_img: PIL Image of output image (decodable processed image), expected size 432x288
    coors: 4 coordinates of a qr code.
    """
    # T.RandomRotate()
    # T.ColorJitter()  /
    # T.RandomVertialFlip()   /
    # T.RandomHorizontalFlip()  /
    # translation
    input_img = input_img.resize(size_wh)
    output_img = output_img.resize(size_wh)
    size = size_wh[1]
    # input_width = 512
    bb = find_bounding_box(coors)
    
    augmented_inputs = []
    augmented_outputs = []
    augmented_box = []

    # choices for translation
    # input_width-size+1
    crop_choice = []
    for x in range(0, bb[0]+1 ,2):
        crop_tuple = (x, 0, x+size, size)
        cropped_output =  output_img.crop(crop_tuple)
        if x+size <= input_img.size[0]:
            if aruco:
                if is_aruco_decodable(cv2.cvtColor(np.array(cropped_output), cv2.COLOR_RGB2BGR)):
                    crop_choice.append(x)
            else:
                if is_decodable(cropped_output, qr, detector):
                    crop_choice.append(x)
    
    # translation
    crop_x = random.sample(crop_choice, min(3,len(crop_choice)))
    for x in crop_x:
        crop_tuple = (x, 0, x+size, size)
        augmented_inputs.append(input_img.crop(crop_tuple))
        cropped_output = output_img.crop(crop_tuple)
        augmented_outputs.append(cropped_output)
        augmented_box.append((bb[0]-x, bb[1], bb[2], bb[3]))
    
    # print("crop:", len(crop_x))

    if mode != "train":
        if len(augmented_inputs) != 0:
            return [augmented_inputs[0]], [augmented_outputs[0]], [augmented_box[0]]
        else:
            return [], [], []
    
    if len(crop_x) == 0:
        print("Fail")
        return [], [], []

    idx = random.choice([i for i in range(len(crop_x))])
    original_input = augmented_inputs[idx]
    original_output = augmented_outputs[idx]
    original_bb = augmented_box[idx]
    cutoff = crop_x[idx]
    new_coors = ((coors[0][0]-cutoff, coors[0][1]),(coors[1][0]-cutoff, coors[1][1]), \
        (coors[2][0]-cutoff, coors[2][1]),(coors[3][0]-cutoff, coors[3][1]))

    # vertical flip
    vflip_input, vflip_output, vflip_bb = vertical_flip(original_input, original_output, original_bb, size)
    augmented_inputs.append(vflip_input)
    augmented_outputs.append(vflip_output)
    augmented_box.append(vflip_bb)

    # horizontal flip
    hflip_input, hflip_output, hflip_bb = horizontal_flip(original_input, original_output, original_bb, size)
    augmented_inputs.append(hflip_input)
    augmented_outputs.append(hflip_output)
    augmented_box.append(hflip_bb)

    # brightness
    bright_input = adjust_brightness(original_input)
    augmented_inputs.append(bright_input)
    augmented_outputs.append(original_output)
    augmented_box.append(original_bb)

    # contrast
    contrast_input = adjust_contrast(original_input)
    augmented_inputs.append(contrast_input)
    augmented_outputs.append(original_output)
    augmented_box.append(original_bb)

    # hue
    hue_input = adjust_hue(original_input)
    augmented_inputs.append(hue_input)
    augmented_outputs.append(original_output)
    augmented_box.append(original_bb)

    # all aspects of color
    jitter_input = color_jitter(original_input, hue_flag=False)
    augmented_inputs.append(jitter_input)
    augmented_outputs.append(original_output)
    augmented_box.append(original_bb)

    jitter_input = color_jitter(original_input, hue_flag=True)
    augmented_inputs.append(jitter_input)
    augmented_outputs.append(original_output)
    augmented_box.append(original_bb)

    # rotation
    for i in range(2):
        rotate_input, rotate_output, rotate_bb = rotate(input_img, output_img, coors, size)
        if rotate_input is not None:
            augmented_inputs.append(rotate_input)
            augmented_outputs.append(rotate_output)
            augmented_box.append(rotate_bb)

    return augmented_inputs, augmented_outputs, augmented_box
    
def vertical_flip(input_img, output_img, bb, size=288):

    # vertical flip
    vflip_input = TF.vflip(input_img)
    vflip_output = TF.vflip(output_img)
    vflip_bb = (bb[0], max(0,size-bb[1]-bb[3]), bb[2], bb[3])

    return vflip_input, vflip_output, vflip_bb

def horizontal_flip(input_img, output_img, bb, size=288):

    # horizontal flip
    hflip_input = TF.hflip(input_img)
    hflip_output = TF.hflip(output_img)
    hflip_bb = (max(0,size-bb[0]-bb[2]), bb[1], bb[2], bb[3])

    return hflip_input, hflip_output, hflip_bb

def adjust_brightness(input_img):

    brightness_factor = random.uniform(0.5,1.5)
    return TF.adjust_brightness(input_img, brightness_factor=brightness_factor)

def adjust_contrast(input_img):

    contrast_factor = random.uniform(0.5,1.5)
    return TF.adjust_contrast(input_img, contrast_factor=contrast_factor)

def adjust_hue(input_img):

    hue_factor = random.uniform(-0.2,0.15)
    return TF.adjust_hue(input_img, hue_factor=hue_factor)

def color_jitter(input_img, hue_flag=False):

    transforms_color = T.ColorJitter(brightness=0.5, contrast=0.5)
    if hue_flag:
        transforms_color = T.ColorJitter(brightness=0.5, contrast=0.5, hue=(-0.2,0.15))
    return transforms_color(input_img)

def rotate(input_img, output_img, coors, size=288):
    
    angle = random.randint(-180,180)
    center = (round(input_img.size[0]/2), round(input_img.size[1]/2))
    rotate_input = TF.rotate(input_img, angle, expand=True, center=center, fill=255)
    rotate_output = TF.rotate(output_img, angle, expand=True, center=center, fill=255)

    # get new coordinates of a qr code after rotation
    new_coors_0 = rotate_point(coors[0], angle, center)
    new_coors_1 = rotate_point(coors[1], angle, center)
    new_coors_2 = rotate_point(coors[2], angle, center)
    new_coors_3 = rotate_point(coors[3], angle, center)
    max_x = math.ceil(max(new_coors_0[0], new_coors_1[0], new_coors_2[0], new_coors_3[0]))
    min_x = math.floor(min(new_coors_0[0], new_coors_1[0], new_coors_2[0], new_coors_3[0]))
    max_y = math.ceil(max(new_coors_0[1], new_coors_1[1], new_coors_2[1], new_coors_3[1]))
    min_y = math.floor(min(new_coors_0[1], new_coors_1[1], new_coors_2[1], new_coors_3[1]))
    width = max_x - min_x
    height = max_y-min_y
    
    # get new coordinates of an image after rotation
    corner_0 = rotate_point((0,0), angle, center)
    corner_1 = rotate_point((0, input_img.size[1]), angle, center)
    corner_2 = rotate_point((input_img.size[0],0), angle, center)
    corner_3 = rotate_point((input_img.size[0], input_img.size[1]), angle, center)
    min_x_corner = math.floor(min(corner_0[0], corner_1[0], corner_2[0], corner_3[0]))
    min_y_corner = math.floor(min(corner_0[1], corner_1[1], corner_2[1], corner_3[1]))

    # adjust coordinates of a qr code
    min_x -= min_x_corner
    min_y -= min_y_corner

    choices = []
    for i in range(0, min_x+1, 12):
        for j in range(0, min_y+1, 12):
            if i + size <= rotate_input.size[0] and j + size <= rotate_input.size[1] and \
                 is_inside_box(i, j, size, min_x, min_y, max(width, height)):
                choices.append((i, j, i+size, j+size))
    
    choices_index = [i for i in range(len(choices))]
    if len(choices_index) == 0:
        return None, None, None
    idx = random.choice(choices_index)
    crop_tuple = choices[idx]
    rotate_input = rotate_input.crop(crop_tuple)
    rotate_output = rotate_output.crop(crop_tuple)
    min_x -= crop_tuple[0]
    min_y -= crop_tuple[1]

    new_bb = (min_x, min_y, width, height)

    # draw_bounding_box(rotate_input, (min_x, min_y, length))
    return rotate_input, rotate_output, new_bb

def rotate_point(pts, angle, center):

    c = math.cos(math.radians(angle))
    s = math.sin(math.radians(angle))
    new_x = center[0] + c*(pts[0] - center[0]) + s*(pts[1] - center[1])
    new_y = center[1] - s*(pts[0] - center[0]) + c*(pts[1] - center[1])
    return (new_x, new_y)



def save_augmentation_all(paths, table_files, types, save_dir, mode, size_wh, qr='pyzbar', aruco=False):

    assert len(paths) == len(table_files)
    assert len(paths) == len(types)
    detector = None

    if qr == 'dbr':
        detector = get_dbr_detector()

    for j in range(len(paths)):
        path = paths[j]
        table_file = table_files[j]
        qr_type = types[j]
        
        table = pd.read_csv(table_file)
    
        input_dir_path = os.path.join(path, mode, "input/")
        output_dir_path = os.path.join(path, mode, "masked/")

        input_data = os.listdir(input_dir_path)
        output_data = os.listdir(output_dir_path)

        input_data.sort()
        output_data.sort()

        assert len(input_data) == len(output_data)

        input_images = []
        output_images = []
        boxes = []

        for filename in input_data:
            id = int(filename.split('_')[0])
            print(id)
            input_img = Image.open(input_dir_path + filename)
            output_img = Image.open(output_dir_path + str(id) + "_masked_output.png").convert('RGB')
            coors = eval(table[table['IMAGE_ID'] == filename[:-4]].CODE_DIM.values[0])

            aug_inputs, aug_outputs, aug_bb = augment(input_img, output_img, coors, mode, qr, detector, size_wh, aruco)

            assert len(aug_inputs) == len(aug_outputs)
            assert len(aug_inputs) == len(aug_bb)
            
            input_images.extend(aug_inputs)
            output_images.extend(aug_outputs)
            boxes.extend(aug_bb)

        idx = 0
        save_df_file = Path(save_dir + "/" + mode + "/image_dataset.csv")
        if save_df_file.is_file():
            save_df = pd.read_csv(save_dir + "/" + mode + "/image_dataset.csv")
            idx = len(save_df)
   
        df = pd.DataFrame(columns = ['id', 'bounding_box', 'decodable', 'type'])

        for i in range(len(input_images)):
            decodable = 0
            
            if aruco:
                if is_aruco_decodable(cv2.cvtColor(np.array(output_images[i]), cv2.COLOR_RGB2BGR)):
                    decodable = 1
            else:
                if is_decodable(output_images[i], qr, detector):
                    decodable=1
            
            input_images[i].save(save_dir + "/" + mode + "/input/" + str(idx+i) + ".png", format="png")
            output_images[i].save(save_dir + "/" + mode + "/output/" + str(idx+i) + ".png", format="png")
            df.loc[i] = [idx+i, boxes[i], decodable, qr_type]
    
        if save_df_file.is_file():
            concatenated = pd.concat([save_df, df], ignore_index=True)
        else:
            concatenated = df
    
        concatenated.to_csv(save_dir + "/" + mode + "/image_dataset.csv", index=False)
    del detector

if __name__ == "__main__":

    size_wh = (144,96)

    save_dir = 'temp/'
    Path(save_dir + 'test/input').mkdir(parents=True, exist_ok=False)
    Path(save_dir + 'test/output').mkdir(parents=True, exist_ok=False)
    Path(save_dir + 'train/input').mkdir(parents=True, exist_ok=False)
    Path(save_dir + 'train/output').mkdir(parents=True, exist_ok=False)
    Path(save_dir + 'val/input').mkdir(parents=True, exist_ok=False)
    Path(save_dir + 'val/output').mkdir(parents=True, exist_ok=False)

    paths = [\
        "../opencv/datasets/dataset_AR1_bf0/dataset_images_ml",\
        "../opencv/datasets/dataset_AR1_bn0/dataset_images_ml",\
        "../opencv/datasets/dataset_AR1_wnf0/dataset_images_ml"
    ]

    table_files = [\
        "../opencv/datasets/dataset_AR1_bf0/image_dataset.csv",\
        "../opencv/datasets/dataset_AR1_bn0/image_dataset.csv",\
        "../opencv/datasets/dataset_AR1_wnf0/image_dataset.csv"
    ]

    types = [1,1,1]

    save_augmentation_all(paths, table_files, types, save_dir, 'test', size_wh, aruco=True)
    save_augmentation_all(paths, table_files, types, save_dir, 'train', size_wh, aruco=True)
    save_augmentation_all(paths, table_files, types, save_dir, 'val', size_wh, aruco=True)

    pass

        
    