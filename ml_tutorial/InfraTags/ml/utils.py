import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from torch.nn.modules import module
from cnn import LocalizationModel, Unet
from torchvision import transforms
import torch
import torch.nn as nn
from qr import is_decodable
from dbr_qrcode import get_dbr_detector
from dataloader import get_dataloader_localization
import math


def find_bounding_box(coors):
    """
    Find the bounding box given four coordinates of a QR code.
    coors: list of 4 tuples (x,y) where each is a coordinate.
    Return the bounding box (x,y,w,h) where (x,y) is the top left coordinate, w is the width,
        and h is the height.
    """

    max_x = max(coors[0][0], coors[1][0], coors[2][0], coors[3][0])
    min_x = min(coors[0][0], coors[1][0], coors[2][0], coors[3][0])
    max_y = max(coors[0][1], coors[1][1], coors[2][1], coors[3][1])
    min_y = min(coors[0][1], coors[1][1], coors[2][1], coors[3][1])
    width = max_x - min_x
    height = max_y - min_y
    return (min_x, min_y, width, height)


def iou(predicted, expected):
    """
    Find the intersection over union between predicted and expected bounding boxes.
    predicted: tensor of predicted bounding boxes (x,y,w,h) where (x,y) is the top left coordinate, w is the width,
        and h is the height.
    expected: tensor of expected bounding boxes (x,y,w,h) where (x,y) is the top left coordinate, w is the width,
        and h is the height.
    Return total iou of all pairs.
    """

    predicted = predicted.tolist()
    expected = expected.tolist()
    iou = 0

    for i in range(len(predicted)):
        x_p, y_p, w_p, h_p = predicted[i][0], predicted[i][1], predicted[i][2], predicted[i][3]
        x, y, w, h = expected[i][0], expected[i][1], expected[i][2], expected[i][3]

        if w <= 0 or h <= 0:
            return 0

        if x_p >= x + w or y_p >= y + h:
            return 0

        if x_p + w_p <= x or y_p + h_p <= y:
            return 0

        w_intersect = min(x_p + w_p, x + w) - max(x_p, x)
        h_intersect = min(y_p + h_p, y + h) - max(y_p, y)
        intersection = w_intersect * h_intersect

        iou += intersection / (w * h + w_p * h_p - intersection)

    return iou


def iou_image(predicted_batch, expected_batch):
    """
    Find the intersection over union between predicted and expected images.
    predicted_batch: tensor of predicted image
    expected: tensor of expected image
    Return total iou of all pairs.
    """

    predicted_batch = predicted_batch.detach()
    expected_batch = expected_batch.detach()
    batch_size = predicted_batch.size(dim=0)
    # print(batch_size)
    iou = 0.0

    for i in range(batch_size):
        predicted = predicted_batch[i, :, :, :]
        expected = expected_batch[i, :, :, :]
        union_white = torch.sum(torch.logical_or(predicted, expected)).item()
        overlap_white = torch.sum(predicted * expected).item()

        union_black = torch.sum(torch.logical_or(1 - predicted, 1 - expected)).item()
        overlap_black = torch.sum((1 - predicted) * (1 - expected)).item()

        # print((overlap_white/union_white + overlap_black/union_black)/2)
        iou += (overlap_white / union_white + overlap_black / union_black) / 2

    return iou


def capture_count(predicted, expected, margin=10):
    """
    Find the number of images that a QR code is inside the predicted bounding box plus margin.
    predicted: tensor of predicted bounding box (x,y,w,h) where (x,y) is the top left coordinate, w is the width,
        and h is the height.
    expected: tensor of expected bounding box (x,y,w,h) where (x,y) is the top left coordinate, w is the width,
        and h is the height.
    margin: margin to add on each side of a predicted bounding box
    """

    predicted = predicted.tolist()
    expected = expected.tolist()
    count = 0

    for i in range(len(predicted)):
        x_p, y_p, w_p, h_p = predicted[i][0], predicted[i][1], predicted[i][2], predicted[i][3]
        x, y, w, h = expected[i][0], expected[i][1], expected[i][2], expected[i][3]

        if x_p - margin <= x and x_p + w_p + margin >= x + w and y_p - margin <= y and y_p + h_p + margin >= y + h:
            count += 1

    return count


def check_bounding_box(predicted, expected, img, margin=10):
    """
    Draw the predicted (red) bounding box plus margin and the expected (blue) bounding box on img.
    predicted: tuple (x,y,w,h)
    expected: tuple (x,y,w,h)
    img: PIL Image object
    margin: margin to add on each side of a predicted bounding box
    """

    predicted = predicted.tolist()[0]
    predicted[0] = int(predicted[0])
    predicted[1] = int(predicted[1])
    predicted[2] = round(predicted[2])
    draw = ImageDraw.Draw(img)
    draw.rectangle([(expected[0], expected[1]), (expected[0] + expected[2], expected[1] + expected[2])], outline='blue',
                   width=1)
    draw.rectangle([(predicted[0] - margin, predicted[1] - margin),
                    (predicted[0] + predicted[2] + margin, predicted[1] + predicted[2] + margin)], outline='red',
                   width=1)
    img.show()


def draw_bounding_box(img, bb):
    draw = ImageDraw.Draw(img)
    draw.rectangle([(bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3])], outline='blue', width=1)
    img.show()


def is_inside_box(x_box, y_box, box_size, x, y, l):
    """
    Check if a square with (x,y) as a top left corner, and size l is inside a box
    with (x_box, y_box) as a top left corner, and size box_size.
    """
    return x >= x_box and y >= y_box and x + l <= x_box + box_size and y + l <= y_box + box_size


def check_binarization_result(model, img_file, box, mode, detector):
    t = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    sigmoid = nn.Sigmoid()

    image = Image.open(img_file)
    crop_tuple = (box[0] - 10, box[1] - 10, box[0] + box[2] + 10, box[1] + box[2] + 10)
    if box[0] - 10 < 0 or box[1] - 10 < 0 or box[0] + box[2] + 10 >= 288 or box[1] + box[2] + 10 >= 288:
        image = ImageOps.invert(image)
        image = image.crop(crop_tuple)
        image = ImageOps.invert(image)
    else:
        image = image.crop(crop_tuple)
    image = image.resize((224, 224))
    image_tensor = t(image)
    image_tensor = torch.reshape(image_tensor, (1, image_tensor.size(dim=0), \
                                                image_tensor.size(dim=1), image_tensor.size(dim=2)))
    output = model(image_tensor)
    # output = torch.round(sigmoid(output))
    output = sigmoid(output)
    # print(output)
    output = torch.round(output)
    output = to_pil(output[0, :, :, :])
    print(img_file)
    print(is_decodable(output, mode, detector))
    print("--------------")
    # if not is_decodable(output, mode, detector):
    #     output.show()
    # output.show()


def check_rescaled_decodability(img_file, box, mode, detector=None, size=224):
    image = Image.open(img_file).convert('RGB')
    width, height = image.size
    # crop_tuple = (box[0], box[1], box[0]+box[2], box[1]+box[2])
    crop_tuple = (box[0] - 10, box[1] - 10, box[0] + box[2] + 10, box[1] + box[2] + 10)
    # crop_tuple = (max(0,box[0]-10), max(0,box[1]-10), min(width, box[0]+box[2]+10), min(height,box[1]+box[2]+10))
    # image = ImageOps.invert(image)

    # if box[0] < 0 or box[1]< 0 or box[0]+box[2] >= width or box[1]+box[2] >= height:
    if box[0] - 10 < 0 or box[1] - 10 < 0 or box[0] + box[2] + 10 >= width or box[1] + box[2] + 10 >= height:
        image = ImageOps.invert(image)
        image = image.crop(crop_tuple)
        image = ImageOps.invert(image)
        # image.show()
    else:
        image = image.crop(crop_tuple)

    # image = image.crop(crop_tuple)
    # image = ImageOps.invert(image)
    # image.show()

    image = image.resize((size, size))

    if not is_decodable(image, mode, detector):
        print(img_file)
        # image.show()

    return is_decodable(image, mode, detector) or is_decodable(ImageOps.invert(image), mode, detector)


def dir_decodability_rate(root, size=224, mode=None):
    table_file = root + "/image_dataset.csv"
    df = pd.read_csv(table_file)
    count = 0
    detector = None

    if mode == 'dbr':
        detector = get_dbr_detector()

    for i in range(len(df)):

        img_file = root + "/output/" + str(i) + ".png"
        bb = eval(df["bounding_box"].iloc[i])

        if check_rescaled_decodability(img_file, bb, mode, detector, size):
            count += 1

    print(count / len(df))
    del detector


def pad_white(img, width):
    return Image.fromarray(np.pad(np.array(img), width, 'constant', constant_values=255))


def pad_black(img, width):
    return Image.fromarray(np.pad(np.array(img), width, 'constant', constant_values=0))


def expected_cropped(img_file, box):
    image = Image.open(img_file).convert('RGB')
    width, height = image.size
    # crop_tuple = (box[0], box[1], box[0]+box[2], box[1]+box[2])
    crop_tuple = (box[0] - 10, box[1] - 10, box[0] + box[2] + 10, box[1] + box[2] + 10)
    # crop_tuple = (max(0,box[0]-10), max(0,box[1]-10), min(width, box[0]+box[2]+10), min(height,box[1]+box[2]+10))
    # image = ImageOps.invert(image)

    # if box[0] < 0 or box[1]< 0 or box[0]+box[2] >= width or box[1]+box[2] >= height:
    if box[0] - 10 < 0 or box[1] - 10 < 0 or box[0] + box[2] + 10 >= width or box[1] + box[2] + 10 >= height:
        image = ImageOps.invert(image)
        image = image.crop(crop_tuple)
        image = ImageOps.invert(image)
        # image.show()
    else:
        image = image.crop(crop_tuple)

    # image = image.crop(crop_tuple)
    # image = ImageOps.invert(image)
    # image.show()

    image = image.resize((224, 224))
    image.show()

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

if __name__ == "__main__":

    model_path = './weights_binarization/dbr_qr1479_gray/bestnet_15_decode.pt'
    in_channels = 1 # rgb
    out_channels = 1 # binary
    model = Unet(in_channels, out_channels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    decode_rate = 0.0
    acc = 0.0
    t = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    sigmoid = nn.Sigmoid()
    mode = 'dbr'
    detector = get_dbr_detector()

    im_file = './dataset_QR1479/test/input/198.png'
    img = cv2.imread(im_file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image_tensor = t(img_gray)
    print(image_tensor.size())
    image_tensor = torch.reshape(image_tensor, (1, image_tensor.size(dim=0), \
                        image_tensor.size(dim=1), image_tensor.size(dim=2)))

    output = model(image_tensor)
    output = torch.round(sigmoid(output))
    output = to_pil(output[0,:,:,:])
    output = np.array(output)
    # dbr detector needs 3 dimensions.
    if len(output.shape) == 2:
        output = np.stack((output,)*3, axis=-1)
    if is_decodable(output, mode, detector):
        print('yes')
        decoded = detector.decode_buffer(output)
        text = decoded[0].barcode_text
        coors = decoded[0].localization_result.localization_points
        cv2.polylines(output, [np.array(coors)], True, (51,153,255), 2)
        draw_text(output, text, coors)

        cv2.imwrite('example2.png', output)
    

    pass