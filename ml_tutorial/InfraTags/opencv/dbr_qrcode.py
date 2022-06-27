from dbr import *


def get_dbr_detector():
    reader = BarcodeReader()

    reader.init_license(
        "f0069fQAAAJPA8uit7wnzQOv0uYk4nYfOup8j/qmWQE9MczpNTsOn0U0XPx0qGIhf4PRYFs1lg1Uopyz8iA6Xa6Uuxq1zc415")  # expire: 2023-03-28
    # reader.init_runtime_settings_with_file("../dbr_params_level2.json")

    settings = reader.get_runtime_settings()
    settings.barcode_format_ids = EnumBarcodeFormat.BF_QR_CODE
    settings.barcode_format_ids_2 = EnumBarcodeFormat_2.BF2_NULL
    settings.expected_barcodes_count = 1
    settings.result_coordinate_type = EnumResultCoordinateType.RCT_PIXEL
    deblur_modes = [EnumDeblurMode.DM_BASED_ON_LOC_BIN, EnumDeblurMode.DM_THRESHOLD_BINARIZATION, \
                    EnumDeblurMode.DM_DIRECT_BINARIZATION, EnumDeblurMode.DM_SMOOTHING, \
                    EnumDeblurMode.DM_GRAY_EQUALIZATION, EnumDeblurMode.DM_MORPHING, EnumDeblurMode.DM_DEEP_ANALYSIS]
    for i in range(len(deblur_modes)):
        settings.deblur_modes[i] = deblur_modes[i]
    reader.update_runtime_settings(settings)

    return reader


if __name__ == "__main__":
    pass

    # reader = get_dbr_detector()
    # try:
    #    image = "./dataset/val/output/22.png"
    #    text_results = reader.decode_file(image)
    #    if text_results != None:
    #       for text_result in text_results:
    #          print("Barcode Format : " + text_result.barcode_format_string)
    #          if len(text_result.barcode_format_string) == 0:
    #             print("Barcode Format : " + text_result.barcode_format_string_2)
    #          else:
    #             print("Barcode Format : " + text_result.barcode_format_string)
    #          print("Barcode Text : " + text_result.barcode_text)
    #          print(text_result.localization_result.localization_points)
    # except BarcodeReaderError as bre:
    #    print(bre)

    # del reader

    # table_file = './dataset/train/image_dataset.csv'
    # df= pd.read_csv(table_file)
    # detector = get_dbr_detector()
    # print(len(df))
    # # start = time.time()
    # count = 0

    # for i in range(len(df)):
    #    img_file = "./dataset/train/output/"+ str(i) + ".png"
    #      # img = Image.open(img_file)
    #    box = eval(df["bounding_box"].iloc[i])
    #    # crop_tuple = (box[0], box[1], box[0]+box[2], box[1]+box[2])
    #    crop_tuple = (box[0]-10, box[1]-10, box[0]+box[2]+10, box[1]+box[2]+10)
    #    image = Image.open(img_file) #.convert('L')
    #    # image = cv2.imread(img_file)
    #    # image = np.array(image)
    #    # print(np.max(image))

    #    image = image.crop(crop_tuple)
    #    image = image.resize((224,224))
    #    image = np.dstack((image, image, image))
    #    # image = image.reshape((288,288,3))
    #    # print(image.shape)

    #    # crop_tuple = (box[0], box[1], box[0]+box[2], box[1]+box[2])
    #      # crop_tuple = (box[0]-10, box[1]-10, box[0]+box[2]+10, box[1]+box[2]+10)
    #    # image = image.crop(crop_tuple)
    #    # img = img[box[0]:box[0]+box[2]+1, box[1]:box[1]+box[2]+1]
    #      # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    #      # detector.detectAndDecode(img)
    #      # decode_qr(img)
    #    text_results = detector.decode_buffer(image)
    #    text_results_2 = detector.decode_buffer(255-image)
    #    if text_results != None or text_results_2 != None:
    #       count += 1

    # # end = time.time()
    # # print(end-start)
    # print(count/len(df))
    # del detector
