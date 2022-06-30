from dbr import *


def get_dbr_detector():
    """
    Return a DBR QR code detector.
    """
    reader = BarcodeReader()
    reader.init_license("Download trial from https://www.dynamsoft.com/barcode-reader/downloads/")

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

    reader = get_dbr_detector()
    try:
        image = "./dataset_dbr_2_masked/val/output/106.png"
        text_results = reader.decode_file(image)
        print(text_results)
        if text_results != None:
            for text_result in text_results:
                print("Barcode Format : " + text_result.barcode_format_string)
                if len(text_result.barcode_format_string) == 0:
                    print("Barcode Format : " + text_result.barcode_format_string_2)
                else:
                    print("Barcode Format : " + text_result.barcode_format_string)
                print("Barcode Text : " + text_result.barcode_text)
                print(text_result.localization_result.localization_points)
    except BarcodeReaderError as bre:
        print(bre)

    del reader