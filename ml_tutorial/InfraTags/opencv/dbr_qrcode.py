from dbr import *


def get_dbr_detector():
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
    pass