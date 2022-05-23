from dbr import *

# one day license
licenseKey = "DLS2eyJoYW5kc2hha2VDb2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6IjIwMDAwMSIsInNlc3Npb25QYXNzd29yZCI6IndTcGR6Vm05WDJrcEQ5YUoifQ== "

BarcodeReader.init_license(licenseKey)

ipModesCount = 2
binModesCount = 3
locModesCount = 2

tpDecoding = "templates/Decoding.json"
tpDecodingBinImage = "templates/DecodingBinImage.json"
tpHead = "{\"Version\":\"3.0\", \"ImageParameterContentArray\" : [{ \"Name\": \"Test\", \"MaxAlgorithmthreadCount\" : " \
         "4, \"ExpectedBarcodesCount\" : 0, \"ScaleDownThreshold\" : 700, "
intermediateResultTypes = [
    "\"IntermediateResultTypes\" : [\"IRT_PREPROCESSED_IMAGE\"],",
    "\"IntermediateResultTypes\" : [\"IRT_BINARIZED_IMAGE\"],",
    "\"IntermediateResultTypes\" : [\"IRT_TYPED_BARCODE_ZONE\"],"
]

ipModes = [
    "\"ImagePreprocessingModes\": [{\"Mode\": \"IPM_SHARPEN_SMOOTH\",\"SharpenBlockSizeX\" : 3,\"SharpenBlockSizeY\" "
    ": 3,\"SmoothBlockSizeX\" : 5,\"SmoothBlockSizeY\" : 5}],",
    "\"ImagePreprocessingModes\": [{\"Mode\": \"IPM_SHARPEN_SMOOTH\",\"SharpenBlockSizeX\" : 3,\"SharpenBlockSizeY\" "
    ": 3,\"SmoothBlockSizeX\" : 3,\"SmoothBlockSizeY\" : 3}],",
    "\"ImagePreprocessingModes\": [{\"Mode\": \"IPM_SKIP\"}],"
]

binModes = [
    "\"BinarizationModes\": [{\"Mode\": \"BM_LOCAL_BLOCK\", \"BlockSizeX\" : 30, \"BlockSizeY\" : 30, "
    "\"EnableFillBinaryVacancy\" : 0, \"ThreshValueCoefficient\" : 1}],",
    "\"BinarizationModes\": [{\"Mode\": \"BM_LOCAL_BLOCK\", \"BlockSizeX\" : 15, \"BlockSizeY\" : 15, "
    "\"EnableFillBinaryVacancy\" : 0, \"ThreshValueCoefficient\" : 0}],",
    "\"BinarizationModes\": [{\"Mode\": \"BM_LOCAL_BLOCK\", \"BlockSizeX\" : 8, \"BlockSizeY\" : 8, "
    "\"EnableFillBinaryVacancy\" : 0, \"ThreshValueCoefficient\" : 1}], "
]

locModes = [
    "\"LocalizationModes\": [\"LM_CONNECTED_BLOCKS\"],",
    "\"LocalizationModes\": [\"LM_SCAN_DIRECTLY\"],"
]

terminatePhases = [
    "\"TerminatePhase\" : \"TP_IMAGE_PREPROCESSED\",",
    "\"TerminatePhase\" : \"TP_IMAGE_BINARIZED\",",
    "\"TerminatePhase\" : \"TP_BARCODE_TYPE_DETERMINED\","
]

tpFoot = "\"TextureDetectionModes\": [{ \"Mode\": \"TDM_SKIP\" }], \"TimeOut\" : 100000 }]}"

reader1 = BarcodeReader()
reader21 = BarcodeReader()
reader22 = BarcodeReader()
reader23 = BarcodeReader()
reader3 = BarcodeReader()


def dbr_decode(frame):
    try:
        # ============== Step 1: check if need Convert colour ===============
        reader1.decode_buffer(frame)
        grayscaleTransformationMode = -1
        interMResults = reader1.get_all_intermediate_results()
        if interMResults != None:
            original = 0
            invert = 0
            for im in interMResults:
                if im.result_type == EnumIntermediateResultType.IRT_TYPED_BARCODE_ZONE:
                    if im.grayscale_transformation_mode == EnumGrayscaleTransformationMode.GTM_ORIGINAL:
                        original = original + 1
                    else:
                        invert = invert + 1
            if original > invert:
                grayscaleTransformationMode = EnumGrayscaleTransformationMode.GTM_ORIGINAL
            elif invert > original:
                grayscaleTransformationMode = EnumGrayscaleTransformationMode.GTM_INVERTED
        # ============== Step 2: generate intermediate result ===============
        success = False
        gtmMode = ""
        for gtmi in range(2):
            if success:
                break
            if grayscaleTransformationMode == EnumGrayscaleTransformationMode.GTM_ORIGINAL:
                if gtmi > 0:
                    break
                gtmMode = "\"GrayscaleTransformationModes\": [{\"Mode\": \"GTM_ORIGINAL\"}],"
            elif grayscaleTransformationMode == EnumGrayscaleTransformationMode.GTM_INVERTED:
                if gtmi > 0:
                    break
                gtmMode = "\"GrayscaleTransformationModes\": [{\"Mode\": \"GTM_INVERTED\"}],"
            else:
                if gtmi == 0:
                    gtmMode = "\"GrayscaleTransformationModes\": [{\"Mode\": \"GTM_ORIGINAL\"}],"
                else:
                    gtmMode = "\"GrayscaleTransformationModes\": [{\"Mode\": \"GTM_INVERTED\"}],"

            # imagePreProcessingModes loop
            settingsT = tpHead + gtmMode
            for ipmi in range(ipModesCount):
                if success:
                    break
                settingsT2 = settingsT + ipModes[ipmi]
                settingsGetGrayImg = settingsT2 + terminatePhases[0] + intermediateResultTypes[0] + tpFoot
                error = reader21.init_runtime_settings_with_string(settingsGetGrayImg)
                if error[0] != EnumErrorCode.DBR_OK:
                    # print("ImagePreprocessingMode:init Runtime Failed:", error[1])
                    continue
                reader21.decode_buffer(frame)

                # binarizationModes loop
                IMRA21 = reader21.get_all_intermediate_results()
                if IMRA21 is not None:
                    for bini in range(binModesCount):
                        if success:
                            break
                        settingsT3 = settingsT2 + binModes[bini]
                        settingsGetBinImg = settingsT3 + terminatePhases[1] + intermediateResultTypes[1] + tpFoot
                        error = reader22.init_runtime_settings_with_string(settingsGetBinImg)
                        if error[0] != EnumErrorCode.DBR_OK:
                            # print("BinarizationMode:init Runtime Failed:", error[1])
                            continue

                        reader22.decode_intermediate_results(IMRA21)

                        IMRA22 = reader22.get_all_intermediate_results()
                        if IMRA22 is not None:
                            noIpmSettings = settingsT + ipModes[2] + binModes[bini]
                            for loci in range(locModesCount):
                                if success:
                                    break
                                settingsT4 = noIpmSettings + locModes[loci]
                                settingsGetZone = settingsT4 + terminatePhases[2] + intermediateResultTypes[2] + tpFoot
                                error = reader23.init_runtime_settings_with_string(settingsGetZone)
                                if error[0] != EnumErrorCode.DBR_OK:
                                    # print("LocalizationMode:init Runtime Failed:", error[1])
                                    continue
                                reader23.decode_intermediate_results(IMRA22)
                                # ============== Step 3: decode intermediate results ===============
                                IMRA23 = reader23.get_all_intermediate_results()
                                if IMRA23 is not None:
                                    img = IMRA22[0].results[0]

                                    text_results = reader3.decode_buffer_manually(img.bytes, img.width, img.height,
                                                                                  img.stride, img.image_pixel_format)
                                    if text_results is not None:
                                        for result in text_results:
                                            continue
                                        return result.barcode_text
                                    break

    except BarcodeReaderError as bre:
        print(bre)
