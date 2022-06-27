import cv2 as cv
import numpy as np

##########################################################################################################################################
# IMAGE TRANSFORMATIONS
##########################################################################################################################################
DEFAULT_PARMAS = {"SCALE_IN": 40,
                  "CLIPLIMIT_IN": 20,
                  "GRIDSIZE_IN": 8,
                  "BLUR_IN": 7,
                  "BLOCKSIZE_IN": 101,
                  "SUBTRACTED_IN": 4}


def transforms_v1(image, params_dict=DEFAULT_PARMAS):
    """
    :param image: OpenCV image frame
    :param params_dict: dictionary containing the parameters
    :param invert: if true output image is inverted
    :return: transformed image
    """
    (width, height) = int(image.shape[1] * params_dict["SCALE_IN"] / 100), int(
        image.shape[0] * params_dict["SCALE_IN"] / 100)
    resize = cv.resize(image, (width, height))
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(params_dict["CLIPLIMIT_IN"], (params_dict["GRIDSIZE_IN"], params_dict["GRIDSIZE_IN"])).apply(
        gray)
    blur = cv.GaussianBlur(clahe, (params_dict["BLUR_IN"], params_dict["BLUR_IN"]), 0)
    adaptive_threshold = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                              params_dict["BLOCKSIZE_IN"],
                                              params_dict["SUBTRACTED_IN"])
    # integral_threshold = thresholdIntegral(blur, cv.integral(blur))
    output = adaptive_threshold  # change this value to output different transforms
    return output


##########################################################################################################################################
# CUSTOM TRANSFORMATIONS
##########################################################################################################################################

def thresholdIntegral(inputMat, s, T=0.15):
    # outputMat=np.uint8(np.on@jit(nopython=True)es(inputMat.shape)*255)
    outputMat = np.zeros(inputMat.shape)
    nRows = inputMat.shape[0]
    nCols = inputMat.shape[1]
    S = int(max(nRows, nCols) / 8)

    s2 = int(S / 4)

    for i in range(nRows):
        y1 = i - s2
        y2 = i + s2

        if y1 < 0:
            y1 = 0
        if y2 >= nRows:
            y2 = nRows - 1

        for j in range(nCols):
            x1 = j - s2
            x2 = j + s2

            if x1 < 0:
                x1 = 0
            if x2 >= nCols:
                x2 = nCols - 1
            count = (x2 - x1) * (y2 - y1)

            add = s[y2][x2] - s[y2][x1] - s[y1][x2] + s[y1][x1]

            if int(inputMat[i][j] * count) < int(add * (1.0 - T)):
                outputMat[i][j] = 255
    return outputMat


def mask(image, polygon):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv.fillPoly(mask, [np.array(polygon)], (255, 255, 255))
    kern = np.ones((5,5), np.uint8)
    dilateted_mask = cv.dilate(mask,kernel=kern )
    mask = dilateted_mask
    just_qr = cv.bitwise_and(image, image, mask=mask)
    return cv.bitwise_or(just_qr, 255 - mask)

##########################################################################################################################################
# MAIN
##########################################################################################################################################

if __name__ == "__main__":
    pass
