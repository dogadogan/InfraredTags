"""
    AUTHOR: Ahmad Taka

"""
import cv2
import pyzbar.pyzbar as zbar
from image_transforms import transforms_v1


#########################################################################################
# Detectors
##########################################################################################

def is_pyzbar_qr(image):
    """
   :param image: image to be decoded
   :return: tuple(bool(is_detected), list(results))
   is_detected = True if detected false otherwise, result = (text, outline of code)
   """

    is_detected = False
    result = []
    image = transforms_v1(image)
    decodedObjects = zbar.decode(image, symbols=[zbar.ZBarSymbol.QRCODE])
    if not decodedObjects:
        decodedObjects = zbar.decode(255 - image,
                                     symbols=[zbar.ZBarSymbol.QRCODE])
    if decodedObjects is not None:
        is_detected = True
        for code in decodedObjects:
            (p1, p2, p3, p4) = code.polygon
            text, _ = code.data.decode("utf-8"), code.type
            coors = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)]
            result.append((text, coors))
    return is_detected, result
