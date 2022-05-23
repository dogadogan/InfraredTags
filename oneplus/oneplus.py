import datetime
import time
from multiprocessing import Process, Queue
import os
import cv2
import scrcpy
from adbutils import adb

from dbr_decode import dbr_decode

#################################################################
# Global Definitions
#################################################################
# Mutable
SCREEN_SIZE = (0, 0)
FRAME_BUFFER = None
DECODE_BUFFER = dict()
DECODE_TIME = 0
DECODE_FPS = 0
ACCURACY_BUFFER = []
ACCURACY = 0
# Const
MAX_ADB_WINDOW_FRAME = 350 #300 #480
ASPECT_RATIO = 19.8 / 9  # Height/Width
DISPLAY_WINDOW = True
VERBOSE_DECODE_HANDLE = True
VERBOSE_SOCKETIO = True


#################################################################
# Util
#################################################################


def decode_handler(frame, verbose=False):
    global DECODE_BUFFER, DECODE_TIME, DECODE_FPS, ACCURACY_BUFFER, ACCURACY
    if frame is not None:
        DECODE_TIME = time.time()
        # print(frame)
        msg = dbr_decode(frame)
        rct = []
        messages = []

        if msg is not None:
            messages.append({"RECT": rct, "MSG": msg})
            
            # command = "adb shell cmd notification post [flags] $(echo 'QR Code Detected: " + msg + "' | sed 's/ /\\\\ /g')"

            # Ahmad's old script:
            command = "adb shell cmd notification post [flags] '{}' ".format("QR Code Detected: " + msg)

            os.system(command)


        if verbose:
            print("Time: {}, Full Message: {}".format(datetime.datetime.now(), msg))
            print("Rect: {}, Message_{}: {}".format(rct, 0, msg))

        DECODE_BUFFER["MSGS"] = messages
        DECODE_BUFFER["S_SIZE"] = SCREEN_SIZE

        if verbose:
            if len(messages) == 0:
                ACCURACY_BUFFER.append(0)
            else:
                ACCURACY_BUFFER.append(1)

            if len(ACCURACY_BUFFER) > 15:
                ACCURACY_BUFFER.pop(0)
            ACCURACY = round(float(sum(ACCURACY_BUFFER)) / len(ACCURACY_BUFFER), 4)
            DECODE_FPS = round(1 / (time.time() - DECODE_TIME), 4)
            print("DECODE_BUFFER: {}".format(DECODE_BUFFER))
            print("DECODE_FPS: {}, ACCURACY: {}, MSGS: {}, S_SIZE: {}".format(DECODE_FPS, ACCURACY,
                                                                              len(DECODE_BUFFER["MSGS"]), DECODE_BUFFER[
                                                                                  "S_SIZE"]))  # Proof that DECODE_BUFFER was edited
            print("###################################################################################")
    return


def on_frame_receive(frame):
    global FRAME_BUFFER, SCREEN_SIZE
    if frame is not None:
        FRAME_BUFFER = frame
        if DISPLAY_WINDOW:
            cv2.namedWindow("SCRCPY", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("SCRCPY", 362, 800)
            cv2.moveWindow("SCRCPY", 100, 50)
            cv2.imshow("SCRCPY", frame)
    cv2.waitKey(1)
    return


def handle_scrcpy(queue):
    print("STARTING SCRCPY......")
    # max-width really means height for vertically oriented phone

    client = scrcpy.Client(device=adb.device_list()[0], max_width=MAX_ADB_WINDOW_FRAME, max_fps=30, bitrate=2000000)
    client.add_listener(scrcpy.EVENT_FRAME, on_frame_receive)

    client.start(threaded=True)

    while True:
        try:
            if FRAME_BUFFER is not None:
                queue.put(FRAME_BUFFER)
        except Exception as e:
            print(str(e))
            break


def handle_decode(queue):
    while True:
        try:
            frame = queue.get()
            decode_handler(frame, verbose=VERBOSE_DECODE_HANDLE)
        except Exception as e:
            print(str(e))
            break


#################################################################
# Main
#################################################################
def main():
    print("\nStruct Codes DBR Decoder")

    q = Queue(maxsize=1)
    producer_branch = Process(target=handle_scrcpy, args=(q,))
    consumer_branch = Process(target=handle_decode, args=(q,))
    producer_branch.start()
    consumer_branch.start()


if __name__ == '__main__':
    main()
