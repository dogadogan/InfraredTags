"""


"""

import datetime
import time
from multiprocessing import Process, Queue
from threading import Thread
import cv2
from aruco_decode import is_aruco

#################################################################
# Global Definitions
#################################################################
# Const
DISPLAY_WINDOW = True
VERBOSE_DECODE_HANDLE = True
VERBOSE_SOCKETIO = True
CAMERA_STREAM = 1  # 0 for default cam 1 for usb and [url] for wireless stream

#################################################################
# Util
#################################################################

# Mutable Globals
FRAME_BUFFER = None
SCREEN_SIZE = (0, 0)
DECODE_BUFFER = dict()
DECODE_TIME = 0
DECODE_FPS = 0
ACCURACY_BUFFER = []
ACCURACY = 0


def decode_handler(frame, verbose=False):
    global DECODE_BUFFER, DECODE_TIME, DECODE_FPS, ACCURACY_BUFFER, ACCURACY
    if frame is not None:
        DECODE_TIME = time.time()

        rct = []
        messages = []
        msg = is_aruco(frame)
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
                                                                                  "S_SIZE"]))  # Proof that
            # DECODE_BUFFER was edited
            print("###################################################################################")
    return


def handle_opencv():
    global FRAME_BUFFER
    cap = cv2.VideoCapture(CAMERA_STREAM)

    while True:
        ret, frame = cap.read()
        FRAME_BUFFER = frame
        if DISPLAY_WINDOW:
            cv2.namedWindow("ARUCO", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ARUCO", 640, 420)
            cv2.moveWindow("ARUCO", 100, 50)
            cv2.imshow("ARUCO", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            # break out of the while loop
            break

    cv2.destroyAllWindows()
    cap.release()


def handle_produce(queue):
    global FRAME_BUFFER
    print("OPENCV STARTING......")
    # max-width really means height for vertically oriented phone
    th = Thread(target=handle_opencv)
    th.start()
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
    print("\nInfrared Tags AruCo Decoder")
    q = Queue(maxsize=1)
    producer_branch = Process(target=handle_produce, args=(q,))
    consumer_branch = Process(target=handle_decode, args=(q,))
    producer_branch.start()
    consumer_branch.start()


if __name__ == '__main__':
    main()
