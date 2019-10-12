import cv2
import os
from utils import find_cb_points


class CameraCalibration:
    def __init__(self, cb_size=(6, 8)):
        self.cb_size = cb_size

    def _capture_webcam_chessboard_image(self):
        """
        Function that returns a valid image with a chessboard pattern
        """
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret == True:
                cb_ret, corners = find_cb_points(frame, self.cb_size)
                if cb_ret == True:
                    break

        cap.release()
        return frame, corners

    def calibrate(self):
        key = ord("c")
        while key != ord("q"):
            img, corners = self._capture_webcam_chessboard_image()
            cv2.drawChessboardCorners(img, self.cb_size, corners, True)
            cv2.imshow("Chessboard", img)
            key = cv2.waitKey()
            cv2.destroyAllWindows()

        print("Exiting program")


if __name__ == "__main__":
    calib = CameraCalibration()
    calib.calibrate()
