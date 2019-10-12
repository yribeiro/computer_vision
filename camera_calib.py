import copy
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

    def capture_valid_images(self):
        key = None
        while key != ord("q"):
            img, corners = self._capture_webcam_chessboard_image()
            # create a copy for display purposes
            img_temp = copy.deepcopy(img)
            cv2.drawChessboardCorners(img_temp, self.cb_size, corners, True)
            cv2.imshow("Chessboard", img_temp)
            key = cv2.waitKey()
            cv2.destroyAllWindows()

            if key == ord("s"):
                # persist image
                rel_folder = os.path.join(os.getcwd(), "calibration_images")
                num_files = len(os.listdir(rel_folder))
                cv2.imwrite(os.path.join(rel_folder, f"calib_{num_files}.png"), img)

        print("Finished image capture stage")

    def calibrate(self):
        pass


if __name__ == "__main__":
    calib = CameraCalibration()
    calib.capture_valid_images()
