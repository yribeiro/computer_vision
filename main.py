import cv2
import os

from camera_calib import CameraCalibration
from utils import find_chessboard_and_draw_cube, draw_cube_on_img


def main(cb_size, mtx, dist):
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            img = find_chessboard_and_draw_cube(frame, mtx, dist, cb_size)
            if img is not None:
                cv2.imshow("Cubed Feed!", img)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    break
        else:
            cv2.destroyAllWindows()

    # When everything done, release the capture
    cap.release()
    print("Exiting program ..")


if __name__ == "__main__":
    # chessboard dimensions
    cb_size = (8, 6)
    calib = CameraCalibration(cb_size=cb_size)
    calib.calibrate()
    folder = "calibration_images"
    for idx, img_path in enumerate(os.listdir(folder)):
        img = draw_cube_on_img(
            cv2.imread(os.path.join(folder, img_path)),
            calib.rvecs[idx], calib.tvecs[idx], calib.cam_calib_mat, calib.lens_dist
        )

        cv2.imshow("Cube", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    main(cb_size=cb_size, mtx=calib.cam_calib_mat, dist=calib.lens_dist)
