import copy
import cv2
import os

import numpy as np

from utils import find_cb_points


class CameraCalibration:
    def __init__(self, cb_size=(6, 8)):
        self.cb_size = cb_size

        # init necessary storage variables
        self.cam_calib_mat, self.lens_dist = np.zeros((3, 3)), np.zeros((4, 1))
        self.rvecs, self.tvecs = None, None

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

    def _get_points(self):
        """
        Function to get the object points in 3D chessboard plane coords and
        the corresponding image points on the board.

        Obtains images from the calibration_images folder in the cwd.

        :return: (obj_coords_list, img_coords_list) where obj_coords_list is a collection
        of the chessboard corner coordinates in 3D chessboard plane and the img_coords_list is a
        collection of the corresponding image coords in pixels.
        """
        files = os.listdir("calibration_images")

        # create real world object points and associated real world axes based
        # based on chessboard configuration
        objp = np.zeros((self.cb_size[0] * self.cb_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.cb_size[1], 0:self.cb_size[0]].T.reshape(-1, 2)

        obj_coords_list = []
        img_coords_list = []

        for fname in files:
            img = cv2.imread(os.path.join("calibration_images", fname))
            cb_ret, corners = find_cb_points(img, self.cb_size)

            if cb_ret == True:
                obj_coords_list.append(objp)
                img_coords_list.append(corners)

        return obj_coords_list, img_coords_list

    def calibrate(self, save_flag=False, save_loc=""):
        print("Starting calibration")

        obj_pts_list, img_pts_list = self._get_points()

        # get the size of a random image
        rimg = cv2.imread(os.path.join("calibration_images", os.listdir("calibration_images")[0]))
        shape = (rimg.shape[1], rimg.shape[0])
        rimg = None

        rms, self.cam_calib_mat, self.lens_dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            obj_pts_list, img_pts_list, shape, self.cam_calib_mat, self.lens_dist,
            None, None, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

        print("-"*20)
        print(f"Calibrated off {len(obj_pts_list)} images")
        print(f"DIM = {shape}")
        print(f"Camera Calibration Matrix = {self.cam_calib_mat}")
        print(f"Camera Lens Distortion = {self.lens_dist}")
        print(f"RMS = {rms}")

        if save_flag:
            np.savez(os.path.join(save_loc, "calib_mat"), self.cam_calib_mat, self.lens_dist)


if __name__ == "__main__":
    calib = CameraCalibration()
    calib.calibrate()
