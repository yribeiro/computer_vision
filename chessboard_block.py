import cv2
import numpy as np

from camera_calib import CameraCalibration
from utils import find_cb_points


def _draw(img, corners, imgpts):
    """
    Code taken from: https://docs.opencv.org/master/d7/d53/tutorial_py_pose.html
    :param img:
    :param corners:
    :param imgpts:
    :return:
    """
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img


def draw_box_and_chessboard(img, cb_size, ret, corners, img_pts_3d):
    """
    Function to ingest a chessboard image with a set of pixel coordinates
    representing the points and plot that image to screen.

    Function returns False if the user hits "q", which represents the program
    needing to quit.

    :param img: Image to display
    :param cb_size: size of the chessboard to draw
    :param corners: corners of the chessboard in pixel coords
    :param ret:
        Boolean value representing the presence of a chessboard in img, returned from
        cv2.findChessboardCorners()
    """
    # draw the chessboard
    cv2.drawChessboardCorners(img, cb_size, corners, ret)
    img = _draw(img, corners, img_pts_3d)
    cv2.imshow("Chessboard", img)
    k = cv2.waitKey(1)
    return k


def main(cb_size, mtx, dist):
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            cb_ret, corners = find_cb_points(frame, cb_size)

            if cb_ret == True:
                # create real world object points and associated real world axes based
                # based on chessboard configuration
                objp = np.zeros((cb_size[0] * cb_size[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:cb_size[1], 0:cb_size[0]].T.reshape(-1, 2)

                # create coords of cube
                axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                                   [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

                # Find the rotation and translation vectors.
                ret, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

                user_val = draw_box_and_chessboard(frame, cb_size, cb_ret, corners, imgpts)
                if user_val == ord("q"):
                    cv2.destroyAllWindows()
                    break
            else:
                cv2.destroyAllWindows()

    # When everything done, release the capture
    cap.release()
    print("Exiting program ..")


if __name__ == "__main__":
    # chessboard dimensions
    cb_size = (6, 8)
    calib = CameraCalibration(cb_size=cb_size)
    calib.calibrate()

    main(cb_size=cb_size, mtx=calib.cam_calib_mat, dist=calib.lens_dist)
