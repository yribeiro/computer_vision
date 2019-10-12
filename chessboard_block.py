import cv2
import numpy as np


def find_cb_points(img, cb_size):
    """
    Function returns the corners for an image with a chessboard.
    :param img: Image with chessboard in it.
    :param cb_size: Size of the inner chessboard - tuple
    :return:
        (ret, corners) where ret is a boolean for points found in image and
        corners are the pixel coordinates for the chessboard points.
    """
    _criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, cb_size, None)

    if ret == True:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _criteria)
    else:
        cv2.destroyAllWindows()
        print("No chess board found ...")

    return ret, corners


def _draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
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


def main(cb_size):
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
                axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

                # Find the rotation and translation vectors.
                ret, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

                user_val = draw_box_and_chessboard(frame, cb_size, cb_ret, corners, imgpts)
                if user_val == ord("q"):
                    cv2.destroyAllWindows()
                    print("Exiting program ..")
                    exit(0)

    # When everything done, release the capture
    cap.release()


if __name__ == "__main__":
    # chessboard dimensions
    main(cb_size=(6, 8))
