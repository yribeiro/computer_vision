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

    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _criteria)
    else:
        print("No chess board found ...")

    return ret, corners


def draw_cube_on_img(img, rvec, tvec, K, d):
    """
    Helper function to ingest an image path , along with the rodrigues rotation,
    translation vectors and the camera calibration matrix and distortion coefficients
    to draw a cube on the image.
    :param img: np array representing the image
    :param rvec: rodrigues rotation vector to get toimage plane
    :param tvec: translation vector to get to image plane
    :param K: Camera calibration matrix
    :param d: Camera distortion vector
    :return: Image (np.array()) with the projected cube
    """
    _3d_corners = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                              [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

    print(f"3D corners shape: {_3d_corners}")
    cube_corners_2d, _ = cv2.projectPoints(_3d_corners, rvec, tvec, K, d)
    print(f"2D corners shape: {cube_corners_2d}")

    red, blue, green = (0, 0, 255), (255, 0, 0), (0, 255, 0)  # red (in BGR)
    line_width = 3

    # first draw the base in red
    cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[1][0]), red, line_width)
    cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[2][0]), red, line_width)
    cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[3][0]), red, line_width)
    cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[0][0]), red, line_width)

    # now draw the pillars
    cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[4][0]), blue, line_width)
    cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[5][0]), blue, line_width)
    cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[6][0]), blue, line_width)
    cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[7][0]), blue, line_width)

    # finally draw the top
    cv2.line(img, tuple(cube_corners_2d[4][0]), tuple(cube_corners_2d[5][0]), green, line_width)
    cv2.line(img, tuple(cube_corners_2d[5][0]), tuple(cube_corners_2d[6][0]), green, line_width)
    cv2.line(img, tuple(cube_corners_2d[6][0]), tuple(cube_corners_2d[7][0]), green, line_width)
    cv2.line(img, tuple(cube_corners_2d[7][0]), tuple(cube_corners_2d[4][0]), green, line_width)

    return img


def find_chessboard_and_draw_cube(img, K, d, cb):
    """
    Helper function to ingest an image, find a chessboard on it and draw a cube if it exists
    :param img: Np array for the image
    :param K: Camera calibration matrix
    :param d: Camera distortion vector
    :param cb: Tuple containing the (col, row) for the chessboard
    :return: None if not chessboard, else the image with the cube on it
    """

    _subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    # get 3D image plane coords
    objp = np.zeros((cb[0] * cb[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:cb[0], 0:cb[1]].T.reshape(-1, 2)
    ret, corners = find_cb_points(img, cb)

    if ret:
        print("Found corners")
        _, rvec, tvec, _ = cv2.solvePnPRansac(objp, corners, K, d)
        # print values
        print(f"Found rvec: {rvec.tolist()}, tvec: {tvec.tolist()}")
        return draw_cube_on_img(img, rvec, tvec, K, d)
    else:
        return None
