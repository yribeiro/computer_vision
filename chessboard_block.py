import cv2


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

        # draw the chessboard
        cv2.drawChessboardCorners(img, cb_size, corners, ret)
        cv2.imshow("Chessboard", img)
        k = cv2.waitKey(1)

        if k == ord("q"):
            cv2.destroyAllWindows()
            print("Exiting program")
            exit(0)
    else:
        cv2.destroyAllWindows()
        print("No chess board found ...")

    return ret, corners


def main(cb_size):
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            _ = find_cb_points(frame, cb_size)

    # When everything done, release the capture
    cap.release()


if __name__ == "__main__":
    # chessboard dimensions
    main(cb_size=(6, 8))
