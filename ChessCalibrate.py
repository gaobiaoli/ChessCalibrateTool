import cv2
import glob
import numpy as np


class ChessCalibrate:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self, checkerboard: tuple):
        self.CHECKERBOARD = checkerboard
        self.imgpoints = []
        self.objpoints = []
        self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[
            0 : self.CHECKERBOARD[0], 0 : self.CHECKERBOARD[1]
        ].T.reshape(-1, 2)
        self.imgsize=None

    def addImg(self, img: np.array):
        if self.imgsize is None:
            self.imgsize=img.shape
        retval, corners = cv2.findChessboardCorners(img, self.CHECKERBOARD)
        if retval:
            corners2 = cv2.cornerSubPix(
                img, corners, (11, 11), (-1, -1), self.criteria
            )
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners2)

    def calibrate(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.imgsize[::-1], None, None
        )
        return ret, mtx, dist

    def __len__(self):
        return len(self.imgpoints)


if __name__ == "__main__":
    CHECKERBOARD = (17, 17)
    tool = ChessCalibrate(CHECKERBOARD)
    imglist = glob.glob("./img/*.bmp")
    for img in imglist:
        gray = cv2.imread(img, 0)
        tool.addImg(gray)
        print("Image:",len(tool))
    ret, mtx, dist = tool.calibrate()
    np.savez("./distort", mtx=mtx, dist=dist)
    # Read Saved File
    # distort = np.load("distort.npz")
    # print(distort['mtx'])
    # print(distort['dist'])
