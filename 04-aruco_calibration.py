import numpy as np
import cv2
import argparse
import sys

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def calibrate_aruco(camera_calib_path, image_dir, image_name, image_format, aruco_type, square_size):
    [mtx, dist, R_co, R_oc, T_co, T_oc] = load_coefficients(camera_calib_path)
    arucoType = ARUCO_DICT[aruco_type]

    frame = capture_img(image_dir, image_name, image_format)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # verify that the supplied ArUCo tag exists and is supported by OpenCV
    if ARUCO_DICT.get(aruco_type, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(aruco_type))
        sys.exit(0)

    # load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
    print("[INFO] detecting '{}' tags...".format(aruco_type))
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    print("[INFO] Num of detected Tags: ",len(corners))

    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image
            cv2.putText(frame, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            print("[INFO] ArUco marker ID: {}".format(markerID))
    # show the output image
    cv2.imshow("Image", frame)
    cv2.waitKey(0)


def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    R_co = cv_file.getNode("R_co").mat()
    R_oc = cv_file.getNode("R_oc").mat()
    T_co = cv_file.getNode("T_co").mat()
    T_oc = cv_file.getNode("T_oc").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix, R_co, R_oc, T_co, T_oc]

def capture_img(image_dir, image_name, image_format):
    cam = cv2.VideoCapture(4)
    # cam.set(3,3840)
    # cam.set(4,2160)
    # cam.set(3,640)
    # cam.set(4,480)

    print("Hit SPACE key to capture, Hit ESC key to continue")

    img_name = image_dir + "/" + image_name + "." + image_format

    cv2.namedWindow("test")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            img = cv2.imread(img_name)
            break

        elif k%256 == 32:
            # SPACE pressed
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))

    cam.release()
    cv2.destroyAllWindows()
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Aruco calibration')

    parser.add_argument('--calib_file', type=str, required=True, help='YML file to read calibration matrices')
    parser.add_argument('--image_dir', type=str, required=True, help='image directory path')
    parser.add_argument('--image_format', type=str, required=True,  help='image format, png/jpg')
    parser.add_argument('--image_name', type=str, required=True, help='image name without extension or location')
    parser.add_argument('--aruco_type', type=str, required=True, default="DICT_5X5_50", help="type of ArUCo tag to detect")
    parser.add_argument('--square_size', type=float, required=True, help='aruco tag square size')
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')

    args = parser.parse_args()

    # mtx, dist, rvecs, tvecs =
    calibrate_aruco(args.calib_file, args.image_dir, args.image_name, args.image_format, args.aruco_type, args.square_size)
    # save_coefficients(mtx, dist, rvecs, tvecs, args.save_file)
    print("Aruco Calibration is finished.")