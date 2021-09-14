import numpy as np
import cv2
import argparse
import sys
import pandas as pd
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def calibrate_aruco(intrinsic_calib_path, intrinsic_calib_path_undistorted, image_dir, image_name, image_format, aruco_tags_info_path):
    
    [mtx, dist, R_co, R_oc, T_co, T_oc] = load_coefficients(intrinsic_calib_path)
    [mtx_new, dist_new, R_co, R_oc, T_co, T_oc] = load_coefficients(intrinsic_calib_path_undistorted)
    
    frame = capture_img(image_dir, image_name, image_format)

    # try undistorted image
    # h,  w = frame.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    # # frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    frame = cv2.undistort(frame, mtx, dist, None, mtx)
    # # # crop the image
    # # x, y, w, h = roi
    # # frame = frame[y:y+h, x:x+w]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # read csv file that includes places of the aruco tags, their aruco type, ids, sizes and locations wrt their places
    df = pd.read_csv(aruco_tags_info_path)
    # get aruco dictionary types on the floor as a list
    aruco_types = list(df[df["place"]=="floor"]["aruco_type"].unique())

    corners_all = []
    ids_all = []
    sizes_all = []

    for aruco_type in aruco_types:
        arucoType = ARUCO_DICT[aruco_type]

        # verify that the supplied ArUCo tag exists and is supported by OpenCV
        if ARUCO_DICT.get(aruco_type, None) is None:
            print("[ERROR] ArUCo tag of '{}' is not supported".format(aruco_type))
            sys.exit(0)

        # load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
        print("[INFO] detecting '{}' tags...".format(aruco_type))
        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
        
        # only keep the detections that are on the floor by looking at the IDs
        ids_on_floor_with_current_aruco_type = list(df[(df["place"]=="floor") & (df["aruco_type"]==aruco_type)]["id"])
        if len(corners) > 0: # verify *at least* one ArUco marker was detected            
            for i, markerID in enumerate(list(ids.flatten())): # loop over the detected ArUCo corners
                if markerID in ids_on_floor_with_current_aruco_type:
                    corners_all.append(corners[i])
                    ids_all.append(markerID)
                    markerSize = float(df[(df["place"]=="floor") & (df["aruco_type"]==aruco_type) & (df["id"]==markerID)]["size_mm"])
                    sizes_all.append(markerSize)

        # print(corners_all)
        # print(ids_all)
        # print(sizes_all)

    print("[INFO] Num of detected Tags: ",len(corners_all))

    corners_all = np.array(corners_all)
    ids_all = np.array(ids_all)
    sizes_all = np.array(sizes_all)

    # verify *at least* one ArUco marker was detected on floor
    if len(corners_all) > 0:
        rvecs = []
        tvecs = []
        # loop over the detected ArUCo corners and draw  ids and bounding boxes around the detected markers
        for (markerCorner, markerID, markerSize) in zip(corners_all, ids_all, sizes_all):
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
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 1)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 1)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 1)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 1)
            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image
            cv2.putText(frame, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

            # Estimate the pose of the detected marker in camera frame
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorner, markerSize, mtx_new, dist_new)
            
            cv2.aruco.drawAxis(frame, mtx_new, dist_new, rvec, tvec, markerSize*0.75)  # Draw Axis

            # print("[INFO] ArUco marker ID: {}".format(markerID))
            # print(tvec[0].flatten()) # in camera's frame)
            # print(rvec[0].flatten()) # in camera's frame)
            rvecs.append(rvec[0])
            tvecs.append(tvec[0])

        rvecs = np.array(rvecs)
        tvecs = np.array(tvecs)

        # # Estimate the pose of the detected marker in camera frame at once 
        # rvecs, tvecs, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners_all, 190, mtx_new, dist_new)
        # cv2.aruco.drawDetectedMarkers(frame, corners_all, ids_all)  # Draw Bounding boxes
        # for (rvec, tvec) in zip(rvecs, tvecs):
        #     cv2.aruco.drawAxis(frame, mtx_new, dist_new, rvec, tvec, 190*0.75)  # Draw Axis
    
        # show the output image
        cv2.imshow("Image", frame)
        cv2.waitKey(0)

        # print(tvecs)
        # print(type(tvec))
        # print(type(tvecs))

        # Transform detected locations from camera frame to World frame
        tvecs = np.squeeze(tvecs).T # (3,N)
        tvecs = R_oc @ tvecs # (3,N)
        tvecs = T_oc + tvecs # (3,N)
        # print(np.shape(tvecs))

        # Calculate the best fitting plane
        origin = np.mean(tvecs,axis=1).reshape(3,1)
        tvecs2 = np.squeeze(tvecs - origin) #  (3, N)
        (U,S,Vt) = np.linalg.svd(tvecs2)
        normal_vec = U[:,-1].reshape(3,1)
        
        # Calculate residuals of the plane fitting
        distances = normal_vec.T @ tvecs2
        RMSE = math.sqrt(np.mean(np.square(distances)))
        # print("RMSE: ", RMSE, " (mm)")

        # Plot the data and fitting plane
        # plot data
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.scatter(tvecs[0,:], tvecs[1,:], tvecs[2,:], color='b')

        # plot plane
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        X,Y = np.meshgrid(np.arange(xlim[0], xlim[1], step=500),
                        np.arange(ylim[0], ylim[1], step=500))

        Z = -(normal_vec[0]/normal_vec[2])*(X-origin[0]) - (normal_vec[1]/normal_vec[2])*(Y-origin[1]) + origin[2] 

        ax.plot_wireframe(X,Y,Z, color='k')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

        ## Calculate the frame on the best fit plane based on world origin
        # Project the world origin onto the best fit plane
        origin_new = (normal_vec.T @ origin) * normal_vec
        # Projecy world frame x axis onto the origin 
        x = np.array([[1],[0],[0]])
        x_p = x - (normal_vec.T @ (x - origin) ) * normal_vec
        # calculate the corresponding x axis on the plane
        x_new = (x_p - origin_new) / np.sqrt(np.sum((x_p - origin_new)**2))
        # calculate the corresponding y axis on the plane
        y_new = np.cross(np.squeeze(normal_vec), np.squeeze(x_new)).reshape(3,1)
        # y_new = y_new / np.sqrt(np.sum(y_new**2)) # Normalization is not necessary since x_new and normal_vec are perpendicular

        # Define the rotation matrix from world frame to new plane frame
        R_op = np.concatenate((x_new, y_new,normal_vec), axis=1)
        R_po = R_op.T

        # Define the translation between world frame to plane frame
        T_op = origin_new
        T_po = -R_po @ T_op

        # As an example project all data points to the plane
        tvecs3 = R_po @ tvecs # (3,N)
        tvecs3 = T_po + tvecs3 # (3,N)
        tvecs3 = tvecs3[0:2,:]

        plt.figure()
        plt.title("2D Data")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.scatter(tvecs3[0,:], tvecs3[1,:], marker='x')
        plt.axis('equal')
        plt.grid()
        plt.show()

    return R_op, R_po, T_op, T_po, RMSE





def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    try:
        R_co = cv_file.getNode("R_co").mat()
        R_oc = cv_file.getNode("R_oc").mat()
        T_co = cv_file.getNode("T_co").mat()
        T_oc = cv_file.getNode("T_oc").mat()
    except:
        print("[INFO]: could not read R_co, R_oc, T_co, T_oc from: {}".format(path))
        print(str(R_co), str(R_oc), str(T_co), str(T_oc))
        cv_file.release()
        return [camera_matrix, dist_matrix]

    cv_file.release()
    return [camera_matrix, dist_matrix, R_co, R_oc, T_co, T_oc]

def capture_img(image_dir, image_name, image_format):
    cam = cv2.VideoCapture(1)
    cam.set(3,3840)
    cam.set(4,2160)
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


def save_coefficients(R_op, R_po, T_op, T_po, RMSE, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    
    # Rotation matrix 
    print("R_op: " + str(R_op))
    cv_file.write("R_op", R_op )

    print("R_po: " + str(R_po))
    cv_file.write("R_po", R_po)

    # Tranlastion vector
    cv_file.write("T_op", T_op)
    print("T_op: " + str(T_op))

    print("T_po: " + str(T_po))
    cv_file.write("T_po", T_po)

    print("RMSE: ", RMSE, " (mm)")
    cv_file.write("RMSE", RMSE)

    # note you *release* you don't close() a FileStorage object
    cv_file.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Aruco calibration')

    parser.add_argument('--calib_file', type=str, required=True, help='YML file to read calibration matrices')
    parser.add_argument('--calib_file_undistorted', type=str, required=True, help='YML file to read calibration matrices of undistorted images')
    parser.add_argument('--image_dir', type=str, required=True, help='image directory path')
    parser.add_argument('--image_format', type=str, required=True,  help='image format, png/jpg')
    parser.add_argument('--image_name', type=str, required=True, help='image name without extension or location')
    parser.add_argument('--aruco_tags_info_file', type=str, required=True, help="info csv file about the existing Aruco tags in the workspace")
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')

    args = parser.parse_args()
    

    R_op, R_po, T_op, T_po, RMSE = calibrate_aruco(args.calib_file, args.calib_file_undistorted,  args.image_dir, args.image_name, args.image_format, args.aruco_tags_info_file)
    save_coefficients(R_op, R_po, T_op, T_po, RMSE, args.save_file)
    print("Aruco Calibration is finished.")