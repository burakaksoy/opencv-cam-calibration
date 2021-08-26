import numpy as np
import cv2
import argparse
import sys
import pandas as pd
import math
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

def aruco_localize_2D(intrinsic_calib_path, intrinsic_calib_path_undistorted, calib_path_aruco, aruco_tags_info_path):
    
    [mtx, dist, R_co, R_oc, T_co, T_oc] = load_coefficients(intrinsic_calib_path)
    [mtx_new, dist_new, R_co, R_oc, T_co, T_oc] = load_coefficients(intrinsic_calib_path_undistorted)
    [R_op, R_po, T_op, T_po] = load_coefficients_best_fit_plane(calib_path_aruco)
    
    # read csv file that includes places of the aruco tags, their aruco type, ids, sizes and locations wrt their places
    df = pd.read_csv(aruco_tags_info_path)
    # drop rows that are on the floor so that there are only robots to be localized left
    df = df[df["place"]!="floor"]
    # drop duplicate rows with the same place (robot) name
    df = df.drop_duplicates(subset='place', keep="first")
    # get aruco dictionary types on the robots as a list
    aruco_types = list(df["aruco_type"].unique())
    print(aruco_types)

    cam = cv2.VideoCapture(1)
    cam.set(3,3840)
    cam.set(4,2160)
    # cam.set(3,640)
    # cam.set(4,480)

    is_started_plot = False

    # cv2.namedWindow("Image")

    # draw the figure so the animations will work
    fig = plt.figure()
    fig.show()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        
        time_stamp = time.time()
        
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

        places_all = []
        types_all = []
        corners_all = []
        ids_all = []
        sizes_all = []
        xs_all = []
        ys_all = []
        zs_all = []

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
            (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
            
            # only keep the detections that are NOT on the floor by looking at the IDs
            ids_on_robots_with_current_aruco_type = list(df[(df["aruco_type"]==aruco_type)]["id"])
            if len(corners) > 0: # verify *at least* one ArUco marker was detected            
                for i, markerID in enumerate(list(ids.flatten())): # loop over the detected ArUCo corners
                    if markerID in ids_on_robots_with_current_aruco_type:
                        robot_row = df[(df["aruco_type"]==aruco_type) & (df["id"]==markerID)]
                        place = robot_row["place"].item()
                        places_all.append(place)

                        types_all.append(aruco_type)

                        corners_all.append(corners[i])
                        
                        ids_all.append(markerID)
                        
                        markerSize = float(robot_row["size_mm"])
                        sizes_all.append(markerSize)

                        x = float(robot_row["x"])
                        xs_all.append(x)

                        y = float(robot_row["y"])
                        ys_all.append(y)

                        z = float(robot_row["z"])
                        zs_all.append(z)

            # print(places_all)
            # print(types_all)
            # print(corners_all)
            # print(ids_all)
            # print(sizes_all)
            # print(xs_all)
            # print(ys_all)
            # print(zs_all)

        print("[INFO] Num of detected Tags: ",len(corners_all))

        corners_all = np.array(corners_all)
        ids_all = np.array(ids_all)
        sizes_all = np.array(sizes_all)
        xs_all = np.array(xs_all)
        ys_all = np.array(ys_all)
        zs_all = np.array(zs_all)

        # verify *at least* one ArUco marker was remained on robots
        if len(corners_all) > 0:
            # then its a worthy image, save it
            img_name = "./localization_images/image_{}.png".format(time_stamp)
            # cv2.imwrite(img_name, frame)
            # print("{} written!".format(img_name))

            annotations_all = []
            rvecs_all = []
            tvecs_all = []
            # loop over the detected ArUCo corners and draw ids and bounding boxes around the detected markers with the robot information
            for (place, aruco_type, markerCorner, markerID, markerSize, x,y,z) in zip(places_all, types_all, corners_all, ids_all, sizes_all, xs_all,ys_all,zs_all):
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
                place_info = str(place) + ", "+ str(aruco_type) + ", id:" +str(markerID) + ", " + str(markerSize) + " mm"
                annotations_all.append(place_info)
                cv2.putText(frame, place_info,
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

                # Estimate the pose of the detected marker in camera frame
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorner, markerSize, mtx_new, dist_new)
                
                cv2.aruco.drawAxis(frame, mtx_new, dist_new, rvec, tvec, markerSize*0.75)  # Draw Axis

                # Add the Transform from robot frame to marker frame
                R_cm = cv2.Rodrigues(rvec.flatten())[0] # 3x3
                T_cm = tvec[0].T # 3x1

                R_rm = np.eye(3)# Marker and the Robot has the same orientation assumption, otherwise we had to parse it from csv file adding orientation paramaters
                T_rm = np.array([x,y,z]).reshape(3,1) # 3x1

                R_cr = R_cm @ R_rm.T
                T_cr = T_cm - R_cr@T_rm

                # print("[INFO] ArUco marker ID: {}".format(markerID))
                # print(tvec[0].flatten()) # in camera's frame)
                # print(rvec[0].flatten()) # in camera's frame)
                rvecs_all.append(R_cr)
                tvecs_all.append(T_cr)

            rvecs_all = np.array(rvecs_all) # (N,3,3)
            tvecs_all = np.array(tvecs_all) # (N,3,1)
            tvecs_all = np.squeeze(tvecs_all).T # (3,N)

            # show the output image
            # cv2.imshow("Image", frame)
            # k = cv2.waitKey(1)

            # Transform detected robot locations from camera frame to World frame
            rvecs_all = R_oc @ rvecs_all # (N,3,3) # R_or 

            tvecs_all = R_oc @ tvecs_all # (3,N)
            tvecs_all = T_oc + tvecs_all # (3,N) # T_or 
            # print(np.shape(tvecs_all))

            # Transform detected robot locations from World frame to plane frame
            rvecs_all = R_po @ rvecs_all # (N,3,3) # R_pr 

            tvecs_all = R_po @ tvecs_all # (3,N)
            tvecs_all = T_po + tvecs_all # (3,N) # T_pr

            # Convert 3D info to 2D data
            # Convert rotation matrices to ZYX euler angles
            THETAs = []
            for R in rvecs_all:
                x,y,z = euler_angles_from_rotation_matrix(R)
                THETAs.append(z) # rad

            tvecs_all = tvecs_all[0:2,:] # remove the z axis values since we need 2D data
            Xs = tvecs_all[0,:]
            Ys = tvecs_all[1,:]
            
            
            # plt.figure()
            plt.title("2D Data")
            plt.xlabel("X")
            plt.ylabel("Y")

            plt.scatter(Xs, Ys, marker='o')
            plt.axis('equal')
            for i, label in enumerate(places_all): 
                plt.text(Xs[i], Ys[i],label)

            for (x,y,theta) in zip(Xs,Ys,THETAs):
                plt.arrow(x, y, math.cos(theta)*250,math.sin(theta)*250, head_width = .1)

            plt.grid()
            plt.pause(0.01)
            # plt.show()
            fig.show()


        """---------------------------------------------------"""
        # if k%256 == 27:
        #     # ESC pressed
        #     print("Escape hit, closing...")
        #     break

    cam.release()
    cv2.destroyAllWindows()

    
#########################################################
# ZYX Euler angles calculation from rotation matrix
def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)

def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if isclose(R[2,0],-1.0):
        theta = math.pi/2.0
        psi = math.atan2(R[0,1],R[0,2])
    elif isclose(R[2,0],1.0):
        theta = -math.pi/2.0
        psi = math.atan2(-R[0,1],-R[0,2])
    else:
        theta = -math.asin(R[2,0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
        phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
    return psi, theta, phi #  x y z for Rz * Ry * Rx
#########################################################

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

def load_coefficients_best_fit_plane(path):
    """ Loads best fitting plane transformation parameters relative to the world frame. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    R_op = cv_file.getNode("R_op").mat()
    R_po = cv_file.getNode("R_po").mat()
    T_op = cv_file.getNode("T_op").mat()
    T_po = cv_file.getNode("T_po").mat()
    
    cv_file.release()
    return [R_op, R_po, T_op, T_po]


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
    parser.add_argument('--calib_file_aruco', type=str, required=True, help='YML file to read transformations from world frame to best fitting 2D plane to aruco')
    parser.add_argument('--aruco_tags_info_file', type=str, required=True, help="info csv file about the existing Aruco tags in the workspace")
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')

    args = parser.parse_args()
    

    # R_op, R_po, T_op, T_po, RMSE = 
    aruco_localize_2D(args.calib_file, args.calib_file_undistorted,  args.calib_file_aruco, args.aruco_tags_info_file)
    # save_coefficients(R_op, R_po, T_op, T_po, RMSE, args.save_file)
    print("Aruco Localization 2D is finished.")