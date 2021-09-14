import numpy as np
import cv2
import argparse

def calibrate_camera_extrinsic(intrinsic_calib_path, intrinsic_calib_path_undistorted, image_dir, image_name, image_format, square_size, width=9, height=6):
    

    [mtx, dist] = load_coefficients(intrinsic_calib_path)
    [mtx_new, dist_new] = load_coefficients(intrinsic_calib_path_undistorted)

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
    
    ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
    assert ret, "Could not find calibration target"

    cv2.drawChessboardCorners(frame, (width, height), corners, ret)

    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    # Find the rotation and translation vectors.
    # ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
    # ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, newcameramtx, None)
    ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx_new, dist_new)

    # axis_sizes= [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    axis_sizes= [1000]
    for axis_size in axis_sizes:
        # axis_size = 4000 # mm
        # axis = np.float32([[axis_size,0,0], [0,axis_size,0], [0,0,axis_size]]).reshape(-1,3) # Default openCV chessboard axes
        axis = np.float32([[0,axis_size,0], [axis_size,0,0], [0,0,-axis_size]]).reshape(-1,3) # For our world we want axes in this config, so draw accordingly
        # axis = np.float32([[axis_size,0,0], [0,0,0], [0,0,0]]).reshape(-1,3)
        # print('axis: ',axis)

        # project 3D points to image plane
        # imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, newcameramtx, None)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx_new, dist_new)

        
        # Show the markers on the image
        frame = draw(frame,corners2,imgpts)
    cv2.namedWindow("test")
    cv2.imshow("test", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()


    return mtx_new, dist_new, rvecs, tvecs

def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

def save_coefficients(mtx, dist, rvecs, tvecs, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    # Camera Matrix
    cv_file.write("K", mtx)
    # Distortion Coefficients
    cv_file.write("D", dist)

    R_bo = np.array([[0,1.,0],[1.,0,0],[0,0,-1.]]) # Rotation matrix btw chessBoard and the world frame we want

    # Rotation matrix 
    
    R_cb = cv2.Rodrigues(rvecs.flatten())[0] 
    R_co = R_cb @ R_bo
    print("R_co: " + str(R_co))
    cv_file.write("R_co", R_co )

    print("R_oc: " + str(R_co.T))
    cv_file.write("R_oc", R_co.T)

    # Tranlastion vector
    cv_file.write("T_co", tvecs)
    print("T_co: " + str(tvecs))

    print("T_oc: " + str( -R_co.T.dot(tvecs)))
    cv_file.write("T_oc", -R_co.T.dot(tvecs))

    # note you *release* you don't close() a FileStorage object
    cv_file.release()

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

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 1)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 1)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 1)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Extrinsic calibration')
    parser.add_argument('--image_dir', type=str, required=True, help='image directory path')
    parser.add_argument('--image_format', type=str, required=True,  help='image format, png/jpg')
    parser.add_argument('--image_name', type=str, required=True, help='image name without extension or location')
    parser.add_argument('--square_size', type=float, required=False, help='chessboard square size')
    parser.add_argument('--width', type=int, required=False, help='chessboard width size, default is 9')
    parser.add_argument('--height', type=int, required=False, help='chessboard height size, default is 6')
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')
    parser.add_argument('--calib_file', type=str, required=True, help='YML file to read calibration matrices')
    parser.add_argument('--calib_file_undistorted', type=str, required=True, help='YML file to read calibration matrices of undistorted images')

    args = parser.parse_args()
    mtx, dist, rvecs, tvecs = calibrate_camera_extrinsic(args.calib_file, args.calib_file_undistorted, args.image_dir, args.image_name, args.image_format, args.square_size, args.width, args.height)
    save_coefficients(mtx, dist, rvecs, tvecs, args.save_file)
    print("Extrinsic Calibration is finished.")