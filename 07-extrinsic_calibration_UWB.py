import numpy as np
import cv2
import glob
import argparse
import pandas as pd

from numpy.lib.function_base import append

def calibrate_uwb_extrinsic(intrinsic_calib_path, intrinsic_calib_path_undistorted,  calib_path_aruco, image_dir, prefix, image_format, square_size, width, height, uwb_tags_info_path):
    [mtx, dist, R_co, R_oc, T_co, T_oc] = load_coefficients(intrinsic_calib_path)
    [mtx_new, dist_new, R_co, R_oc, T_co, T_oc] = load_coefficients(intrinsic_calib_path_undistorted)
    [R_op, R_po, T_op, T_po] = load_coefficients_best_fit_plane(calib_path_aruco)
    [T_iu_vecs, RMSE_vals] = load_uwb_locations(uwb_tags_info_path)

    if image_dir[-1:] == '/':
        image_dir = image_dir[:-1]

    if image_format[:1] == '.':
        image_format = image_format[1:]

    images = sorted(glob.glob(image_dir+'/' + prefix + '*.' + image_format))

    print(image_dir+'/' + prefix + '*.' + image_format)
    print(len(images))

    T_pu_vecs = []

    for fname in images:
        frame = cv2.imread(fname)
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
        # cv2.namedWindow(fname, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(fname, (1280, 720))
        # cv2.imshow(fname, frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # print("Frame name: ", fname)
        # Find Pose of the UWB frame wrt to camera 
        R_bu = np.array([[0,1.,0],[1.,0,0],[0,0,-1.]]) # Rotation matrix btw chessBoard(b) and the uwb frame(u) we want
        R_cb = cv2.Rodrigues(rvecs.flatten())[0]  # Rotation matrix btw camera(c) and the chessboard(b)
        R_cu = R_cb @ R_bu # 3x3
        T_cu = tvecs # 3x1

        # Rotation Matrices btw camera and uwb
        # print("R_cu: " + str(R_cu))
        # print("R_uc: " + str(R_cu.T))

        # Tranlastion vectors btw camera and uwb
        # print("T_cu: " + str(tvecs)) # in camera frame
        # print("T_uc: " + str( -R_cu.T.dot(tvecs))) # in uwb frame

        # Find pose of the UWB frame(u) wrt best fitting plane(p)
        R_cp = R_co @ R_op
        R_pc = R_cp.T
        R_pu = R_pc @ R_cu
        R_up = R_pu.T

        T_pu = - R_pc @ T_co + T_po + R_pc @ T_cu
        # print("R_pu: ", R_pu)
        # print("T_pu: ", T_pu)

        T_pu_vecs.append(np.squeeze(T_pu)+1000.0)

    T_pu_vecs = np.array(T_pu_vecs) # Nx3
    # print("T_pu_vecs:", T_pu_vecs)

    # Use only the positions with a lower RMSE than a certain threshold
    rmse_threshold = 60.0 # mm
    print("rmse_threshold: ", rmse_threshold, " mm")
    valid_indices = np.argwhere( RMSE_vals <= rmse_threshold)[:,0] # (N,)
    print("valid_indices:", valid_indices)
    print("number of valid_indices:", len(valid_indices))

    RMSE_vals = RMSE_vals[valid_indices,:]
    T_pu_vecs = T_pu_vecs[valid_indices,:] # Tranlation vectors from best fit plane(p) to detected uwb calibration frame(u) in best fit plane frame(p)
    T_iu_vecs = T_iu_vecs[valid_indices,:] # Tranlation vectors from inertial uwb base(i) to detected uwb calibration frame(u) in inertial uwb base frame(i)
    
    # print("RMSE_vals:", RMSE_vals)
    print("T_iu_vecs:", T_iu_vecs)
    print("T_pu_vecs:", T_pu_vecs)

    R_pi,T_pi = rigid_transform_3D(T_iu_vecs.T,T_pu_vecs.T)
    # print("R_pi: ", R_pi) # Rotation matrix from best fitting plane(p) to inertial uwb base frame(i)
    # print("T_pi:",  T_pi) # Translation vector from best fitting plane(p) to inertial uwb base frame(i) in (p) 
    # print("R_ip: ", R_pi.T) # Rotation matrix from inertial uwb base frame(i) to best fitting plane(p) 
    # print("T_ip: ",  -R_pi.T@T_pi) # Translation vector from inertial uwb base frame(i) to best fitting plane(p) in (i)

    return R_pi,T_pi

def rigid_transform_3D(A, B):
    # This function is taken from https://github.com/nghiaho12/rigid_transform_3D.git
    # Further information: http://nghiaho.com/?page_id=671
    
    # Input: expects 3xN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix from B to A
    # t = 3x1 column vector from B to A, in B
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    # print("centroid_A:" , centroid_A)
    centroid_B = centroid_B.reshape(-1, 1)
    # print("centroid_B:" , centroid_B)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)
    # print("H:", H)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def load_uwb_locations(path):
    # read csv file that includes positions of the uwb tags wrt uwb base inertial frame (i)
    df = pd.read_csv(path)
    # Select xyz positions
    T_iu_vecs = df[['x','y','z']].values.tolist()
    T_iu_vecs = np.array(T_iu_vecs)*1000.0 # Nx3 # convert to mm from m

    # Select RMSE values
    RMSE_vals = df[['rmse']].values.tolist()
    RMSE_vals = np.array(RMSE_vals)*10.0 # Nx1 # convert to mm from cm

    # print("T_iu_vecs:", T_iu_vecs)
    # print("RMSE_vals:", RMSE_vals)

    return [T_iu_vecs, RMSE_vals]

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

def save_coefficients(R_pi,T_pi, path):
    """ Save the poses btw best fitting plane (p) and inertial uwb base frame(i) to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)

    # Rotation matrix     
    print("R_pi: ", R_pi) # Rotation matrix from best fitting plane(p) to inertial uwb base frame(i)
    cv_file.write("R_pi", R_pi )

    print("R_ip: ", R_pi.T) # Rotation matrix from inertial uwb base frame(i) to best fitting plane(p) 
    cv_file.write("R_ip", R_pi.T)

    # Tranlastion vector
    print("T_pi:",  T_pi) # Translation vector from best fitting plane(p) to inertial uwb base frame(i) in (p) 
    cv_file.write("T_pi", T_pi)

    print("T_ip: ",  -R_pi.T@T_pi) # Translation vector from inertial uwb base frame(i) to best fitting plane(p) in (i)
    cv_file.write("T_ip", -R_pi.T@T_pi)

    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 1)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 1)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 1)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extrinsic calibration between The Best Fitting Plane and UWB Base')
    parser.add_argument('--calib_file', type=str, required=True, help='YML file to read calibration matrices')
    parser.add_argument('--calib_file_undistorted', type=str, required=True, help='YML file to read calibration matrices of undistorted images')
    parser.add_argument('--calib_file_aruco', type=str, required=True, help='YML file to read transformations from world frame to best fitting 2D plane to aruco')

    parser.add_argument('--image_dir', type=str, required=True, help='image directory path')
    parser.add_argument('--prefix', type=str, required=True, help='image name without extension or location')
    parser.add_argument('--image_format', type=str, required=True,  help='image format, png/jpg')
    
    parser.add_argument('--square_size', type=float, required=False, help='chessboard square size')
    parser.add_argument('--width', type=int, required=True, help='chessboard width size (num of squares on the chessboard minus one)')
    parser.add_argument('--height', type=int, required=True, help='chessboard height size (num of squares on the chessboard minus one)')
    
    parser.add_argument('--uwb_tags_info_file', type=str, required=True, help='chessboard height size (num of squares on the chessboard minus one)')

    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')


    args = parser.parse_args()

    R_pi,T_pi = calibrate_uwb_extrinsic(args.calib_file, args.calib_file_undistorted, args.calib_file_aruco,  args.image_dir, args.prefix, args.image_format, args.square_size, args.width, args.height, args.uwb_tags_info_file)
    save_coefficients(R_pi,T_pi, args.save_file)
    print("UWB Extrinsic Calibration is finished.")