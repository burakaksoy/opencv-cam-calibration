import numpy as np
import cv2
import glob
import argparse

def undistort(intrinsic_calib_path, dirpath, prefix, image_format, save_dir):
    """ Undistort operation for images in the given directory path. """

    [mtx, dist] = load_coefficients(intrinsic_calib_path)

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    if image_format[:1] == '.':
        image_format = image_format[1:]

    images = sorted(glob.glob(dirpath+'/' + prefix + '*.' + image_format))

    print(dirpath+'/' + prefix + '*.' + image_format)
    print(len(images))

    for fname in images:
        img = cv2.imread(fname)
        
        # img = cv2.imread(images[0])
        # h,  w = img.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # print("newcameramtx")
        # print(str(newcameramtx))

        # undistort
        # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        dst = cv2.undistort(img, mtx, dist, None, mtx)

        # # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]

        # cv2.imshow(fname, img)
        # cv2.waitKey(10)

        fname_new = save_dir + fname.split("./calibration_images")[1]
        print(fname_new)
        cv2.imwrite(fname_new, dst)

    cv2.destroyAllWindows()
    
    return

    
def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    # Camera Matrix
    cv_file.write("K", mtx)
    # Distortion Coefficients
    cv_file.write("D", dist)
    # # Rotation matrix 
    # R_co, _ = cv2.Rodrigues(rvecs[0]) 
    # cv_file.write("R_co", R_co )
    # # Tranlastion vector
    # cv_file.write("T_co", tvecs[0])
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

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
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image undistortion code to undistort images with given camera parameters')
    parser.add_argument('--calib_file', type=str, required=True, help='YML file to read calibration matrices')
    parser.add_argument('--image_dir', type=str, required=True, help='image directory path to read')
    parser.add_argument('--image_format', type=str, required=True,  help='image format, png/jpg')
    parser.add_argument('--prefix', type=str, required=True, help='image prefix')
    parser.add_argument('--save_dir', type=str, required=True, help='image directory path to save the undistorted images')

    args = parser.parse_args()
    undistort(args.calib_file, args.image_dir, args.prefix, args.image_format, args.save_dir)
    print("Undistortion is finished")
