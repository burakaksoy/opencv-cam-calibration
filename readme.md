# Codes for a camera calibration with opencv python

Run the files in the numbered order.

## 01-capture.py  

```
python3 01-capture.py
```
Keep pressing 'SPACE' in your keyboard to capture as many as images you want. Then Press 'ESC' to close the program. Images are saved to folder `calibration_images/` with a numbered name combined with prefix `image_` and type `.png` (for example `image_01.png`)

## 02-intrinsic_calibration.py 

```
python3 02-instrinsic_calibration.py --image_dir ./calibration_images --image_format png --prefix image_ --square_size 29 --width 7 --height 6 --save_file ./calibration_files/camera.yml
```
(Square size: is in mm, length of one edge in a square at the chessboard)  
(Width: number of squares minus 1 at the long edge of the chessboard)  
(Height: number of squares minus 1 at the short edge of the chessboard)  

The terminal output will be similar to below:
```
./calibration_images/image_*.png
82
[[ 0.08351066 -0.02958139  0.00231711 -0.0038417  -0.00335872]]
mtx
[[1.88145992e+03 0.00000000e+00 1.90107419e+03]
 [0.00000000e+00 1.88377572e+03 1.10234687e+03]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
total error: 0.09799668193226092
Calibration is finished. RMS:  0.7461922466288191
```
These parameters are saved to `calibration_files/` folder with the name specified as a yaml file. For example `camera.yml`.

## 02.5-undistort_images.py

Undistort the images with the found calibration parameters.
```
python3 02.5-undistort_images.py --image_dir ./calibration_images --prefix image_ --image_format png --save_dir ./calibration_images_undistorted --calib_file ./calibration_files/camera.yml
```

## Re-calibration on undistorted images

```
python3 02-instrinsic_calibration.py --image_dir ./calibration_images_undistorted --image_format png --prefix image_ --square_size 29 --width 7 --height 6 --save_file ./calibration_files/camera_undistorted.yml
```  
   
  
*Note: After this you will observe that the new distortion coefficients will be closer to 0 and the camera matrix should be almost the same as the previous one.  
For example, the new results could be:*
```
./calibration_images_undistorted/image_*.png
82
[[ 0.00035412 -0.0006639  -0.00021844  0.00017642  0.00053579]]
mtx
[[1.88162163e+03 0.00000000e+00 1.90171344e+03]
 [0.00000000e+00 1.88384708e+03 1.10042690e+03]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
total error: 0.09327865766887405
Calibration is finished. RMS:  0.713327834560297
```


## 03-extrinsic_calibration.py
This defines an origin point and, the x and y axes in the world. The following command captures 1 more chessboard image with the same properties above, saves it into `calibration_image_extrinsic/` folder with the name `image_extrinsic` and image type `png`. Then calculates the extrinsic parameters. Also draws the calculated axes on the image with 1000 mm length. The calculated results are added to `./calibration_files/camera_undistorted.yml`

```
python3 03-extrinsic_calibration.py --image_dir ./calibration_image_extrinsic --image_name image_extrinsic  --image_format png --square_size 65 --width 8 --height 7 --save_file ./calibration_files/camera_undistorted.yml --calib_file ./calibration_files/camera.yml --calib_file_undistorted ./calibration_files/camera_undistorted.yml  
```

An example output would be:
```
Hit SPACE key to capture, Hit ESC key to continue
Escape hit, closing...
axis:  [[1000.    0.    0.]
 [   0. 1000.    0.]
 [   0.    0. 1000.]]
R_co: [[ 0.99795671  0.04989442 -0.03991183]
 [-0.06373225  0.82171304 -0.56632666]
 [ 0.00453953  0.56771316  0.82321392]]
R_oc: [[ 0.99795671 -0.06373225  0.00453953]
 [ 0.04989442  0.82171304  0.56771316]
 [-0.03991183 -0.56632666  0.82321392]]
T_co: [[-118.43600945]
 [-618.61076188]
 [4200.12307269]]
T_oc: [[   59.70197468]
 [-1870.23531575]
 [-3812.66255087]]
Extrinsic Calibration is finished.
```
(`R_co`: Rotation matrix from camera frame to origin)  
(`R_oc`: Rotation matrix from origin frame to camera)  
(`T_co`: Translation vector from camera frame to origin in camera frame)  
(`T_oc`: Rotation matrix from origin frame to camera in origin frame)

## 04-aruco_calibration.py
```
python3 04-aruco_calibration.py --calib_file ./calibration_files/camera.yml --image_dir ./calibration_image_aruco --image_name image_aruco --image_format png --aruco_type DICT_5X5_50 --square_size 200 --save_file ./calibration_files/camera_aruco.yml
```


