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
(`R_co`: Rotation matrix from camera frame to world frame)  
(`R_oc`: Rotation matrix from world frame to camera)  
(`T_co`: Translation vector from camera frame to world frame in camera frame)  
(`T_oc`: Translation vector from world frame to camera in world frame)

## 04-aruco_calibration.py

This script finds the best fitting plane with respect to world frame (defined in extrinsic calibration) as "floor" using the ArUco tags that are spread on a surface. 
The properties of the ArUco tags are defined in a `csv` file with the following strutcture (for example see: `./aruco_tags_info.csv`):


| place    | aruco_type  | id  | size_mm | x   | y   | z   |
| ---      | ---         | --- | ---     | --- | --- | --- |
| floor    | DICT_5X5_50 | 12  | 190     | 0 | 0 | 0 |

`place` key is for defining the frame which an Aruco tag is attached, `aruco_type` is the dictionary type of the Aruco tag. `id` is the corresponding Aruco tag id. `size_mm` is the one side size of the Aruco tags in millimeters. `x,y,z` keys are not used for this script but they will be used for defining the relative position of the aruco tags that they are attached to, for instance where the aruco tag is attached with respect to a robot's center. Note that the orientations of the tags are not considered for this script, for later scripts we will assume that they are aligned with the frames that they are attached to, hence we did not define orientation paramaters in the `csv` file.


*Note: OpenCv accepts the following Aruco tag types as of today(August 25, 2021): [See here](https://docs.opencv.org/3.4/d9/d6a/group__aruco.html#gac84398a9ed9dd01306592dd616c2c975)*
```
	"DICT_4X4_50"
	"DICT_4X4_100"
	"DICT_4X4_250"
	"DICT_4X4_1000"
	"DICT_5X5_50"
	"DICT_5X5_100"
	"DICT_5X5_250"
	"DICT_5X5_1000"
	"DICT_6X6_50"
	"DICT_6X6_100"
	"DICT_6X6_250"
	"DICT_6X6_1000"
	"DICT_7X7_50"
	"DICT_7X7_100"
	"DICT_7X7_250"
	"DICT_7X7_1000"
	"DICT_ARUCO_ORIGINAL"
	"DICT_APRILTAG_16h5"
	"DICT_APRILTAG_25h9"
	"DICT_APRILTAG_36h10"
	"DICT_APRILTAG_36h11"
```

 The following command reads the previously calculated default and undistorted intrinsic parameters and the defined world frame as extrinsic parameters from `./calibration_files/camera.yml` and `./calibration_files/camera_undistorted.yml`, captures 1 image, saves it into `calibration_image_aruco/` folder with the name `image_aruco` and image type `png`. Undistorts the image to prepare for aruco tag detection and detects 3D position of the tags with respect to the defined world origin. Draws bounding boxes around the detected tags (this could be used for visual inspection for the detection accuracy and the inaccurate looking ones can be eliminated from the csv file) and also draws their frames. Then finds the best fitting plane for the detected 3D points. Best fitting plane is defined with its passing-through point as the mean of the 3D points and a normal vector represented in the world frame. Fitting residual error value is printed as RMSE into the console. Then, the world frame origin and it's x-axis are projected into the best fitting plane, and are used to define the best plane frame and the origin.  The calculated results are saved to `./calibration_files/camera_aruco.yml`

```
python3 04-aruco_calibration.py --calib_file ./calibration_files/camera.yml --calib_file_undistorted ./calibration_files/camera_undistorted.yml --image_dir ./calibration_image_aruco --image_name image_aruco --image_format png --aruco_tags_info_file ./aruco_tags_info2.csv  --save_file ./calibration_files/camera_aruco.yml
```

An example output would be:
```
Hit SPACE key to capture, Hit ESC key to continue
Escape hit, closing...
[INFO] detecting 'DICT_5X5_50' tags...
[INFO] Num of detected Tags:  13
R_op: [[ 9.99702278e-01 -8.23993651e-18  2.43998990e-02]
 [ 1.16489311e-03  9.98859714e-01 -4.77275046e-02]
 [-2.43720761e-02  4.77417184e-02  9.98562332e-01]]
R_po: [[ 9.99702278e-01  1.16489311e-03 -2.43720761e-02]
 [-8.23993651e-18  9.98859714e-01  4.77417184e-02]
 [ 2.43998990e-02 -4.77275046e-02  9.98562332e-01]]
T_op: [[  5.56848635]
 [-10.89225649]
 [227.88949733]]
T_po: [[-2.06856754e-12]
 [ 0.00000000e+00]
 [-2.28217599e+02]]
RMSE:  34.99319118786799  (mm)
Aruco Calibration is finished.
```
(`R_op`: Rotation matrix from world frame to best fitting plane frame)  
(`R_po`: Rotation matrix from origin frame to world frame)  
(`T_op`: Translation vector from world frame to best fitting plane frame in world frame)  
(`T_po`: Translation vector from best fitting plane frame to world frame in best fitting plane frame)

## 05-aruco_localization_2D_realtime.py
Finds the 
```
python3 05-aruco_localization_2D_realtime.py --calib_file ./calibration_files/camera.yml --calib_file_undistorted ./calibration_files/camera_undistorted.yml --calib_file_aruco ./calibration_files/camera_aruco.yml  --aruco_tags_info_file ./aruco_tags_info3.csv  --save_file ./localization_files/robot_positions.csv
```

## 06-capture_UWB.py
Use it to capture a series of checkerboard images attached to UWB tags. These images are going to be used to find (calibrate) the relative pose between the vision system (the best fitting plane frame that is previously found) and the UWB anchors origin.  
```
python3 06-capture_UWB.py
```
Keep pressing 'SPACE' in your keyboard to capture as many as images you want. Then Press 'ESC' to close the program. Images are saved to folder `calibration_images_uwb/` with a numbered name combined with prefix `image_` and type `.png` (for example `image_01.png`)

## 07-extrinsic_calibration_UWB.py
Assuming the UWB xyz locations are also captured in the previous step along with the images, this step finds (calibrates) the relative pose between the vision system (the best fitting plane frame that is previously found) and the UWB anchors origin. 

```
python3 07-extrinsic_calibration_UWB.py --calib_file ./calibration_files/camera.yml --calib_file_undistorted ./calibration_files/camera_undistorted.yml --calib_file_aruco ./calibration_files/camera_aruco.yml  --image_dir ./calibration_images_uwb --prefix image_  --image_format png --square_size 65 --width 8 --height 7 --uwb_tags_info_file ./uwb_tags_info.csv --save_file ./calibration_files/bestFitPlane_to_inertialUWB.yaml
```

python3 07-extrinsic_calibration_UWB.py 
--calib_file ./calibration_files/camera.yml 
--calib_file_undistorted ./calibration_files/camera_undistorted.yml 
--calib_file_aruco ./calibration_files/camera_aruco.yml  

--image_dir ./calibration_images_uwb 
--prefix image_  
--image_format png 

--square_size 65 
--width 8 
--height 7 
--uwb_tags_info_file ./uwb_tags_info.csv 
--save_file ./localization_files/robot_positions.csv
