# 3D Multi-Camera Pose Estimation System

This README provides an overview of the key functions in the 3D multi-camera pose estimation system as documented in pose_explain.html.

## Camera Calibration Functions

### calibrate_single_camera
**Inputs:**
- `images_folder`: Path to folder containing calibration images
- `pattern_size`: Chessboard pattern dimensions (default: 9,6)
- `square_size`: Size of chessboard squares in mm (default: 30.0)

**Outputs:**
- `ret`: Calibration error (average reprojection error)
- `mtx`: Camera intrinsic matrix (3x3)
- `dist`: Distortion coefficients
- `rvecs`: Rotation vectors for each calibration image
- `tvecs`: Translation vectors for each calibration image

### calibrate_three_cameras
**Inputs:**
- `cam1_folder`: Path to calibration images for camera 1
- `cam2_folder`: Path to calibration images for camera 2
- `cam3_folder`: Path to calibration images for camera 3

**Outputs:**
- `save_path`: Path to the saved calibration file (.npz)

## ArUco Marker Detection

### get_aruco_axis
**Inputs:**
- `img_L/M/R`: Images from three cameras
- `aruco_detector`: ArUco marker detector object
- `board_coord`: Dictionary of ArUco marker coordinates (ID → coordinates)
- `cams_params`: Camera parameters tuple (mtx1, dist1, mtx2, dist2, mtx3, dist3)

**Outputs:**
- `R_aruco2camL/M/R`: Rotation matrices from ArUco to each camera
- `t_aruco2camL/M/R`: Translation vectors from ArUco to each camera
- `R_camL/M/R2aruco`: Rotation matrices from each camera to ArUco
- `t_camL/M/R2aruco`: Translation vectors from each camera to ArUco
- `img_L/M/R`: Processed images with visualizations

## 2D Pose Estimation and 3D Reconstruction

### process_frame
**Inputs:**
- `detector`: MediaPipe pose detector object
- `frame_left/middle/right`: Original frames from three cameras
- `img_L/M/R`: Visualization images
- `cams_params`: Camera parameters tuple
- `cam_P`: Camera projection matrices

**Outputs:**
- `points_3d`: 3D reconstructed points (33 keypoints × 3 coordinates)
- `img_L/M/R`: Updated visualization images
- `rmse`: Tuple of reprojection errors (left, middle, right)

### triangulate_points_three_cameras
**Inputs:**
- `points_left/middle/right`: 2D keypoints from three cameras (each 33×2)
- `mtx1/2/3`: Camera intrinsic matrices
- `dist1/2/3`: Distortion coefficients
- `P1/2/3`: Projection matrices ([R|t])

**Outputs:**
- `points_3d`: 3D reconstructed points (33×3 array)

### calculate_reprojection_error
**Inputs:**
- `points_3d`: 3D points (N×3)
- `points_2d`: 2D detected points (N×2)
- `P`: Projection matrix (3×4)
- `mtx`: Camera intrinsic matrix
- `dist`: Distortion coefficients

**Outputs:**
- `rmse`: Root mean square reprojection error

## Visualization

### visualize_3d_animation_three_cameras
**Inputs:**
- `points`: 3D keypoint sequence [frames, 33, 3]
- `aruco_axis`: ArUco coordinate system data
- `camL/M/R_axis`: Camera coordinate system data
- `title`: Window title (optional)

**Outputs:**
- Interactive 3D animation window

## Main Processing Pipeline

### process_videos
**Inputs:**
- `video_path_left/center/right`: Paths to video files from three cameras
- `start_frame`: Starting frame index (default: 0)

**Outputs:**
- `all_points_3d`: All reconstructed 3D points sequence
- `aruco_axis`: ArUco coordinate system data sequence
- `camL/M/R_axis`: Camera coordinate system data sequences
- `all_rmse`: Reprojection errors for all frames

### main
**Inputs:** None (hardcoded video paths)

**Outputs:**
- 3D visualization window
- Reprojection error plot

## System Requirements

- Three synchronized cameras (webcams or industrial cameras)
- Python with OpenCV, NumPy, MediaPipe, and Matplotlib
- ArUco markers and a calibration chessboard pattern
- Sufficient computing power (recommended: 16GB+ RAM and a GPU)

## Key Features

- Markerless motion capture using MediaPipe pose detection
- Geometric 3D reconstruction through triangulation
- Real-time processing capability
- Comprehensive error analysis through reprojection error
- Colored 3D visualization with skeleton connections
- Support for multiple coordinate systems 