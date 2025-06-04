import cv2
import numpy as np
import sys
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess
import json
from scipy.optimize import minimize
from collections import defaultdict

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.io import savemat

import os
os.environ['PATH'] += ';D:\\software\\ffmpeg-7.1-essentials_build'
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = 'D:\\software\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe'


# ---------------------------- 1. 单独相机校正 ---------------------------- #
def cam_intrinsic_calib():
    # Number of inner corners in the chessboard
    nx = 9  # 水平方向的角点数
    ny = 6  # 垂直方向的角点数
    w = 1920  # 添加图像宽度
    h = 1080  # 添加图像高度

    # Termination criteria for corner sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    # Prepare object points, like (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    dir_path = "path/to/in_img"
    cam_ids = ['cam1', 'cam2']
    cam_intrinsic_params = {}
    for cam_id in cam_ids:
        img_dir = os.path.join(dir_path, cam_id)
        file_nms = os.listdir(img_dir)
        img_nms = [file_nm for file_nm in file_nms if file_nm.endswith(('jpg'))]
        objpoints = []      # 3D point in real world space
        imgpoints = []      # 2D points in left image plane
        for img_nm in img_nms:
            img_path = os.path.join(dir_path, cam_id, img_nm)
            img = cv2.imread(img_path)
            ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)
            if not ret:
                print(img_nm, 'is not detected corners')
                continue
            # Refine corner locations
            corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                            corners, (11, 11), (-1, -1), criteria)
            # Add object points, image points
            objpoints.append(objp)
            imgpoints.append(corners)

        # 检查是否成功检测到足够的角点进行校准
        if len(objpoints) == 0:
            print(cam_id, "没有足够的有效数据进行相机校准。请检查棋盘图像的质量和路径。")
            sys.exit()

        # Perform camera calibration for each camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), None, None)
        
        cam_intrinsic_params[cam_id] = {'mtx': mtx, 'dist': dist}
        print(cam_id)
        print(mtx)
        print(dist)
    # 保存单相机校准结果
    np.savez('cam_intrinsic_params.npz', cam_intrinsic_params)


# ---------------------------- 2. 多个相机外参同时标定 ---------------------------- #
def sys_extrinsic_calib(dict_type='DICT_ARUCO_ORIGINAL'):
    """
    相机外参标定
    Args:
        dict_type: ArUco字典类型，支持以下类型：
            - DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000
            - DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000
            - DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000
            - DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000
            - DICT_ARUCO_ORIGINAL
    """
    # ArUco字典映射
    ARUCO_DICTS = {
        # 4x4
        'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
        'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
        'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
        'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
        # 5x5
        'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
        'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
        'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
        'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
        # 6x6
        'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
        'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
        'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
        'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
        # Original
        'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL
    }

    # 设置aruco参数
    if dict_type not in ARUCO_DICTS:
        raise ValueError(f"不支持的ArUco字典类型: {dict_type}")
    
    used_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_type])
    # 定义 ArUco 标记板的坐标
    board_length = 160
    board_gap = 30
    base_coord = np.array([[0,0],[0,1],[1,1],[1,0]])
    board_coord = {
        0: base_coord * board_length + [0,0],
        1: base_coord * board_length + [board_length+board_gap,0],
        2: base_coord * board_length + [0,board_length+board_gap],
        3: base_coord * board_length + [board_length+board_gap,board_length+board_gap],
        4: base_coord * board_length + [0,(board_length+board_gap)*2],
        5: base_coord * board_length + [board_length+board_gap,(board_length+board_gap)*2],
    }
    # 显示坐标系
    axis_coord = np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ],dtype=np.float32)
    axis_coord = axis_coord * 500
    cam_intrinsic_params = np.load('cam_intrinsic_params.npz')
    dir_path =  "path/to/ex_img"
    cam_ids = ['cam1', 'cam2']
    cam_extrinsic_params = {}
    for cam_id in cam_ids:
        mtx_left = cam_intrinsic_params[cam_id]['mtx']
        dist_left = cam_intrinsic_params[cam_id]['dist']
        img_path = os.path.join(dir_path, cam_id, 'img.jpg')
        img = cv2.imread(img_path)
        # 在图像中检测 ArUco 标记
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, used_dict)
        img_coords = []
        point_coords = []
        aruco_ps_2cam = []
        cam_extrinsic_params[cam_id] = {}
        if ids is None:
            print(cam_id, ' is not detected corners for ex_calib.')
            return
        for i in range(len(ids)):
            if ids[i][0] not in board_coord.keys():
                continue
            tmp_marker = corners[i][0]
            tmp_marker_tl = (int(tmp_marker[0][0]), int(tmp_marker[0][1]))
            tmp_marker_tr = (int(tmp_marker[1][0]), int(tmp_marker[1][1]))
            tmp_marker_br = (int(tmp_marker[2][0]), int(tmp_marker[2][1]))
            tmp_marker_bl = (int(tmp_marker[3][0]), int(tmp_marker[3][1]))
            cv2.circle(img, tmp_marker_tl, 10, (0, 0, 255), -1)
            cv2.circle(img, tmp_marker_tr, 10, (0, 255, 0), -1)
            cv2.circle(img, tmp_marker_br, 10, (255, 0, 0), -1)
            cv2.circle(img, tmp_marker_bl, 10, (0, 170, 255), -1)
            cv2.putText(img, "ID: " + str(ids[i]), (int(tmp_marker_tl[0] + 10), int(tmp_marker_tl[1] + 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1,
                        cv2.LINE_AA)
            img_coord = np.array([
                [tmp_marker[0][0], tmp_marker[0][1]],
                [tmp_marker[1][0], tmp_marker[1][1]],
                [tmp_marker[2][0], tmp_marker[2][1]],
                [tmp_marker[3][0], tmp_marker[3][1]]
            ])
            img_coords.append(np.squeeze(img_coord))
            tem_coord = np.hstack((board_coord[ids[i][0]], np.zeros(len(board_coord[ids[i][0]]))[:,None]))
            point_coords.append(tem_coord)

            image_C = np.ascontiguousarray(img_coord[:,:2]).reshape((-1,1,2))
            ret, rvec, tvec = cv2.solvePnP(tem_coord, image_C, mtx_left, dist_left)
            R_aruco2cam, _ = cv2.Rodrigues(rvec)
            t_aruco2cam = tvec
            aruco_p_2cam = np.dot(R_aruco2cam, tem_coord.T).T + t_aruco2cam.T
            aruco_ps_2cam.append(aruco_p_2cam)

        img_coords = np.concatenate(img_coords, axis=0)
        img_coords = np.hstack((img_coords, np.ones(len(img_coords))[:,None]))
        point_coords = np.concatenate(point_coords, axis=0)
        aruco_ps_2cam = np.concatenate(aruco_ps_2cam, axis=0)

        if len(img_coords) > 0 and len(point_coords) > 0:
            # 初始化一个字典来存储聚类  
            clusters = defaultdict(list)  
            cluster_ids = []  
            cluster_indx = {}
            for i, point in enumerate(aruco_ps_2cam):  
                new_cluster = True  
                for cluster_id, cluster_points in clusters.items():  
                    for cluster_point in cluster_points:  
                        if np.linalg.norm(point-cluster_point) <= board_coord[5][3,1]:  
                            clusters[cluster_id].append(point)  
                            cluster_indx[cluster_id].append(i)
                            new_cluster = False  
                            break  
                    if not new_cluster:  
                        break
            
                if new_cluster:  
                    cluster_id = len(cluster_ids)  
                    cluster_ids.append(cluster_id)  
                    clusters[cluster_id] = [point]  
                    cluster_indx[cluster_id] = [i]

            cluster_max_indxs = []
            cluster_max_id = None
            for cluster_id, indxs in cluster_indx.items():
                if len(indxs) > len(cluster_max_indxs):
                    cluster_max_indxs = indxs
                    cluster_max_id = cluster_id
            cluster_max_indxs.sort()
            img_coords = img_coords[cluster_max_indxs]
            point_coords = point_coords[cluster_max_indxs]
            image_C = np.ascontiguousarray(img_coords[:,:2]).reshape((-1,1,2))
            ret, rvec, tvec = cv2.solvePnP(point_coords, image_C, mtx_left, dist_left)
            image_points, _ = cv2.projectPoints(axis_coord, rvec, tvec, mtx_left, dist_left)
            image_points = image_points.reshape(-1, 2).astype(np.int16)
            cv2.line(img, (image_points[0,0], image_points[0,1]), (image_points[1,0], image_points[1,1]), (0,0,255), 5) 
            cv2.line(img, (image_points[0,0], image_points[0,1]), (image_points[2,0], image_points[2,1]), (0,255,0), 5) 
            cv2.line(img, (image_points[0,0], image_points[0,1]), (image_points[3,0], image_points[3,1]), (255,0,0), 5) 

            # 计算旋转矩阵和变换矩阵
            R_aruco2cam, _ = cv2.Rodrigues(rvec)
            t_aruco2cam = tvec
            # 计算相机到 ArUco 的变换
            R_cam2aruco = R_aruco2cam.T
            t_cam2aruco = -R_aruco2cam.T @ t_aruco2cam

            cam_extrinsic_params[cam_id] = {
                'R_aruco2cam': R_aruco2cam,
                't_aruco2cam': t_aruco2cam,
                'R_cam2aruco': R_cam2aruco,
                't_cam2aruco': t_cam2aruco,
            }

    np.savez('cam_extrinsic_params.npz', cam_extrinsic_params)

'''# Lesson.md
## Camera Calibration Notes
1. Intrinsic calibration requires:
   - Chessboard pattern (9x6 corners)
   - Multiple images from different angles
   - Image resolution: 1920x1080

2. Extrinsic calibration setup:
   - ArUco marker board configuration:
     - Board length: 160
     - Gap between markers: 30
   - Uses DICT_ARUCO_ORIGINAL dictionary
   - Visualization with 500-unit axis
   '''
