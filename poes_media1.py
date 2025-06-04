import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
import matplotlib.pyplot as plt
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import time
import os
import pandas as pd  # 用于读取CSV文件
import pyvista as pv  # 用于3D可视化
from pyvistaqt import BackgroundPlotter  # 用于交互式3D绘图
import math
from matplotlib.animation import FuncAnimation
from collections import defaultdict  
import glob

# 设置环境变量
os.environ['PATH'] += ';D:\\software\\ffmpeg-7.1-essentials_build'
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = 'D:\\software\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe'

def calibrate_single_camera(images_folder, pattern_size=(9,6), square_size=30.0):
    """
    單相機標定函數
    
    Args:
        images_folder: 標定圖片所在文件夾路徑
        pattern_size: 棋盤格內角點數量(寬,高)
        square_size: 棋盤格方格實際尺寸(mm)
        
    Returns:
        ret: 標定誤差
        mtx: 相機內參矩陣 
        dist: 畸變係數
        rvecs: 旋轉向量
        tvecs: 平移向量
    """
    # 準備標定板角點的世界坐標
    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
    objp = objp * square_size  # 轉換為實際尺寸
    
    # 存儲所有圖像的3D點和2D點
    objpoints = [] # 3D點
    imgpoints = [] # 2D點
    
    # 獲取所有校正圖片
    images = glob.glob(os.path.join(images_folder, '*.jpeg'))
    if not images:
        images = glob.glob(os.path.join(images_folder, '*.png'))
    
    # 打印找到的圖片數量
    print(f"在 {images_folder} 中找到 {len(images)} 張圖片")
    
    # 遍歷每張標定圖片
    for image_filename in images:
        img = cv2.imread(image_filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 在灰度圖中查找棋盤格角點
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        # 如果成功找到所有角點
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            # 在圖片上繪製檢測到的棋盤格角點
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow('Corners', cv2.resize(img, (800,600)))
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    # 執行相機標定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # 計算重投影誤差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    print(f"平均重投影誤差: {mean_error/len(objpoints)}")
    
    return ret, mtx, dist, rvecs, tvecs

def calibrate_three_cameras(images_folders, pattern_size=(9,6), square_size=30.0, save_file=None):
    """
    三相機標定主函數
    
    Args:
        images_folders: 包含三個相機標定圖片所在文件夾路徑的列表
        pattern_size: 棋盤格內角點數量(寬,高)
        square_size: 棋盤格方格實際尺寸(mm)
        save_file: 保存標定結果的文件路徑，如果為None則保存到當前工作目錄
        
    Returns:
        mtx1: 相機1內參矩陣
        dist1: 相機1畸變係數
        mtx2: 相機2內參矩陣
        dist2: 相機2畸變係數
        mtx3: 相機3內參矩陣
        dist3: 相機3畸變係數
    """
    if len(images_folders) != 3:
        raise ValueError("必須提供三個相機的圖片文件夾路徑")
    
    # 提取文件夾路徑
    cam1_folder, cam2_folder, cam3_folder = images_folders
    
    print("開始相機1標定...")
    ret1, mtx1, dist1, rvecs1, tvecs1 = calibrate_single_camera(cam1_folder, pattern_size, square_size)
    
    print("\n開始相機2標定...")
    ret2, mtx2, dist2, rvecs2, tvecs2 = calibrate_single_camera(cam2_folder, pattern_size, square_size)
    
    print("\n開始相機3標定...")
    ret3, mtx3, dist3, rvecs3, tvecs3 = calibrate_single_camera(cam3_folder, pattern_size, square_size)
    
    # 设置保存路径
    if save_file is None:
        save_file = 'three_camera_calibration.npz'
    
    # 保存標定結果
    np.savez(save_file,
             mtx1=mtx1, dist1=dist1,
             mtx2=mtx2, dist2=dist2,
             mtx3=mtx3, dist3=dist3)
    print(f"\n標定結果已保存到: {os.path.abspath(save_file)}")
    
    # 打印相機內參
    print("\n相機1內參矩陣:")
    print(mtx1)
    print("相機1畸變係數:")
    print(dist1)
    
    print("\n相機2內參矩陣:")
    print(mtx2)
    print("相機2畸變係數:")
    print(dist2)
    
    print("\n相機3內參矩陣:")
    print(mtx3)
    print("相機3畸變係數:")
    print(dist3)
    
    return mtx1, dist1, mtx2, dist2, mtx3, dist3

def test_calibration_image(image_path, camera_params):
    """
    測試標定結果
    
    Args:
        image_path: 測試圖片路徑
        camera_params: 相機參數(mtx, dist)
    """
    mtx, dist = camera_params
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # 使用cv2.getOptimalNewCameraMatrix獲取優化後的相機矩陣和ROI區域
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    # 使用cv2.undistort對圖像進行校正
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # 從ROI中獲取裁剪區域的座標和大小
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    cv2.imshow('Original', cv2.resize(img, (800,600)))
    cv2.imshow('Calibrated', cv2.resize(dst, (800,600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_camera_params(file_path):
    """
    从文件加载三相机参数
    Args:
        file_path: 相机参数文件路径
    Returns:
        mtx1: 相机1内参矩阵
        dist1: 相机1畸变系数
        mtx2: 相机2内参矩阵 
        dist2: 相机2畸变系数
        mtx3: 相机3内参矩阵
        dist3: 相机3畸变系数
    """
    data = np.load(file_path)
    return data['mtx1'], data['dist1'], data['mtx2'], data['dist2'], data['mtx3'], data['dist3']

def get_aruco_axis(img_L, img_M, img_R, aruco_detector, board_coord, cams_params):
    """
    从三个相机图像中检测ArUco标记并估计其姿态
    Args:
        img_L: 左相机图像
        img_M: 中间相机图像
        img_R: 右相机图像 
        aruco_detector: ArUco检测器
        board_coord: ArUco标记板坐标字典
        cams_params: 相机参数元组(mtx1, dist1, mtx2, dist2, mtx3, dist3)
    Returns:
        R_aruco2camL: ArUco到左相机的旋转矩阵
        t_aruco2camL: ArUco到左相机的平移向量
        R_aruco2camM: ArUco到中间相机的旋转矩阵
        t_aruco2camM: ArUco到中间相机的平移向量
        R_aruco2camR: ArUco到右相机的旋转矩阵
        t_aruco2camR: ArUco到右相机的平移向量
        R_camL2aruco: 左相机到ArUco的旋转矩阵
        t_camL2aruco: 左相机到ArUco的平移向量
        R_camM2aruco: 中间相机到ArUco的旋转矩阵
        t_camM2aruco: 中间相机到ArUco的平移向量
        R_camR2aruco: 右相机到ArUco的旋转矩阵
        t_camR2aruco: 右相机到ArUco的平移向量
        img_L: 标注后的左图像
        img_M: 标注后的中间图像
        img_R: 标注后的右图像
    """
    # 解包相机参数
    (mtx1, dist1, mtx2, dist2, mtx3, dist3) = cams_params
    
    # 定义坐标轴点
    axis_coord = np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ],dtype=np.float32)
    axis_coord = axis_coord * 500  # 放大坐标轴显示
    
    # 在三个相机图像中检测ArUco标记
    corners_L, ids_L, rejectedImgPoints_L = aruco_detector.detectMarkers(img_L)
    corners_M, ids_M, rejectedImgPoints_M = aruco_detector.detectMarkers(img_M)
    corners_R, ids_R, rejectedImgPoints_R = aruco_detector.detectMarkers(img_R)

    # 对左相机图像进行亚像素角点精化处理
    if ids_L is not None and len(ids_L) > 0:
        gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY) if len(img_L.shape) == 3 else img_L
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for i in range(len(corners_L)):
            refined_corners = cv2.cornerSubPix(gray_L, corners_L[i], (3, 3), (-1, -1), criteria)
            corners_L[i] = refined_corners

    # 对中间相机图像进行亚像素角点精化处理
    if ids_M is not None and len(ids_M) > 0:
        gray_M = cv2.cvtColor(img_M, cv2.COLOR_BGR2GRAY) if len(img_M.shape) == 3 else img_M
        for i in range(len(corners_M)):
            corners_M[i] = cv2.cornerSubPix(gray_M, corners_M[i], (3, 3), (-1, -1), criteria)

    # 对右相机图像进行亚像素角点精化处理
    if ids_R is not None and len(ids_R) > 0:
        gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY) if len(img_R.shape) == 3 else img_R
        for i in range(len(corners_R)):
            corners_R[i] = cv2.cornerSubPix(gray_R, corners_R[i], (3, 3), (-1, -1), criteria)
    
    # 初始化存储列表
    img_coords_L, point_coords_L, aruco_ps_L_2camL = [], [], []
    img_coords_M, point_coords_M, aruco_ps_M_2camM = [], [], []
    img_coords_R, point_coords_R, aruco_ps_R_2camR = [], [], []

    # 处理左相机图像
    if ids_L is not None:
        for i in range(len(ids_L)):
            if ids_L[i][0] not in board_coord.keys():
                continue
            
            tmp_marker = corners_L[i][0]
            
            tmp_marker_tl = (int(tmp_marker[0][0]), int(tmp_marker[0][1]))
            tmp_marker_tr = (int(tmp_marker[1][0]), int(tmp_marker[1][1]))
            tmp_marker_br = (int(tmp_marker[2][0]), int(tmp_marker[2][1]))
            tmp_marker_bl = (int(tmp_marker[3][0]), int(tmp_marker[3][1]))
            
            cv2.circle(img_L, tmp_marker_tl, 10, (0, 0, 255), -1)
            cv2.circle(img_L, tmp_marker_tr, 10, (0, 255, 0), -1)
            cv2.circle(img_L, tmp_marker_br, 10, (255, 0, 0), -1)
            cv2.circle(img_L, tmp_marker_bl, 10, (0, 170, 255), -1)
            
            cv2.putText(img_L, f"ID: {ids_L[i][0]}", (int(tmp_marker_tl[0] + 10), int(tmp_marker_tl[1] + 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            
            img_coord = np.array([tmp_marker[j] for j in range(4)])
            img_coords_L.append(np.squeeze(img_coord))
            
            tem_coord = np.hstack((board_coord[ids_L[i][0]], np.zeros(len(board_coord[ids_L[i][0]]))[:,None]))
            point_coords_L.append(tem_coord)
            
            image_C_L = np.ascontiguousarray(img_coord[:,:2]).reshape((-1,1,2))
            
            ret_L, rvec_L, tvec_L = cv2.solvePnP(tem_coord, image_C_L, mtx1, dist1)
            R_aruco2camL, _ = cv2.Rodrigues(rvec_L)
            t_aruco2camL = tvec_L
            
            aruco_p_L_2camL = np.dot(R_aruco2camL, tem_coord.T).T + t_aruco2camL.T
            aruco_ps_L_2camL.append(aruco_p_L_2camL)

    # 处理中间相机图像
    if ids_M is not None:
        for i in range(len(ids_M)):
            if ids_M[i][0] not in board_coord.keys():
                continue
                
            tmp_marker = corners_M[i][0]
            
            tmp_marker_tl = (int(tmp_marker[0][0]), int(tmp_marker[0][1]))
            tmp_marker_tr = (int(tmp_marker[1][0]), int(tmp_marker[1][1]))
            tmp_marker_br = (int(tmp_marker[2][0]), int(tmp_marker[2][1]))
            tmp_marker_bl = (int(tmp_marker[3][0]), int(tmp_marker[3][1]))
            
            cv2.circle(img_M, tmp_marker_tl, 10, (0, 0, 255), -1)
            cv2.circle(img_M, tmp_marker_tr, 10, (0, 255, 0), -1)
            cv2.circle(img_M, tmp_marker_br, 10, (255, 0, 0), -1)
            cv2.circle(img_M, tmp_marker_bl, 10, (0, 170, 255), -1)
            
            cv2.putText(img_M, f"ID: {ids_M[i][0]}", (int(tmp_marker_tl[0] + 10), int(tmp_marker_tl[1] + 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            
            img_coord = np.array([tmp_marker[j] for j in range(4)])
            img_coords_M.append(np.squeeze(img_coord))
            
            tem_coord = np.hstack((board_coord[ids_M[i][0]], np.zeros(len(board_coord[ids_M[i][0]]))[:,None]))
            point_coords_M.append(tem_coord)
            
            image_C_M = np.ascontiguousarray(img_coord[:,:2]).reshape((-1,1,2))
            ret_M, rvec_M, tvec_M = cv2.solvePnP(tem_coord, image_C_M, mtx2, dist2)
            R_aruco2camM, _ = cv2.Rodrigues(rvec_M)
            t_aruco2camM = tvec_M
            
            aruco_p_M_2camM = np.dot(R_aruco2camM, tem_coord.T).T + t_aruco2camM.T
            aruco_ps_M_2camM.append(aruco_p_M_2camM)

    # 处理右相机图像
    if ids_R is not None:
        for i in range(len(ids_R)):
            if ids_R[i][0] not in board_coord.keys():
                continue
                
            tmp_marker = corners_R[i][0]
            
            tmp_marker_tl = (int(tmp_marker[0][0]), int(tmp_marker[0][1]))
            tmp_marker_tr = (int(tmp_marker[1][0]), int(tmp_marker[1][1]))
            tmp_marker_br = (int(tmp_marker[2][0]), int(tmp_marker[2][1]))
            tmp_marker_bl = (int(tmp_marker[3][0]), int(tmp_marker[3][1]))
            
            cv2.circle(img_R, tmp_marker_tl, 10, (0, 0, 255), -1)
            cv2.circle(img_R, tmp_marker_tr, 10, (0, 255, 0), -1)
            cv2.circle(img_R, tmp_marker_br, 10, (255, 0, 0), -1)
            cv2.circle(img_R, tmp_marker_bl, 10, (0, 170, 255), -1)
            
            cv2.putText(img_R, f"ID: {ids_R[i][0]}", (int(tmp_marker_tl[0] + 10), int(tmp_marker_tl[1] + 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            
            img_coord = np.array([tmp_marker[j] for j in range(4)])
            img_coords_R.append(np.squeeze(img_coord))
            
            tem_coord = np.hstack((board_coord[ids_R[i][0]], np.zeros(len(board_coord[ids_R[i][0]]))[:,None]))
            point_coords_R.append(tem_coord)
            
            image_C_R = np.ascontiguousarray(img_coord[:,:2]).reshape((-1,1,2))
            ret_R, rvec_R, tvec_R = cv2.solvePnP(tem_coord, image_C_R, mtx3, dist3)
            R_aruco2camR, _ = cv2.Rodrigues(rvec_R)
            t_aruco2camR = tvec_R
            
            aruco_p_R_2camR = np.dot(R_aruco2camR, tem_coord.T).T + t_aruco2camR.T
            aruco_ps_R_2camR.append(aruco_p_R_2camR)

    # 合并检测到的点
    img_coords_L = np.concatenate(img_coords_L, axis=0) if img_coords_L else np.array([])
    img_coords_M = np.concatenate(img_coords_M, axis=0) if img_coords_M else np.array([])
    img_coords_R = np.concatenate(img_coords_R, axis=0) if img_coords_R else np.array([])
    
    img_coords_L = np.hstack((img_coords_L, np.ones((len(img_coords_L), 1)))) if len(img_coords_L) > 0 else np.array([])
    img_coords_M = np.hstack((img_coords_M, np.ones((len(img_coords_M), 1)))) if len(img_coords_M) > 0 else np.array([])
    img_coords_R = np.hstack((img_coords_R, np.ones((len(img_coords_R), 1)))) if len(img_coords_R) > 0 else np.array([])
    
    point_coords_L = np.concatenate(point_coords_L, axis=0) if point_coords_L else np.array([])
    point_coords_M = np.concatenate(point_coords_M, axis=0) if point_coords_M else np.array([])
    point_coords_R = np.concatenate(point_coords_R, axis=0) if point_coords_R else np.array([])
    
    aruco_ps_L_2camL = np.concatenate(aruco_ps_L_2camL, axis=0) if aruco_ps_L_2camL else np.array([])
    aruco_ps_M_2camM = np.concatenate(aruco_ps_M_2camM, axis=0) if aruco_ps_M_2camM else np.array([])
    aruco_ps_R_2camR = np.concatenate(aruco_ps_R_2camR, axis=0) if aruco_ps_R_2camR else np.array([])

    # 检查是否所有相机都检测到标记
    if len(img_coords_L) == 0 or len(point_coords_L) == 0:
        print("左图像中未检测到 ArUco 标记")
        return (None,) * 15

    if len(img_coords_M) == 0 or len(point_coords_M) == 0:
        print("中间图像中未检测到 ArUco 标记")
        return (None,) * 15

    if len(img_coords_R) == 0 or len(point_coords_R) == 0:
        print("右图像中未检测到 ArUco 标记")
        return (None,) * 15

    # 对每个相机进行聚类处理
    # 左相机聚类
    clusters_L = defaultdict(list)
    cluster_ids_L = []
    cluster_indx_L = {}
    
    for i, point in enumerate(aruco_ps_L_2camL):
        new_cluster = True
        
        for cluster_id, cluster_points in clusters_L.items():
            for cluster_point in cluster_points:
                if np.linalg.norm(point - cluster_point) <= board_coord[5][3,1]:
                    clusters_L[cluster_id].append(point)
                    cluster_indx_L[cluster_id].append(i)
                    new_cluster = False
                    break
            
            if not new_cluster:
                break
        
        if new_cluster:
            cluster_id = len(cluster_ids_L)
            cluster_ids_L.append(cluster_id)
            clusters_L[cluster_id] = [point]
            cluster_indx_L[cluster_id] = [i]

    # 中间相机聚类
    clusters_M = defaultdict(list)
    cluster_ids_M = []
    cluster_indx_M = {}
    for i, point in enumerate(aruco_ps_M_2camM):
        new_cluster = True
        for cluster_id, cluster_points in clusters_M.items():
            for cluster_point in cluster_points:
                if np.linalg.norm(point-cluster_point) <= board_coord[5][3,1]:
                    clusters_M[cluster_id].append(point)
                    cluster_indx_M[cluster_id].append(i)
                    new_cluster = False
                    break
            if not new_cluster:
                break
        if new_cluster:
            cluster_id = len(cluster_ids_M)
            cluster_ids_M.append(cluster_id)
            clusters_M[cluster_id] = [point]
            cluster_indx_M[cluster_id] = [i]

    # 右相机聚类    
    clusters_R = defaultdict(list)
    cluster_ids_R = []
    cluster_indx_R = {}
    for i, point in enumerate(aruco_ps_R_2camR):
        new_cluster = True
        for cluster_id, cluster_points in clusters_R.items():
            for cluster_point in cluster_points:
                if np.linalg.norm(point-cluster_point) <= board_coord[5][3,1]:
                    clusters_R[cluster_id].append(point)
                    cluster_indx_R[cluster_id].append(i)
                    new_cluster = False
                    break
            if not new_cluster:
                break
        if new_cluster:
            cluster_id = len(cluster_ids_R)
            cluster_ids_R.append(cluster_id)
            clusters_R[cluster_id] = [point]
            cluster_indx_R[cluster_id] = [i]

    # 获取最大聚类的索引
    cluster_max_indxs_L = max(cluster_indx_L.values(), key=len) if cluster_indx_L else []
    cluster_max_indxs_M = max(cluster_indx_M.values(), key=len) if cluster_indx_M else []
    cluster_max_indxs_R = max(cluster_indx_R.values(), key=len) if cluster_indx_R else []

    # 对每个相机的最大聚类索引进行排序
    cluster_max_indxs_L.sort()
    cluster_max_indxs_M.sort()
    cluster_max_indxs_R.sort()

    # 使用最大聚类的索引选择对应的点
    img_coords_L = img_coords_L[cluster_max_indxs_L]
    img_coords_M = img_coords_M[cluster_max_indxs_M]
    img_coords_R = img_coords_R[cluster_max_indxs_R]

    point_coords_L = point_coords_L[cluster_max_indxs_L]
    point_coords_M = point_coords_M[cluster_max_indxs_M]
    point_coords_R = point_coords_R[cluster_max_indxs_R]

    # 使用选定的点解决每个相机的PnP问题
    image_C_L = np.ascontiguousarray(img_coords_L[:,:2]).reshape((-1,1,2))
    ret_L, rvec_L, tvec_L = cv2.solvePnP(point_coords_L, image_C_L, mtx1, dist1)
    
    image_C_M = np.ascontiguousarray(img_coords_M[:,:2]).reshape((-1,1,2))
    ret_M, rvec_M, tvec_M = cv2.solvePnP(point_coords_M, image_C_M, mtx2, dist2)
    
    image_C_R = np.ascontiguousarray(img_coords_R[:,:2]).reshape((-1,1,2))
    ret_R, rvec_R, tvec_R = cv2.solvePnP(point_coords_R, image_C_R, mtx3, dist3)

    # 在每个相机图像上绘制坐标轴
    image_points, _ = cv2.projectPoints(axis_coord, rvec_L, tvec_L, mtx1, dist1)
    image_points = image_points.reshape(-1, 2).astype(np.int16)
    cv2.line(img_L, (image_points[0,0], image_points[0,1]), (image_points[1,0], image_points[1,1]), (0,0,255), 5)
    cv2.line(img_L, (image_points[0,0], image_points[0,1]), (image_points[2,0], image_points[2,1]), (0,255,0), 5)
    cv2.line(img_L, (image_points[0,0], image_points[0,1]), (image_points[3,0], image_points[3,1]), (255,0,0), 5)

    image_points, _ = cv2.projectPoints(axis_coord, rvec_M, tvec_M, mtx2, dist2)
    image_points = image_points.reshape(-1, 2).astype(np.int16)
    cv2.line(img_M, (image_points[0,0], image_points[0,1]), (image_points[1,0], image_points[1,1]), (0,0,255), 5)
    cv2.line(img_M, (image_points[0,0], image_points[0,1]), (image_points[2,0], image_points[2,1]), (0,255,0), 5)
    cv2.line(img_M, (image_points[0,0], image_points[0,1]), (image_points[3,0], image_points[3,1]), (255,0,0), 5)

    image_points, _ = cv2.projectPoints(axis_coord, rvec_R, tvec_R, mtx3, dist3)
    image_points = image_points.reshape(-1, 2).astype(np.int16)
    cv2.line(img_R, (image_points[0,0], image_points[0,1]), (image_points[1,0], image_points[1,1]), (0,0,255), 5)
    cv2.line(img_R, (image_points[0,0], image_points[0,1]), (image_points[2,0], image_points[2,1]), (0,255,0), 5)
    cv2.line(img_R, (image_points[0,0], image_points[0,1]), (image_points[3,0], image_points[3,1]), (255,0,0), 5)

    # 计算最终的旋转矩阵和变换矩阵
    R_aruco2camL, _ = cv2.Rodrigues(rvec_L)
    t_aruco2camL = tvec_L
    R_aruco2camM, _ = cv2.Rodrigues(rvec_M)
    t_aruco2camM = tvec_M
    R_aruco2camR, _ = cv2.Rodrigues(rvec_R)
    t_aruco2camR = tvec_R

    # 计算从相机坐标系到ArUco坐标系的变换
    R_camL2aruco = R_aruco2camL.T
    t_camL2aruco = -R_aruco2camL.T @ t_aruco2camL
    R_camM2aruco = R_aruco2camM.T
    t_camM2aruco = -R_aruco2camM.T @ t_aruco2camM
    R_camR2aruco = R_aruco2camR.T
    t_camR2aruco = -R_aruco2camR.T @ t_aruco2camR

    # 返回所有计算得到的变换矩阵和处理后的图像
    return (R_aruco2camL, t_aruco2camL,
            R_aruco2camM, t_aruco2camM,
            R_aruco2camR, t_aruco2camR,
            R_camL2aruco, t_camL2aruco,
            R_camM2aruco, t_camM2aruco,
            R_camR2aruco, t_camR2aruco,
            img_L, img_M, img_R)

def calculate_reprojection_error(points_3d, points_2d, P, mtx, dist):
    """
    計算重投影誤差
    Args:
        points_3d: 3D點坐標 (N, 3)
        points_2d: 2D點坐標 (N, 2)
        P: 投影矩陣 (3, 4)
        mtx: 相機內參矩陣
        dist: 畸變係數
    Returns:
        rmse: 均方根誤差
    """
    points_3d_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_cam = np.dot(points_3d_homo, P.T)
    points_proj = points_cam[:, :2] / points_cam[:, 2:]
    points_proj_dist = cv2.projectPoints(points_3d, P[:, :3], P[:, 3], mtx, dist)[0].reshape(-1, 2)
    error = np.linalg.norm(points_2d - points_proj_dist, axis=1)
    rmse = np.sqrt(np.mean(error ** 2))
    return rmse

def process_frame(detector, frame_left, frame_middle, frame_right, img_L, img_M, img_R, cams_params, cam_P):
    """
    處理三個相機的單幀圖像並計算重投影誤差
    Args:
        detector: 姿態檢測器對象
        frame_left: 左相機的幀圖像
        frame_middle: 中間相機的幀圖像
        frame_right: 右相機的幀圖像
        img_L: 左相機的輸出圖像
        img_M: 中間相機的輸出圖像
        img_R: 右相機的輸出圖像
        cams_params: 相機參數元組(mtx1,dist1,mtx2,dist2,mtx3,dist3)
        cam_P: 相機投影矩陣元組(R_L,t_L,R_M,t_M,R_R,t_R)
    Returns:
        points_3d: 三維重建點雲
        img_L, img_M, img_R: 標註後的圖像
        (rmse_left, rmse_middle, rmse_right): 三個相機的重投影誤差
    """
    mp_images = [
        mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for frame in [frame_left, frame_middle, frame_right]
    ]
    
    detection_results = [detector.detect(img) for img in mp_images]
    
    poses = []
    frames = [frame_left, frame_middle, frame_right]
    cam_names = ["左側", "中間", "右側"]
    
    detection_failed = False
    failed_cameras = []
    
    for i, result in enumerate(detection_results):
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            pose = np.array([[landmark.x * frames[i].shape[1], landmark.y * frames[i].shape[0]]
                            for landmark in result.pose_landmarks[0]])
            poses.append(pose)
        else:
            detection_failed = True
            failed_cameras.append(cam_names[i])
    
    if detection_failed:
        print(f"以下相機未檢測到人體姿勢: {', '.join(failed_cameras)}")
        return np.zeros((33, 3)), img_L, img_M, img_R, (0, 0, 0)
    
    (mtx1, dist1, mtx2, dist2, mtx3, dist3) = cams_params
    (R_aruco2camL, t_aruco2camL, R_aruco2camM, t_aruco2camM, R_aruco2camR, t_aruco2camR) = cam_P
    
    points_3d = triangulate_points_three_cameras(
        poses[0], poses[1], poses[2],
        mtx1, dist1, mtx2, dist2, mtx3, dist3,
        np.hstack((R_aruco2camL, t_aruco2camL)),
        np.hstack((R_aruco2camM, t_aruco2camM)),
        np.hstack((R_aruco2camR, t_aruco2camR))
    )
    
    images = [img_L, img_M, img_R]
    for i, pose in enumerate(poses):
        for mark_i in range(33):
            mark_coord = (int(pose[mark_i,0]), int(pose[mark_i,1]))
            cv2.circle(images[i], mark_coord, 10, (0, 0, 255), -1)
    
    rmse_left = calculate_reprojection_error(
        points_3d, poses[0],
        np.hstack((R_aruco2camL, t_aruco2camL)),
        mtx1, dist1
    )
    
    rmse_middle = calculate_reprojection_error(
        points_3d, poses[1],
        np.hstack((R_aruco2camM, t_aruco2camM)),
        mtx2, dist2
    )
    
    rmse_right = calculate_reprojection_error(
        points_3d, poses[2],
        np.hstack((R_aruco2camR, t_aruco2camR)),
        mtx3, dist3
    )
    
    return points_3d, img_L, img_M, img_R, (rmse_left, rmse_middle, rmse_right)

def triangulate_points_three_cameras(points_left, points_middle, points_right,
                                   mtx1, dist1, mtx2, dist2, mtx3, dist3,
                                   P1, P2, P3):
    """
    使用三個相機進行三角測量
    
    Args:
        points_left/middle/right: 三個相機的2D點
        mtx1/2/3, dist1/2/3: 相機內參和畸變係數
        P1/2/3: 投影矩陣
    Returns:
        points_3d: 三維點
    """
    points_3d = []
    
    for pt_left, pt_middle, pt_right in zip(points_left, points_middle, points_right):
        pt_left_undist = cv2.undistortPoints(pt_left.reshape(1, 1, 2), mtx1, dist1)
        pt_middle_undist = cv2.undistortPoints(pt_middle.reshape(1, 1, 2), mtx2, dist2)
        pt_right_undist = cv2.undistortPoints(pt_right.reshape(1, 1, 2), mtx3, dist3)
        
        A = np.zeros((6, 4))
        
        A[0] = pt_left_undist[0, 0, 0] * P1[2] - P1[0]
        A[1] = pt_left_undist[0, 0, 1] * P1[2] - P1[1]
        
        A[2] = pt_middle_undist[0, 0, 0] * P2[2] - P2[0]
        A[3] = pt_middle_undist[0, 0, 1] * P2[2] - P2[1]
        
        A[4] = pt_right_undist[0, 0, 0] * P3[2] - P3[0]
        A[5] = pt_right_undist[0, 0, 1] * P3[2] - P3[1]
        
        _, _, Vt = np.linalg.svd(A)
        point_4d = Vt[-1]
        point_3d = (point_4d / point_4d[3])[:3]
        points_3d.append(point_3d)
        
    return np.array(points_3d)

def process_videos(video_paths, output_folder, calib_file, aruco_dict_type=cv2.aruco.DICT_5X5_250):
    """
    處理多個相機的視頻文件
    
    Args:
        video_paths: 包含三個相機視頻路徑的列表
        output_folder: 輸出文件夾
        calib_file: 相機標定參數文件
        aruco_dict_type: ArUco字典類型
    
    Returns:
        points_3d_list: 每幀的3D姿勢關鍵點列表
    """
    # 創建輸出文件夾
    os.makedirs(output_folder, exist_ok=True)
    
    # 加載相機參數
    mtx1, dist1, mtx2, dist2, mtx3, dist3 = load_camera_params(calib_file)
    cams_params = (mtx1, dist1, mtx2, dist2, mtx3, dist3)
    
    # 配置MediaPipe姿態檢測器
    base_options = python.BaseOptions(model_asset_path='C:\\Users\\godli\\Downloads\\pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    detector = vision.PoseLandmarker.create_from_options(options)
    
    # 設置ArUco檢測器
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # 設置ArUco標記板坐標
    board_coord = {}
    board_coord[5] = np.array([
        [0.00, 0.00],
        [58.0, 0.00],
        [58.0, 58.0],
        [0.00, 58.0]
    ], dtype=np.float32)
    
    # 打開視頻
    cap_left = cv2.VideoCapture(video_paths[0])
    cap_middle = cv2.VideoCapture(video_paths[1])
    cap_right = cv2.VideoCapture(video_paths[2])
    
    # 獲取視頻信息
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in [cap_left, cap_middle, cap_right])
    print(f"處理總幀數: {n_frames}")
    
    # 初始化投影矩陣
    R_aruco2camL, t_aruco2camL = None, None
    R_aruco2camM, t_aruco2camM = None, None
    R_aruco2camR, t_aruco2camR = None, None
    
    # 列表用於存儲每幀的三維姿勢點
    points_3d_list = []
    
    # 不需要每幀都處理
    frame_step = 10
    
    # 處理每一幀
    for frame_idx in range(0, n_frames, frame_step):
        # 設置視頻位置
        cap_left.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap_middle.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # 讀取幀
        ret_left, frame_left = cap_left.read()
        ret_middle, frame_middle = cap_middle.read()
        ret_right, frame_right = cap_right.read()
        
        if not (ret_left and ret_middle and ret_right):
            print(f"跳過幀 {frame_idx}: 無法從一個或多個視頻讀取")
            continue
        
        # 複製圖像用於顯示
        img_L = frame_left.copy()
        img_M = frame_middle.copy()
        img_R = frame_right.copy()
        
        # 檢測 ArUco 標記並估計其姿態
        if R_aruco2camL is None or frame_idx % 30 == 0:
            (R_aruco2camL, t_aruco2camL,
             R_aruco2camM, t_aruco2camM,
             R_aruco2camR, t_aruco2camR,
             _, _, _,
             _, _, _,
             img_L, img_M, img_R) = get_aruco_axis(img_L, img_M, img_R, aruco_detector, board_coord, cams_params)
            
            if R_aruco2camL is None:
                print(f"跳過幀 {frame_idx}: 無法檢測到 ArUco 標記")
                continue
        
        # 處理幀圖像並獲取三維姿勢點
        cam_P = (R_aruco2camL, t_aruco2camL, R_aruco2camM, t_aruco2camM, R_aruco2camR, t_aruco2camR)
        points_3d, img_L, img_M, img_R, rmse = process_frame(detector, frame_left, frame_middle, frame_right, img_L, img_M, img_R, cams_params, cam_P)
        
        # 打印重投影誤差
        if rmse[0] > 0:
            print(f"幀 {frame_idx} - 左側相機RMSE: {rmse[0]:.4f}, 中間相機RMSE: {rmse[1]:.4f}, 右側相機RMSE: {rmse[2]:.4f}")
        
        # 將三維姿勢點保存到列表
        points_3d_list.append(points_3d)
        
        # 保存結果圖像
        output_path = os.path.join(output_folder, f"frame_{frame_idx:04d}")
        cv2.imwrite(f"{output_path}_left.jpg", img_L)
        cv2.imwrite(f"{output_path}_middle.jpg", img_M)
        cv2.imwrite(f"{output_path}_right.jpg", img_R)
    
    # 關閉視頻
    cap_left.release()
    cap_middle.release()
    cap_right.release()
    
    # 保存3D點到CSV文件
    output_csv = os.path.join(output_folder, "points_3d.csv")
    with open(output_csv, 'w') as f:
        f.write("frame_idx,point_idx,x,y,z\n")
        for i, points in enumerate(points_3d_list):
            frame_idx = i * frame_step
            for j, point in enumerate(points):
                f.write(f"{frame_idx},{j},{point[0]},{point[1]},{point[2]}\n")
    
    print(f"已保存3D點到 {output_csv}")
    return points_3d_list

def visualize_3d_animation_three_cameras(points_3d_list, output_file, 
                                        R_aruco2camL, t_aruco2camL, 
                                        R_aruco2camM, t_aruco2camM, 
                                        R_aruco2camR, t_aruco2camR,
                                        interval=50):
    """
    将姿态点动画可视化为3D动画
    
    Args:
        points_3d_list: 每帧的3D姿态关键点列表
        output_file: 输出文件路径
        R_aruco2camL: ArUco到左相机的旋转矩阵
        t_aruco2camL: ArUco到左相机的平移向量
        R_aruco2camM: ArUco到中间相机的旋转矩阵
        t_aruco2camM: ArUco到中间相机的平移向量 
        R_aruco2camR: ArUco到右相机的旋转矩阵
        t_aruco2camR: ArUco到右相机的平移向量
        interval: 帧间隔(毫秒)
    """
    # 检查输入数据
    if not points_3d_list:
        print("没有3D点数据可视化")
        return
    
    # 设置连接线（MediaPipe姿态关键点之间的连接）
    connections = [
        (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), 
        (4, 5), (5, 6), (6, 8), (9, 10), (11, 12),
        (11, 13), (11, 23), (12, 14), (12, 24), (13, 15),
        (14, 16), (15, 17), (15, 19), (15, 21), (16, 18),
        (16, 20), (16, 22), (17, 19), (18, 20), (23, 24),
        (23, 25), (24, 26), (25, 27), (26, 28), (27, 29),
        (27, 31), (28, 30), (28, 32)
    ]
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标范围
    all_points = np.concatenate(points_3d_list, axis=0)
    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)
    
    # 添加一些余量
    margin = 100
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 绘制相机位置
    camera_color = ['r', 'g', 'b']
    camera_label = ['Left Camera', 'Middle Camera', 'Right Camera']
    camera_pos = [
        (np.linalg.inv(R_aruco2camL), -np.linalg.inv(R_aruco2camL) @ t_aruco2camL),
        (np.linalg.inv(R_aruco2camM), -np.linalg.inv(R_aruco2camM) @ t_aruco2camM),
        (np.linalg.inv(R_aruco2camR), -np.linalg.inv(R_aruco2camR) @ t_aruco2camR)
    ]
    
    for i, (R, t) in enumerate(camera_pos):
        ax.scatter(t[0], t[1], t[2], c=camera_color[i], marker='^', s=100, label=camera_label[i])
        
        # 绘制相机轴
        axis_length = 50
        X_axis = R @ np.array([axis_length, 0, 0]) + t
        Y_axis = R @ np.array([0, axis_length, 0]) + t
        Z_axis = R @ np.array([0, 0, axis_length]) + t
        
        ax.plot([t[0], X_axis[0]], [t[1], X_axis[1]], [t[2], X_axis[2]], c='r', linewidth=2)
        ax.plot([t[0], Y_axis[0]], [t[1], Y_axis[1]], [t[2], Y_axis[2]], c='g', linewidth=2)
        ax.plot([t[0], Z_axis[0]], [t[1], Z_axis[1]], [t[2], Z_axis[2]], c='b', linewidth=2)
    
    # 初始化绘图元素
    scatter = ax.scatter([], [], [], c='k', s=20)
    lines = [ax.plot([], [], [], 'k-')[0] for _ in range(len(connections))]
    title = ax.set_title('')
    
    # 动画更新函数
    def update(frame):
        points_3d = points_3d_list[frame]
        
        # 更新散点
        scatter._offsets3d = (points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
        
        # 更新连接线
        for i, (point1_idx, point2_idx) in enumerate(connections):
            if point1_idx < len(points_3d) and point2_idx < len(points_3d):
                point1 = points_3d[point1_idx]
                point2 = points_3d[point2_idx]
                lines[i].set_data([point1[0], point2[0]], [point1[1], point2[1]])
                lines[i].set_3d_properties([point1[2], point2[2]])
        
        # 更新标题
        title.set_text(f'Frame: {frame+1}/{len(points_3d_list)}')
        
        return [scatter] + lines + [title]
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(points_3d_list), 
                          interval=interval, blit=True)
    
    # 添加图例
    ax.legend()
    
    # 保存为MP4文件
    anim.save(output_file, writer='ffmpeg', fps=1000/interval, dpi=100)
    print(f"动画已保存至 {output_file}")
    
    # 显示动画
    plt.close()

def test_calibration():
    """
    測試相機標定功能
    """
    # 相機標定測試代碼
    # 單相機標定
    images_folder = r"C:\Users\godli\Dropbox\camera_8\calibration_image\camera_L"
    K, dist = calibrate_single_camera(images_folder)
    print("單相機標定結果：")
    print("相機內參矩陣：")
    print(K)
    print("畸變係數：")
    print(dist)
    
    # 三相機標定
    images_folders = [
        r"C:\Users\godli\Dropbox\camera_8\calibration_image\camera_L",
        r"C:\Users\godli\Dropbox\camera_8\calibration_image\camera_M",
        r"C:\Users\godli\Dropbox\camera_8\calibration_image\camera_R"
    ]
    
    calib_file = r"C:\Users\godli\Dropbox\camera_8\calibration.npz"
    mtx1, dist1, mtx2, dist2, mtx3, dist3 = calibrate_three_cameras(images_folders, save_file=calib_file)
    print("三相機標定已保存到：", calib_file)

def main():
    """
    主函數，程序的入口點
    """
    # 設置影片路徑和輸出文件夾
    video_path_left = r"C:\Users\godli\Dropbox\camera_8\video\camera_L_20.mp4"
    video_path_middle = r"C:\Users\godli\Dropbox\camera_8\video\camera_M_20.mp4"
    video_path_right = r"C:\Users\godli\Dropbox\camera_8\video\camera_R_20.mp4"
    video_paths = [video_path_left, video_path_middle, video_path_right]
    
    output_folder = r"C:\Users\godli\Dropbox\camera_8\output_frames"
    calib_file = r"C:\Users\godli\Dropbox\camera_8\calibration.npz"
    
    # 處理影片並獲取3D點
    points_3d_list = process_videos(video_paths, output_folder, calib_file)
    
    # 如果處理成功
    if points_3d_list:
        # 從第一幀獲取相機姿態
        mtx1, dist1, mtx2, dist2, mtx3, dist3 = load_camera_params(calib_file)
        cams_params = (mtx1, dist1, mtx2, dist2, mtx3, dist3)
        
        # 設置ArUco檢測器
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        # 設置ArUco標記板坐標
        board_coord = {}
        board_coord[5] = np.array([
            [0.00, 0.00],
            [58.0, 0.00],
            [58.0, 58.0],
            [0.00, 58.0]
        ], dtype=np.float32)
        
        # 讀取第一幀
        cap_left = cv2.VideoCapture(video_paths[0])
        cap_middle = cv2.VideoCapture(video_paths[1])
        cap_right = cv2.VideoCapture(video_paths[2])
        
        ret_left, frame_left = cap_left.read()
        ret_middle, frame_middle = cap_middle.read()
        ret_right, frame_right = cap_right.read()
        
        if ret_left and ret_middle and ret_right:
            img_L = frame_left.copy()
            img_M = frame_middle.copy()
            img_R = frame_right.copy()
            
            # 獲取相機姿態
            (R_aruco2camL, t_aruco2camL,
             R_aruco2camM, t_aruco2camM,
             R_aruco2camR, t_aruco2camR,
             _, _, _,
             _, _, _,
             _, _, _) = get_aruco_axis(img_L, img_M, img_R, aruco_detector, board_coord, cams_params)
            
            if R_aruco2camL is not None:
                # 可視化3D動畫
                output_file = os.path.join(output_folder, "3d_animation.mp4")
                visualize_3d_animation_three_cameras(
                    points_3d_list, output_file,
                    R_aruco2camL, t_aruco2camL,
                    R_aruco2camM, t_aruco2camM,
                    R_aruco2camR, t_aruco2camR
                )
        
        # 關閉視頻
        cap_left.release()
        cap_middle.release()
        cap_right.release()

if __name__ == "__main__":
    main()

