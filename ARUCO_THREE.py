import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import time
import os
import pandas as pd
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import math
from matplotlib.animation import FuncAnimation
os.environ['PATH'] += ';D:\\software\\ffmpeg-7.1-essentials_build'
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = 'D:\\software\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe'
from collections import defaultdict  
import numpy as np
import cv2
import glob
import os
import sys
import argparse
import openpose.pyopenpose as op

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
    
    print(f"在 {images_folder} 中找到 {len(images)} 張圖片")
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盤格角點
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # 亞像素精確化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            # 繪製角點
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

def calibrate_three_cameras(cam1_folder, cam2_folder, cam3_folder):
    """
    三相機標定主函數
    """
    print("開始相機1標定...")
    ret1, mtx1, dist1, rvecs1, tvecs1 = calibrate_single_camera(cam1_folder)
    
    print("\n開始相機2標定...")
    ret2, mtx2, dist2, rvecs2, tvecs2 = calibrate_single_camera(cam2_folder)
    
    print("\n開始相機3標定...")
    ret3, mtx3, dist3, rvecs3, tvecs3 = calibrate_single_camera(cam3_folder)
    
    # 保存標定結果到當前工作目錄
    save_path = 'three_camera_calibration.npz'
    np.savez(save_path,
             mtx1=mtx1, dist1=dist1,
             mtx2=mtx2, dist2=dist2,
             mtx3=mtx3, dist3=dist3)
    
    print(f"\n標定結果已保存到: {os.path.abspath(save_path)}")
    
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
    
    return save_path

def test_calibration(image_path, camera_params):
    """
    測試標定結果
    
    Args:
        image_path: 測試圖片路徑
        camera_params: 相機參數(mtx, dist)
    """
    mtx, dist = camera_params
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # 獲取新的相機矩陣
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    # 校正圖像
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # 裁剪圖像
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    # 顯示結果
    cv2.imshow('Original', cv2.resize(img, (800,600)))
    cv2.imshow('Calibrated', cv2.resize(dst, (800,600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 設置標定圖片文件夾路徑
    cam1_left = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\intrinsic\port_3left"
    cam2_center = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\intrinsic\port_1_mid"
    cam3_right = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\intrinsic\port_2_right"
    
    # 執行三相機標定並獲取保存路徑
    npz_path = calibrate_three_cameras(cam1_left, cam2_center, cam3_right)
    
    # 驗證npz文件是否成功生成和加載
    try:
        calib_data = np.load(npz_path)
        print("\n成功加載標定文件，包含以下數據:")
        print(calib_data.files)
    except Exception as e:
        print(f"\n加載標定文件時出錯: {e}")

# 1. 加载相机参数
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

# 2. 定义函数以获取 ArUco 标记坐标和姿态估计
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
    axis_coord = axis_coord * 50  # 將 200 改小一些，比如 50 或 100
    
    # 在三个相机图像中检测ArUco标记
    corners_L, ids_L, rejectedImgPoints_L = aruco_detector.detectMarkers(img_L)
    corners_M, ids_M, rejectedImgPoints_M = aruco_detector.detectMarkers(img_M)
    corners_R, ids_R, rejectedImgPoints_R = aruco_detector.detectMarkers(img_R)
    
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
            
            # 转换为整数坐标用于绘制
            tmp_marker_tl = (int(tmp_marker[0][0]), int(tmp_marker[0][1]))
            tmp_marker_tr = (int(tmp_marker[1][0]), int(tmp_marker[1][1]))
            tmp_marker_br = (int(tmp_marker[2][0]), int(tmp_marker[2][1]))
            tmp_marker_bl = (int(tmp_marker[3][0]), int(tmp_marker[3][1]))
            
            # 在图像上标记角点
            cv2.circle(img_L, tmp_marker_tl, 10, (0, 0, 255), -1)
            cv2.circle(img_L, tmp_marker_tr, 10, (0, 255, 0), -1)
            cv2.circle(img_L, tmp_marker_br, 10, (255, 0, 0), -1)
            cv2.circle(img_L, tmp_marker_bl, 10, (0, 170, 255), -1)
            cv2.putText(img_L, f"ID: {ids_L[i][0]}", (int(tmp_marker_tl[0] + 10), int(tmp_marker_tl[1] + 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            
            # 构建角点坐标数组
            img_coord = np.array([tmp_marker[j] for j in range(4)])
            img_coords_L.append(np.squeeze(img_coord))
            
            # 创建3D坐标
            tem_coord = np.hstack((board_coord[ids_L[i][0]], np.zeros(len(board_coord[ids_L[i][0]]))[:,None]))
            point_coords_L.append(tem_coord)
            
            # 解决PnP问题
            image_C_L = np.ascontiguousarray(img_coord[:,:2]).reshape((-1,1,2))
            ret_L, rvec_L, tvec_L = cv2.solvePnP(tem_coord, image_C_L, mtx1, dist1)
            R_aruco2camL, _ = cv2.Rodrigues(rvec_L)
            t_aruco2camL = tvec_L
            
            # 计算ArUco点在相机坐标系中的位置
            aruco_p_L_2camL = np.dot(R_aruco2camL, tem_coord.T).T + t_aruco2camL.T
            aruco_ps_L_2camL.append(aruco_p_L_2camL)

    # 处理中间相机图像
    if ids_M is not None:
        for i in range(len(ids_M)):
            if ids_M[i][0] not in board_coord.keys():
                continue
            tmp_marker = corners_M[i][0]
            
            # 转换为整数坐标用于绘制
            tmp_marker_tl = (int(tmp_marker[0][0]), int(tmp_marker[0][1]))
            tmp_marker_tr = (int(tmp_marker[1][0]), int(tmp_marker[1][1]))
            tmp_marker_br = (int(tmp_marker[2][0]), int(tmp_marker[2][1]))
            tmp_marker_bl = (int(tmp_marker[3][0]), int(tmp_marker[3][1]))
            
            # 在图像上标记角点
            cv2.circle(img_M, tmp_marker_tl, 10, (0, 0, 255), -1)
            cv2.circle(img_M, tmp_marker_tr, 10, (0, 255, 0), -1)
            cv2.circle(img_M, tmp_marker_br, 10, (255, 0, 0), -1)
            cv2.circle(img_M, tmp_marker_bl, 10, (0, 170, 255), -1)
            cv2.putText(img_M, f"ID: {ids_M[i][0]}", (int(tmp_marker_tl[0] + 10), int(tmp_marker_tl[1] + 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            
            # 构建角点坐标数组
            img_coord = np.array([tmp_marker[j] for j in range(4)])
            img_coords_M.append(np.squeeze(img_coord))
            
            # 创建3D坐标
            tem_coord = np.hstack((board_coord[ids_M[i][0]], np.zeros(len(board_coord[ids_M[i][0]]))[:,None]))
            point_coords_M.append(tem_coord)
            
            # 解决PnP问题
            image_C_M = np.ascontiguousarray(img_coord[:,:2]).reshape((-1,1,2))
            ret_M, rvec_M, tvec_M = cv2.solvePnP(tem_coord, image_C_M, mtx2, dist2)
            R_aruco2camM, _ = cv2.Rodrigues(rvec_M)
            t_aruco2camM = tvec_M
            
            # 计算ArUco点在相机坐标系中的位置
            aruco_p_M_2camM = np.dot(R_aruco2camM, tem_coord.T).T + t_aruco2camM.T
            aruco_ps_M_2camM.append(aruco_p_M_2camM)

    # 处理右相机图像
    if ids_R is not None:
        for i in range(len(ids_R)):
            if ids_R[i][0] not in board_coord.keys():
                continue
            tmp_marker = corners_R[i][0]
            
            # 转换为整数坐标用于绘制
            tmp_marker_tl = (int(tmp_marker[0][0]), int(tmp_marker[0][1]))
            tmp_marker_tr = (int(tmp_marker[1][0]), int(tmp_marker[1][1]))
            tmp_marker_br = (int(tmp_marker[2][0]), int(tmp_marker[2][1]))
            tmp_marker_bl = (int(tmp_marker[3][0]), int(tmp_marker[3][1]))
            
            # 在图像上标记角点
            cv2.circle(img_R, tmp_marker_tl, 10, (0, 0, 255), -1)
            cv2.circle(img_R, tmp_marker_tr, 10, (0, 255, 0), -1)
            cv2.circle(img_R, tmp_marker_br, 10, (255, 0, 0), -1)
            cv2.circle(img_R, tmp_marker_bl, 10, (0, 170, 255), -1)
            cv2.putText(img_R, f"ID: {ids_R[i][0]}", (int(tmp_marker_tl[0] + 10), int(tmp_marker_tl[1] + 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            
            # 构建角点坐标数组
            img_coord = np.array([tmp_marker[j] for j in range(4)])
            img_coords_R.append(np.squeeze(img_coord))
            
            # 创建3D坐标
            tem_coord = np.hstack((board_coord[ids_R[i][0]], np.zeros(len(board_coord[ids_R[i][0]]))[:,None]))
            point_coords_R.append(tem_coord)
            
            # 解决PnP问题
            image_C_R = np.ascontiguousarray(img_coord[:,:2]).reshape((-1,1,2))
            ret_R, rvec_R, tvec_R = cv2.solvePnP(tem_coord, image_C_R, mtx3, dist3)
            R_aruco2camR, _ = cv2.Rodrigues(rvec_R)
            t_aruco2camR = tvec_R
            
            # 计算ArUco点在相机坐标系中的位置
            aruco_p_R_2camR = np.dot(R_aruco2camR, tem_coord.T).T + t_aruco2camR.T
            aruco_ps_R_2camR.append(aruco_p_R_2camR)

    # 合并检测到的点
    img_coords_L = np.concatenate(img_coords_L, axis=0) if img_coords_L else np.array([])
    img_coords_M = np.concatenate(img_coords_M, axis=0) if img_coords_M else np.array([])
    img_coords_R = np.concatenate(img_coords_R, axis=0) if img_coords_R else np.array([])
    
    # 添加齐次坐标
    img_coords_L = np.hstack((img_coords_L, np.ones((len(img_coords_L), 1)))) if len(img_coords_L) > 0 else np.array([])
    img_coords_M = np.hstack((img_coords_M, np.ones((len(img_coords_M), 1)))) if len(img_coords_M) > 0 else np.array([])
    img_coords_R = np.hstack((img_coords_R, np.ones((len(img_coords_R), 1)))) if len(img_coords_R) > 0 else np.array([])
    
    # 合并3D点坐标
    point_coords_L = np.concatenate(point_coords_L, axis=0) if point_coords_L else np.array([])
    point_coords_M = np.concatenate(point_coords_M, axis=0) if point_coords_M else np.array([])
    point_coords_R = np.concatenate(point_coords_R, axis=0) if point_coords_R else np.array([])
    
    # 合并相机坐标系下的点
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
                if np.linalg.norm(point-cluster_point) <= board_coord[5][3,1]:
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

    # 排序并选择最大聚类的点
    cluster_max_indxs_L.sort()
    cluster_max_indxs_M.sort()
    cluster_max_indxs_R.sort()

    img_coords_L = img_coords_L[cluster_max_indxs_L]
    img_coords_M = img_coords_M[cluster_max_indxs_M]
    img_coords_R = img_coords_R[cluster_max_indxs_R]

    point_coords_L = point_coords_L[cluster_max_indxs_L]
    point_coords_M = point_coords_M[cluster_max_indxs_M]
    point_coords_R = point_coords_R[cluster_max_indxs_R]

    # 解决最终的PnP问题
    # 左相机
    image_C_L = np.ascontiguousarray(img_coords_L[:,:2]).reshape((-1,1,2))
    ret_L, rvec_L, tvec_L = cv2.solvePnP(point_coords_L, image_C_L, mtx1, dist1)
    
    # 中间相机
    image_C_M = np.ascontiguousarray(img_coords_M[:,:2]).reshape((-1,1,2))
    ret_M, rvec_M, tvec_M = cv2.solvePnP(point_coords_M, image_C_M, mtx2, dist2)
    
    # 右相机
    image_C_R = np.ascontiguousarray(img_coords_R[:,:2]).reshape((-1,1,2))
    ret_R, rvec_R, tvec_R = cv2.solvePnP(point_coords_R, image_C_R, mtx3, dist3)

    # 绘制坐标轴
    # 左相机
    image_points, _ = cv2.projectPoints(axis_coord, rvec_L, tvec_L, mtx1, dist1)
    image_points = image_points.reshape(-1, 2).astype(np.int16)
    cv2.line(img_L, (image_points[0,0], image_points[0,1]), (image_points[1,0], image_points[1,1]), (0,0,255), 5)
    cv2.line(img_L, (image_points[0,0], image_points[0,1]), (image_points[2,0], image_points[2,1]), (0,255,0), 5)
    cv2.line(img_L, (image_points[0,0], image_points[0,1]), (image_points[3,0], image_points[3,1]), (255,0,0), 5)

    # 中间相机
    image_points, _ = cv2.projectPoints(axis_coord, rvec_M, tvec_M, mtx2, dist2)
    image_points = image_points.reshape(-1, 2).astype(np.int16)
    cv2.line(img_M, (image_points[0,0], image_points[0,1]), (image_points[1,0], image_points[1,1]), (0,0,255), 5)
    cv2.line(img_M, (image_points[0,0], image_points[0,1]), (image_points[2,0], image_points[2,1]), (0,255,0), 5)
    cv2.line(img_M, (image_points[0,0], image_points[0,1]), (image_points[3,0], image_points[3,1]), (255,0,0), 5)

    # 右相机
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

    # 计算相机到ArUco的变换
    R_camL2aruco = R_aruco2camL.T
    t_camL2aruco = -R_aruco2camL.T @ t_aruco2camL
    R_camM2aruco = R_aruco2camM.T
    t_camM2aruco = -R_aruco2camM.T @ t_aruco2camM
    R_camR2aruco = R_aruco2camR.T
    t_camR2aruco = -R_aruco2camR.T @ t_aruco2camR

    return (R_aruco2camL, t_aruco2camL,
            R_aruco2camM, t_aruco2camM,
            R_aruco2camR, t_aruco2camR,
            R_camL2aruco, t_camL2aruco,
            R_camM2aruco, t_camM2aruco,
            R_camR2aruco, t_camR2aruco,
            img_L, img_M, img_R)

# 3. 定义处理单帧的函数
def process_frame(opWrapper, frame_left, frame_middle, frame_right, img_L, img_M, img_R, cams_params, cam_P):
    """
    處理三個相機的單幀圖像（使用OpenPose）
    Args:
        opWrapper: OpenPose包裝器
        frame_left/middle/right: 三個相機的原始幀
        img_L/M/R: 三個相機的標註圖像
        cams_params: 相機參數(mtx1, dist1, mtx2, dist2, mtx3, dist3)
        cam_P: 相機投影矩陣參數
    Returns:
        points_3d: 三維重建點
        img_L/M/R: 更新後的標註圖像
    """
    # 檢測每個相機視角的關鍵點
    poses = []
    frames = [frame_left, frame_middle, frame_right]
    cam_names = ["左側", "中間", "右側"]
    images = [img_L, img_M, img_R]
    
    # COCO模型的骨架連接
    POSE_PAIRS = [
        [1,0], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10],
        [1,11], [11,12], [12,13], [0,14], [0,15], [14,16], [15,17]
    ]
    
    def draw_skeleton(image, keypoints):
        for pair in POSE_PAIRS:
            if keypoints[pair[0]][2] > 0 and keypoints[pair[1]][2] > 0:
                pt1 = (int(keypoints[pair[0]][0]), int(keypoints[pair[0]][1]))
                pt2 = (int(keypoints[pair[1]][0]), int(keypoints[pair[1]][1]))
                cv2.line(image, pt1, pt2, (0, 255, 255), 2)
    
    detection_failed = False
    failed_cameras = []
    
    # 處理每個相機的圖像
    for i, frame in enumerate(frames):
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        
        if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
            keypoints = datum.poseKeypoints[0]
            pose_2d = keypoints[:18, :2]
            poses.append(pose_2d)
            
            # 繪製關鍵點和骨架
            for j in range(len(keypoints)):
                if keypoints[j][2] > 0:
                    x, y = int(keypoints[j][0]), int(keypoints[j][1])
                    cv2.circle(images[i], (x, y), 5, (0, 255, 0), -1)
            draw_skeleton(images[i], keypoints)
        else:
            # 如果檢測失敗，添加一個空的姿態（全零）
            poses.append(np.zeros((18, 2)))
            detection_failed = True
            failed_cameras.append(cam_names[i])
    
    # 檢查是否所有相機都檢測到姿態
    if detection_failed:
        print(f"警告: 以下相機未檢測到人體姿勢: {', '.join(failed_cameras)}")
        return np.zeros((18, 3)), img_L, img_M, img_R
    
    # 解包相機參數和投影矩陣
    (mtx1, dist1, mtx2, dist2, mtx3, dist3) = cams_params
    (R_aruco2camL, t_aruco2camL, R_aruco2camM, t_aruco2camM, R_aruco2camR, t_aruco2camR) = cam_P
    
    # 構建投影矩陣
    P1 = np.hstack((R_aruco2camL, t_aruco2camL))
    P2 = np.hstack((R_aruco2camM, t_aruco2camM))
    P3 = np.hstack((R_aruco2camR, t_aruco2camR))
    
    # 進行三維重建
    points_3d = triangulate_points_three_cameras(
        poses[0], poses[1], poses[2],
        mtx1, dist1, mtx2, dist2, mtx3, dist3,
        P1, P2, P3
    )
    
    return points_3d, img_L, img_M, img_R

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
        points_3d: 三維點雲
    """
    points_3d = []
    
    # 對每個關鍵點進行三角測量
    for pt_left, pt_middle, pt_right in zip(points_left, points_middle, points_right):
        # 去畸變
        pt_left_undist = cv2.undistortPoints(pt_left.reshape(1, 1, 2), mtx1, dist1)
        pt_middle_undist = cv2.undistortPoints(pt_middle.reshape(1, 1, 2), mtx2, dist2)
        pt_right_undist = cv2.undistortPoints(pt_right.reshape(1, 1, 2), mtx3, dist3)
        
        # 使用DLT方法進行三角測量
        A = np.zeros((6, 4))
        
        # 左相機約束
        A[0] = pt_left_undist[0, 0, 0] * P1[2] - P1[0]
        A[1] = pt_left_undist[0, 0, 1] * P1[2] - P1[1]
        
        # 中間相機約束
        A[2] = pt_middle_undist[0, 0, 0] * P2[2] - P2[0]
        A[3] = pt_middle_undist[0, 0, 1] * P2[2] - P2[1]
        
        # 右相機約束
        A[4] = pt_right_undist[0, 0, 0] * P3[2] - P3[0]
        A[5] = pt_right_undist[0, 0, 1] * P3[2] - P3[1]
        
        # 求解最小二乘問題
        _, _, Vt = np.linalg.svd(A)
        point_4d = Vt[-1]
        point_3d = (point_4d / point_4d[3])[:3]
        
        points_3d.append(point_3d)
    
    return np.array(points_3d)


    
# 5. 定义处理视频的主循环
def process_videos(video_path_left, video_path_center, video_path_right, start_frame=0):
    """
    處理三個相機的視頻
    Args:
        video_path_left: 左相機視頻路徑
        video_path_center: 中間相機視頻路徑
        video_path_right: 右相機視頻路徑
        start_frame: 起始幀（默認為0）
    Returns:
        all_points_3d: 所有幀的3D骨骼點
        all_bone_points: 所有幀的骨頭點
        aruco_axis: ArUco坐標系
        camL_axis: 左相機坐標系
        camM_axis: 中間相機坐標系
        camR_axis: 右相機坐標系
    """
    ## 0. 導入相機內參
    camera_params_path = r"C:\Users\user\Desktop\wenfeng_caliscope\three_camera_calibration.npz"
    if not os.path.exists(camera_params_path):
        print(f'相机参数文件不存在: {camera_params_path}')
        print('请先运行相机标定程序生成参数文件')
        return
    # 載入三個相機的參數
    mtx1, dist1, mtx2, dist2, mtx3, dist3 = load_camera_params(camera_params_path)
    cams_params = (mtx1, dist1, mtx2, dist2, mtx3, dist3)

    ## 1. 設置aruco參數
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # 定義 ArUco 標記板的坐標
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

    ## 2. 設置OpenPose
    params = {
        "model_folder": "./openpose/models/",
        "model_pose": "COCO",  # 明確指定使用COCO模型
        "net_resolution": "-1x368",
        "number_people_max": 1,
        "disable_multi_thread": False,
    }
    
    try:
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
    except Exception as e:
        print(f"錯誤: OpenPose 初始化失敗 - {str(e)}")
        return

    ## 3. 讀取視頻
    print(f"開始處理視頻從幀 {start_frame} 開始...")
    cap_left = cv2.VideoCapture(video_path_left)
    cap_center = cv2.VideoCapture(video_path_center)
    cap_right = cv2.VideoCapture(video_path_right)
    
    if not cap_left.isOpened() or not cap_center.isOpened() or not cap_right.isOpened():
        raise ValueError("無法打開視頻文件")
        
    # 跳過已經處理的幀
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_center.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 收集每幀的點
    all_points_3d = []  # 3d骨骼點 (18點)
    aruco_axis = []    # aruco坐標系
    camL_axis = []     # 左相機坐標系
    camM_axis = []     # 中間相機坐標系
    camR_axis = []     # 右相機坐標系
    frame_count = start_frame
    
    # 坐標系的4個點
    axis_coord = np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ],dtype=np.float32)
    axis_coord = axis_coord * 50  # 將 200 改小一些，比如 50 或 100
    
    # 開始循環讀取每一幀視頻圖像
    while True:
        ret_left, frame_left = cap_left.read()
        ret_center, frame_center = cap_center.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_center or not ret_right:
            break
            
        print(f"處理第 {frame_count + 1} 幀")
            
        # 處理ArUco標記
        result = get_aruco_axis(frame_left, frame_center, frame_right, aruco_detector, board_coord, cams_params)
        if result[0] is None:
            print(f"第 {frame_count + 1} 幀未檢測到 ArUco 標記")
            all_points_3d.append(np.zeros((18, 3)))  # 修改為18點，與COCO模型一致
            aruco_axis.append(np.zeros((4,3)))
            camL_axis.append(np.zeros((4,3)))
            camM_axis.append(np.zeros((4,3)))
            camR_axis.append(np.zeros((4,3)))
            frame_count += 1
            continue
            
        # 解包結果
        (R_aruco2camL, t_aruco2camL, 
         R_aruco2camM, t_aruco2camM,
         R_aruco2camR, t_aruco2camR,
         R_camL2aruco, t_camL2aruco,
         R_camM2aruco, t_camM2aruco,
         R_camR2aruco, t_camR2aruco,
         img_L, img_M, img_R) = result
         
        # 更新坐標系
        aruco_axis.append(axis_coord)
        camL_axis.append(np.dot(R_camL2aruco, (axis_coord).T).T + t_camL2aruco.T)
        camM_axis.append(np.dot(R_camM2aruco, (axis_coord).T).T + t_camM2aruco.T)
        camR_axis.append(np.dot(R_camR2aruco, (axis_coord).T).T + t_camR2aruco.T)

        # 處理姿態估計
        cam_P = (R_aruco2camL, t_aruco2camL, R_aruco2camM, t_aruco2camM, R_aruco2camR, t_aruco2camR)
        points_3d, img_L, img_M, img_R = process_frame(opWrapper, frame_left, frame_center, frame_right, 
                                                   img_L, img_M, img_R, cams_params, cam_P)
        
        if points_3d is not None:
            # 確保只保存18個點
            all_points_3d.append(points_3d[:18])
        else:
            # 如果檢測失敗，添加18個零點
            all_points_3d.append(np.zeros((18, 3)))

        # 顯示圖像
        combined_img = np.hstack((
            cv2.resize(img_L, (426, 320)),
            cv2.resize(img_M, (426, 320)),
            cv2.resize(img_R, (426, 320))
        ))
        cv2.imshow('Three Camera Views', combined_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"已處理 {frame_count} 幀")

    # 釋放資源
    cap_left.release()
    cap_center.release()
    cap_right.release()
    cv2.destroyAllWindows()

    print(f"視頻處理完成。共處理了 {frame_count - start_frame} 幀")
    return (np.array(all_points_3d),
            np.array(aruco_axis),
            np.array(camL_axis),
            np.array(camM_axis),
            np.array(camR_axis))

def visualize_3d_animation_three_cameras(points, aruco_axis, camL_axis, camM_axis, camR_axis, title='3D Visualization'):
    """
    三相機系統的3D點雲動畫可視化，添加骨架連接
    Args:
        points: 3D關鍵點序列 [frames, 18, 3] (OpenPose格式)
        aruco_axis: ArUco坐標系 [frames, 4, 3]
        camL_axis: 左相機坐標系 [frames, 4, 3]
        camM_axis: 中間相機坐標系 [frames, 4, 3]
        camR_axis: 右相機坐標系 [frames, 4, 3]
        title: 視窗標題
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 計算所有點的範圍
    all_points = np.vstack((
        points.reshape(-1, 3),
        aruco_axis.reshape(-1, 3),
        camL_axis.reshape(-1, 3),
        camM_axis.reshape(-1, 3),
        camR_axis.reshape(-1, 3)
    ))
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    range_vals = max_vals - min_vals
    
    # 設置坐標軸範圍，添加邊距
    margin = 0.1 * range_vals
    ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
    ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
    ax.set_zlim(min_vals[2] - margin[2], max_vals[2] + margin[2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 設置視角
    ax.view_init(elev=10, azim=-60)
    
    # 添加地板
    floor_y = min_vals[1]
    x_floor = np.array([min_vals[0] - margin[0], max_vals[0] + margin[0]])
    z_floor = np.array([min_vals[2] - margin[2], max_vals[2] + margin[2]])
    X_floor, Z_floor = np.meshgrid(x_floor, z_floor)
    Y_floor = np.full(X_floor.shape, floor_y)
    ax.plot_surface(X_floor, Y_floor, Z_floor, alpha=0.2, color='gray')
    
    # 初始化散點圖和骨架線條
    scatter = ax.scatter([], [], [], s=20, c='r', alpha=0.6)
    
    # 修正 OpenPose COCO 18點模型的骨架連接
    # COCO 模型的18個關鍵點索引：
    # 0: Nose 鼻子
    # 1: Neck 脖子
    # 2: Right Shoulder 右肩
    # 3: Right Elbow 右肘
    # 4: Right Wrist 右手腕
    # 5: Left Shoulder 左肩
    # 6: Left Elbow 左肘
    # 7: Left Wrist 左手腕
    # 8: Right Hip 右髖
    # 9: Right Knee 右膝
    # 10: Right Ankle 右踝
    # 11: Left Hip 左髖
    # 12: Left Knee 左膝
    # 13: Left Ankle 左踝
    # 14: Right Eye 右眼
    # 15: Left Eye 左眼
    # 16: Right Ear 右耳
    # 17: Left Ear 左耳

    # 定義骨架連接
    connections = [
        # 軀幹
        (1, 0),    # 脖子到鼻子
        (1, 2), (2, 3), (3, 4),      # 右臂
        (1, 5), (5, 6), (6, 7),      # 左臂
        (1, 8), (8, 9), (9, 10),     # 右腿
        (1, 11), (11, 12), (12, 13), # 左腿
        (0, 14), (0, 15),            # 鼻子到眼睛
        (14, 16), (15, 17)           # 眼睛到耳朵
    ]
    
    # 為不同部位設置顏色
    colors = {
        'head': 'purple',    # 頭部（包括臉部特徵）
        'spine': 'blue',     # 脊椎（頸部到髖部）
        'right_limb': 'red', # 右側肢體
        'left_limb': 'green' # 左側肢體
    }
    
    # 定義每個連接的顏色
    connection_colors = []
    for start, end in connections:
        if start == 0 or end == 0 or start in [14,15,16,17] or end in [14,15,16,17]:  # 頭部
            connection_colors.append(colors['head'])
        elif start == 1:  # 脊椎
            connection_colors.append(colors['spine'])
        elif start in [2,3,4,8,9,10] or end in [2,3,4,8,9,10]:  # 右側肢體
            connection_colors.append(colors['right_limb'])
        else:  # 左側肢體
            connection_colors.append(colors['left_limb'])
    
    # 創建線條
    lines = []
    for color in connection_colors:
        line, = ax.plot([], [], [], color=color, alpha=0.8, linewidth=2)
        lines.append(line)
    
    # 添加坐標系線條
    coord_lines = []
    for _ in range(12):  # 4個坐標系 × 3個軸
        line, = ax.plot([], [], [], '-', lw=2, alpha=0.7)
        coord_lines.append(line)
    
    def update(frame):
        # 更新骨骼點
        point_cloud = points[frame]
        scatter._offsets3d = (point_cloud[:,0], point_cloud[:,1], point_cloud[:,2])
        
        # 更新骨架線條
        for i, ((start, end), line) in enumerate(zip(connections, lines)):
            if np.all(point_cloud[start] != 0) and np.all(point_cloud[end] != 0):
                line.set_data_3d([point_cloud[start,0], point_cloud[end,0]],
                               [point_cloud[start,1], point_cloud[end,1]],
                               [point_cloud[start,2], point_cloud[end,2]])
            else:
                line.set_data_3d([], [], [])
        
        # 更新坐標系
        # ArUco坐標系
        for i in range(3):
            coord_lines[i].set_data_3d([aruco_axis[frame,0,0], aruco_axis[frame,i+1,0]],
                                     [aruco_axis[frame,0,1], aruco_axis[frame,i+1,1]],
                                     [aruco_axis[frame,0,2], aruco_axis[frame,i+1,2]])
            coord_lines[i].set_color(['r','g','b'][i])
        
        # 左相機坐標系
        for i in range(3):
            coord_lines[i+3].set_data_3d([camL_axis[frame,0,0], camL_axis[frame,i+1,0]],
                                       [camL_axis[frame,0,1], camL_axis[frame,i+1,1]],
                                       [camL_axis[frame,0,2], camL_axis[frame,i+1,2]])
            coord_lines[i+3].set_color(['r','g','b'][i])
        
        # 中間相機坐標系
        for i in range(3):
            coord_lines[i+6].set_data_3d([camM_axis[frame,0,0], camM_axis[frame,i+1,0]],
                                       [camM_axis[frame,0,1], camM_axis[frame,i+1,1]],
                                       [camM_axis[frame,0,2], camM_axis[frame,i+1,2]])
            coord_lines[i+6].set_color(['r','g','b'][i])
        
        # 右相機坐標系
        for i in range(3):
            coord_lines[i+9].set_data_3d([camR_axis[frame,0,0], camR_axis[frame,i+1,0]],
                                       [camR_axis[frame,0,1], camR_axis[frame,i+1,1]],
                                       [camR_axis[frame,0,2], camR_axis[frame,i+1,2]])
            coord_lines[i+9].set_color(['r','g','b'][i])
        
        # 更新標題顯示當前幀
        ax.set_title(f'{title} - Frame: {frame}')
        
        return [scatter] + lines + coord_lines
    
    # 創建動畫
    anim = FuncAnimation(
        fig,
        update,
        frames=len(points),
        interval=50,
        blit=False,
        repeat=True
    )
    
    plt.show()
    return anim

# 13. 主程
def main():
    # 确保视频路径正确
    video_path_left = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\recordings\port_3\port_3-01082025180809-0000.avi"
    video_path_center = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\recordings\port_1\port_1-01082025180808-0000.avi"
    video_path_right = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\recordings\port_2\port_2-01082025180810-0000.avi"

    # 处理所有帧
    all_points_3d_original, aruco_axises, camL_axises, camM_axises, camR_axises = process_videos(
        video_path_left, video_path_center, video_path_right)

    if len(all_points_3d_original) == 0:
        print("未能从视频中提取任何3D关键点。")
        return

    # 可視化3D點雲
    print("開始3D可視化...")
    visualize_3d_animation_three_cameras(
        all_points_3d_original,
        aruco_axises,
        camL_axises,
        camM_axises,
        camR_axises,
        title='Three Camera Motion Capture'
    )



if __name__ == "__main__":
    main()

