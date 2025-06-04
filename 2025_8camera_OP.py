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
os.environ['PATH'] += ';D:\\software\\ffmpeg-7.1-essentials_build'
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = 'D:\\software\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe'
from collections import defaultdict  
import numpy as np
import cv2
import glob
import os

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
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
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

def calibrate_eight_cameras(left_1_folder, left_2_folder, left_3_folder, 
                          center_folder, 
                          right_1_folder, right_2_folder, right_3_folder, right_4_folder):
    """
    八相機標定主函數
    """
    print("開始左側相機1標定...")
    ret_l1, mtx_l1, dist_l1, rvecs_l1, tvecs_l1 = calibrate_single_camera(left_1_folder)
    
    print("\n開始左側相機2標定...")
    ret_l2, mtx_l2, dist_l2, rvecs_l2, tvecs_l2 = calibrate_single_camera(left_2_folder)
    
    print("\n開始左側相機3標定...")
    ret_l3, mtx_l3, dist_l3, rvecs_l3, tvecs_l3 = calibrate_single_camera(left_3_folder)
    
    print("\n開始中心相機標定...")
    ret_c, mtx_c, dist_c, rvecs_c, tvecs_c = calibrate_single_camera(center_folder)
    
    print("\n開始右側相機1標定...")
    ret_r1, mtx_r1, dist_r1, rvecs_r1, tvecs_r1 = calibrate_single_camera(right_1_folder)
    
    print("\n開始右側相機2標定...")
    ret_r2, mtx_r2, dist_r2, rvecs_r2, tvecs_r2 = calibrate_single_camera(right_2_folder)
    
    print("\n開始右側相機3標定...")
    ret_r3, mtx_r3, dist_r3, rvecs_r3, tvecs_r3 = calibrate_single_camera(right_3_folder)
    
    print("\n開始右側相機4標定...")
    ret_r4, mtx_r4, dist_r4, rvecs_r4, tvecs_r4 = calibrate_single_camera(right_4_folder)
    
    # 保存標定結果到當前工作目錄
    save_path = 'eight_camera_calibration.npz'
    np.savez(save_path,
             mtx_l1=mtx_l1, dist_l1=dist_l1,
             mtx_l2=mtx_l2, dist_l2=dist_l2,
             mtx_l3=mtx_l3, dist_l3=dist_l3,
             mtx_c=mtx_c, dist_c=dist_c,
             mtx_r1=mtx_r1, dist_r1=dist_r1,
             mtx_r2=mtx_r2, dist_r2=dist_r2,
             mtx_r3=mtx_r3, dist_r3=dist_r3,
             mtx_r4=mtx_r4, dist_r4=dist_r4)
    
    print(f"\n標定結果已保存到: {os.path.abspath(save_path)}")
    
    # 打印所有相機內參
    cameras = {
        "左側相機1": (mtx_l1, dist_l1),
        "左側相機2": (mtx_l2, dist_l2),
        "左側相機3": (mtx_l3, dist_l3),
        "中心相機": (mtx_c, dist_c),
        "右側相機1": (mtx_r1, dist_r1),
        "右側相機2": (mtx_r2, dist_r2),
        "右側相機3": (mtx_r3, dist_r3),
        "右側相機4": (mtx_r4, dist_r4)
    }
    
    for name, (mtx, dist) in cameras.items():
        print(f"\n{name}內參矩陣:")
        print(mtx)
        print(f"{name}畸變係數:")
        print(dist)
    
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
    
    while False:
        # 設置標定圖片文件夾路徑
        left_1_folder = r".\20250116\intrinsic\685-left1"
        left_2_folder = r".\20250116\intrinsic\684-left2"
        left_3_folder = r".\20250116\intrinsic\688-left3"
        center_folder = r".\20250116\intrinsic\1034-center"
        right_1_folder = r".\20250116\intrinsic\686-right1"
        right_2_folder = r".\20250116\intrinsic\725-right2"
        right_3_folder = r".\20250116\intrinsic\724-right3"
        right_4_folder = r".\20250116\intrinsic\826-right4"
        # 執行八相機標定並獲取保存路徑
        npz_path = calibrate_eight_cameras(
            left_1_folder, left_2_folder, left_3_folder,
            center_folder,
            right_1_folder, right_2_folder, right_3_folder, right_4_folder
        )
        
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
    從文件加載八相機參數
    Args:
        file_path: 相機參數文件路徑
    Returns:
        mtx_l1: 左側相機1內參矩陣
        dist_l1: 左側相機1畸變係數
        mtx_l2: 左側相機2內參矩陣
        dist_l2: 左側相機2畸變係數
        mtx_l3: 左側相機3內參矩陣
        dist_l3: 左側相機3畸變係數
        mtx_c: 中心相機內參矩陣
        dist_c: 中心相機畸變係數
        mtx_r1: 右側相機1內參矩陣
        dist_r1: 右側相機1畸變係數
        mtx_r2: 右側相機2內參矩陣
        dist_r2: 右側相機2畸變係數
        mtx_r3: 右側相機3內參矩陣
        dist_r3: 右側相機3畸變係數
        mtx_r4: 右側相機4內參矩陣
        dist_r4: 右側相機4畸變係數
    """
    data = np.load(file_path)
    return (data['mtx_l1'], data['dist_l1'],
            data['mtx_l2'], data['dist_l2'],
            data['mtx_l3'], data['dist_l3'],
            data['mtx_c'], data['dist_c'],
            data['mtx_r1'], data['dist_r1'],
            data['mtx_r2'], data['dist_r2'],
            data['mtx_r3'], data['dist_r3'],
            data['mtx_r4'], data['dist_r4'])

def get_aruco_axis(img_l1, img_l2, img_l3, img_c, img_r1, img_r2, img_r3, img_r4, 
                   aruco_detector, board_coord, cams_params):
    """
    從八個相機圖像中檢測ArUco標記並估計其姿態
    Args:
        img_l1/l2/l3: 左側相機1/2/3圖像
        img_c: 中心相機圖像
        img_r1/r2/r3/r4: 右側相機1/2/3/4圖像
        aruco_detector: ArUco檢測器
        board_coord: ArUco標記板坐標字典
        cams_params: 相機參數元組(mtx_l1, dist_l1, ..., mtx_r4, dist_r4)
    Returns:
        各個相機的旋轉矩陣和平移向量，以及標註後的圖像
    """
    # 解包相機參數
    (mtx_l1, dist_l1, mtx_l2, dist_l2, mtx_l3, dist_l3,
     mtx_c, dist_c, mtx_r1, dist_r1, mtx_r2, dist_r2,
     mtx_r3, dist_r3, mtx_r4, dist_r4) = cams_params
    
    # 定義坐標軸點
    axis_coord = np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ],dtype=np.float32)
    axis_coord = axis_coord * 400  # 放大坐標軸顯示
    
    # 在八個相機圖像中檢測ArUco標記
    corners_l1, ids_l1, _ = aruco_detector.detectMarkers(img_l1)
    corners_l2, ids_l2, _ = aruco_detector.detectMarkers(img_l2)
    corners_l3, ids_l3, _ = aruco_detector.detectMarkers(img_l3)
    corners_c, ids_c, _ = aruco_detector.detectMarkers(img_c)
    corners_r1, ids_r1, _ = aruco_detector.detectMarkers(img_r1)
    corners_r2, ids_r2, _ = aruco_detector.detectMarkers(img_r2)
    corners_r3, ids_r3, _ = aruco_detector.detectMarkers(img_r3)
    corners_r4, ids_r4, _ = aruco_detector.detectMarkers(img_r4)
    
    # 初始化存儲列表
    cameras = {
        'l1': {'img': img_l1, 'corners': corners_l1, 'ids': ids_l1, 'mtx': mtx_l1, 'dist': dist_l1},
        'l2': {'img': img_l2, 'corners': corners_l2, 'ids': ids_l2, 'mtx': mtx_l2, 'dist': dist_l2},
        'l3': {'img': img_l3, 'corners': corners_l3, 'ids': ids_l3, 'mtx': mtx_l3, 'dist': dist_l3},
        'c': {'img': img_c, 'corners': corners_c, 'ids': ids_c, 'mtx': mtx_c, 'dist': dist_c},
        'r1': {'img': img_r1, 'corners': corners_r1, 'ids': ids_r1, 'mtx': mtx_r1, 'dist': dist_r1},
        'r2': {'img': img_r2, 'corners': corners_r2, 'ids': ids_r2, 'mtx': mtx_r2, 'dist': dist_r2},
        'r3': {'img': img_r3, 'corners': corners_r3, 'ids': ids_r3, 'mtx': mtx_r3, 'dist': dist_r3},
        'r4': {'img': img_r4, 'corners': corners_r4, 'ids': ids_r4, 'mtx': mtx_r4, 'dist': dist_r4}
    }
    
    # 在檢測 ArUco 標記後添加可視化代碼
    for cam_id, cam_data in cameras.items():
        if cam_data['ids'] is not None:
            for i in range(len(cam_data['ids'])):
                if cam_data['ids'][i][0] not in board_coord.keys():
                    continue
                    
                tmp_marker = cam_data['corners'][i][0]
                
                # 繪製檢測到的角點
                for j in range(4):
                    cv2.circle(cam_data['img'], 
                             (int(tmp_marker[j][0]), int(tmp_marker[j][1])), 
                             5, (0, 255, 0), -1)  # 綠色圓點標記角點
                
                # 繪製 ID
                center_x = int(np.mean(tmp_marker[:, 0]))
                center_y = int(np.mean(tmp_marker[:, 1]))
                cv2.putText(cam_data['img'], 
                          f"ID:{cam_data['ids'][i][0]}", 
                          (center_x, center_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 0, 255), 2)
                
                # 繪製邊框
                for j in range(4):
                    pt1 = (int(tmp_marker[j][0]), int(tmp_marker[j][1]))
                    pt2 = (int(tmp_marker[(j+1)%4][0]), int(tmp_marker[(j+1)%4][1]))
                    cv2.line(cam_data['img'], pt1, pt2, (255, 0, 0), 2)

        # 顯示檢測結果
        cv2.imshow(f'ArUco Detection - Camera {cam_id}', 
                  cv2.resize(cam_data['img'], (800, 600)))
        cv2.waitKey(1)
    
    results = {}
    for cam_id, cam_data in cameras.items():
        img_coords = []
        point_coords = []
        aruco_ps = []
        
        if cam_data['ids'] is not None:
            for i in range(len(cam_data['ids'])):
                if cam_data['ids'][i][0] not in board_coord.keys():
                    continue
                    
                tmp_marker = cam_data['corners'][i][0]
                
                # 轉換為整數坐標用於繪製
                corners = [(int(tmp_marker[j][0]), int(tmp_marker[j][1])) for j in range(4)]
                colors = [(0,0,255), (0,255,0), (255,0,0), (0,170,255)]
                
                # 在圖像上標記角點
                for corner, color in zip(corners, colors):
                    cv2.circle(cam_data['img'], corner, 10, color, -1)
                
                cv2.putText(cam_data['img'], f"ID: {cam_data['ids'][i][0]}", 
                           (corners[0][0] + 10, corners[0][1] + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                
                # 構建角點坐標數組
                img_coord = np.array([tmp_marker[j] for j in range(4)])
                img_coords.append(np.squeeze(img_coord))
                
                # 創建3D坐標
                tem_coord = np.hstack((board_coord[cam_data['ids'][i][0]], 
                                     np.zeros(len(board_coord[cam_data['ids'][i][0]]))[:,None]))
                point_coords.append(tem_coord)
                
                # 解決PnP問題
                image_C = np.ascontiguousarray(img_coord[:,:2]).reshape((-1,1,2))
                ret, rvec, tvec = cv2.solvePnP(tem_coord, image_C, 
                                             cam_data['mtx'], cam_data['dist'])
                R, _ = cv2.Rodrigues(rvec)
                
                # 計算ArUco點在相機坐標系中的位置
                aruco_p = np.dot(R, tem_coord.T).T + tvec.T
                aruco_ps.append(aruco_p)
        
        # 保存結果
        results[cam_id] = {
            'img_coords': np.array(img_coords) if img_coords else np.array([]),
            'point_coords': np.array(point_coords) if point_coords else np.array([]),
            'aruco_ps': np.array(aruco_ps) if aruco_ps else np.array([])
        }

    # 合併檢測到的點並進行後續處理
    for cam_id in cameras.keys():
        if len(results[cam_id]['img_coords']) > 0:
            # 合併檢測到的點
            results[cam_id]['img_coords'] = np.concatenate(results[cam_id]['img_coords'], axis=0)
            # 添加齊次坐標
            results[cam_id]['img_coords'] = np.hstack((
                results[cam_id]['img_coords'],
                np.ones((len(results[cam_id]['img_coords']), 1)))
            )
            # 合併3D點坐標
            results[cam_id]['point_coords'] = np.concatenate(results[cam_id]['point_coords'], axis=0)
            # 合併相機坐標系下的點
            results[cam_id]['aruco_ps'] = np.concatenate(results[cam_id]['aruco_ps'], axis=0)
        else:
            print(f"{cam_id}相機中未檢測到 ArUco 標記")
            return (None,) * (16 + 8)  # 8個相機的R和t，加上8張圖像
    
    # 對每個相機進行聚類處理
    for cam_id in cameras.keys():
        clusters = defaultdict(list)
        cluster_ids = []
        cluster_indx = {}
        
        for i, point in enumerate(results[cam_id]['aruco_ps']):
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
        
        # 獲取最大聚類的索引
        if cluster_indx:
            cluster_max_indxs = max(cluster_indx.values(), key=len)
            cluster_max_indxs.sort()
            
            # 選擇最大聚類的點
            results[cam_id]['img_coords'] = results[cam_id]['img_coords'][cluster_max_indxs]
            results[cam_id]['point_coords'] = results[cam_id]['point_coords'][cluster_max_indxs]
    
    # 解決最終的PnP問題並繪製坐標軸
    R_results = {}
    t_results = {}
    
    for cam_id, cam_data in cameras.items():
        # 解決PnP問題
        image_points = np.ascontiguousarray(results[cam_id]['img_coords'][:,:2]).reshape((-1,1,2))
        ret, rvec, tvec = cv2.solvePnP(
            results[cam_id]['point_coords'],
            image_points,
            cam_data['mtx'],
            cam_data['dist']
        )
        
        # 計算旋轉矩陣
        R_aruco2cam, _ = cv2.Rodrigues(rvec)
        t_aruco2cam = tvec
        
        # 計算相機到ArUco的變換
        R_cam2aruco = R_aruco2cam.T
        t_cam2aruco = -R_aruco2cam.T @ t_aruco2cam
        
        # 保存結果
        R_results[f'R_aruco2{cam_id}'] = R_aruco2cam
        R_results[f'R_{cam_id}2aruco'] = R_cam2aruco
        t_results[f't_aruco2{cam_id}'] = t_aruco2cam
        t_results[f't_{cam_id}2aruco'] = t_cam2aruco
        
        # 繪製坐標軸
        image_points, _ = cv2.projectPoints(axis_coord, rvec, tvec, 
                                          cam_data['mtx'], cam_data['dist'])
        image_points = image_points.reshape(-1, 2).astype(np.int16)
        
        # 繪製三個坐標軸
        for start, end, color in zip(
            [0] * 3,
            range(1, 4),
            [(0,0,255), (0,255,0), (255,0,0)]
        ):
            cv2.line(cam_data['img'],
                    (image_points[start,0], image_points[start,1]),
                    (image_points[end,0], image_points[end,1]),
                    color, 5)
    
    # 在函數結尾處添加保存外部參數的代碼
    extrinsic_path = 'eight_camera_extrinsic.npz'
    np.savez(extrinsic_path,
             # ArUco到相機的變換
             R_aruco2l1=R_results['R_aruco2l1'], t_aruco2l1=t_results['t_aruco2l1'],
             R_aruco2l2=R_results['R_aruco2l2'], t_aruco2l2=t_results['t_aruco2l2'],
             R_aruco2l3=R_results['R_aruco2l3'], t_aruco2l3=t_results['t_aruco2l3'],
             R_aruco2c=R_results['R_aruco2c'], t_aruco2c=t_results['t_aruco2c'],
             R_aruco2r1=R_results['R_aruco2r1'], t_aruco2r1=t_results['t_aruco2r1'],
             R_aruco2r2=R_results['R_aruco2r2'], t_aruco2r2=t_results['t_aruco2r2'],
             R_aruco2r3=R_results['R_aruco2r3'], t_aruco2r3=t_results['t_aruco2r3'],
             R_aruco2r4=R_results['R_aruco2r4'], t_aruco2r4=t_results['t_aruco2r4'],
             # 相機到ArUco的變換
             R_l12aruco=R_results['R_l12aruco'], t_l12aruco=t_results['t_l12aruco'],
             R_l22aruco=R_results['R_l22aruco'], t_l22aruco=t_results['t_l22aruco'],
             R_l32aruco=R_results['R_l32aruco'], t_l32aruco=t_results['t_l32aruco'],
             R_c2aruco=R_results['R_c2aruco'], t_c2aruco=t_results['t_c2aruco'],
             R_r12aruco=R_results['R_r12aruco'], t_r12aruco=t_results['t_r12aruco'],
             R_r22aruco=R_results['R_r22aruco'], t_r22aruco=t_results['t_r22aruco'],
             R_r32aruco=R_results['R_r32aruco'], t_r32aruco=t_results['t_r32aruco'],
             R_r42aruco=R_results['R_r42aruco'], t_r42aruco=t_results['t_r42aruco'])
    
    print(f"\n外部參數已保存到: {os.path.abspath(extrinsic_path)}")
    
    return (R_results, t_results, cameras)

# 3. 定义处理单帧的函数
def process_frame(detector, frames, images, cams_params, cam_P):
    """
    處理八個相機的單幀圖像
    """
    # 相機配置
    cam_order = ['l1', 'l2', 'l3', 'c', 'r1', 'r2', 'r3', 'r4']
    cam_names = {
        'l1': "左側相機1", 'l2': "左側相機2", 'l3': "左側相機3",
        'c': "中心相機",
        'r1': "右側相機1", 'r2': "右側相機2", 'r3': "右側相機3", 'r4': "右側相機4"
    }
    
    # 解包相機參數
    (mtx_l1, dist_l1, mtx_l2, dist_l2, mtx_l3, dist_l3,
     mtx_c, dist_c,
     mtx_r1, dist_r1, mtx_r2, dist_r2, mtx_r3, dist_r3, mtx_r4, dist_r4) = cams_params
    
    # 定義坐標軸點
    axis_coord = np.array([
        [0,0,0],
        [200,0,0],  # X軸，紅色
        [0,200,0],  # Y軸，綠色
        [0,0,200]   # Z軸，藍色
    ], dtype=np.float32)
    
    # 在每個相機圖像上繪製 ArUco 坐標系
    camera_params = {
        'l1': (mtx_l1, dist_l1), 'l2': (mtx_l2, dist_l2), 'l3': (mtx_l3, dist_l3),
        'c': (mtx_c, dist_c),
        'r1': (mtx_r1, dist_r1), 'r2': (mtx_r2, dist_r2), 'r3': (mtx_r3, dist_r3), 'r4': (mtx_r4, dist_r4)
    }
    
    for cam_id in cam_order:
        mtx, dist = camera_params[cam_id]
        R = cam_P[f'R_aruco2{cam_id}']
        t = cam_P[f't_aruco2{cam_id}']
        
        # 將3D坐標投影到圖像平面
        rvec, _ = cv2.Rodrigues(R)
        imgpts, _ = cv2.projectPoints(axis_coord, rvec, t, mtx, dist)
        imgpts = imgpts.astype(np.int32)
        
        # 繪製坐標軸
        origin = tuple(imgpts[0].ravel())
        images[cam_id] = cv2.line(images[cam_id], origin, tuple(imgpts[1].ravel()), (0,0,255), 3)  # X軸，紅色
        images[cam_id] = cv2.line(images[cam_id], origin, tuple(imgpts[2].ravel()), (0,255,0), 3)  # Y軸，綠色
        images[cam_id] = cv2.line(images[cam_id], origin, tuple(imgpts[3].ravel()), (255,0,0), 3)  # Z軸，藍色
    
    # 原有的姿態檢測和處理代碼...
    mp_images = {
        cam_id: mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frames[cam_id], cv2.COLOR_BGR2RGB)
        )
        for cam_id in cam_order
    }
    
    detection_results = {
        cam_id: detector.detect(mp_images[cam_id])
        for cam_id in cam_order
    }
    
    # 檢查每個相機是否檢測到關鍵點
    poses = {}
    failed_cameras = []
    
    for cam_id in cam_order:
        result = detection_results[cam_id]
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            # 只有成功檢測到姿態的相機才會被加入poses字典
            poses[cam_id] = np.array([
                [landmark.x * frames[cam_id].shape[1], 
                 landmark.y * frames[cam_id].shape[0]]
                for landmark in result.pose_landmarks[0]
            ])
        else:
            failed_cameras.append(cam_names[cam_id])
    
    if failed_cameras:
        print(f"以下相機未檢測到人體姿勢: {', '.join(failed_cameras)}")
    
    # 如果檢測到的相機少於3個，則無法進行三維重建
    if len(poses) < 3:
        return np.zeros((33, 3)), images
    
    # 構建投影矩陣（只使用有檢測到的相機）
    cam_matrices = {}
    camera_params = {}
    for cam_id in poses.keys():  # 只處理成功檢測的相機
        R = cam_P[f'R_aruco2{cam_id}']
        t = cam_P[f't_aruco2{cam_id}']
        cam_matrices[cam_id] = np.hstack((R, t))
        
        # 獲取對應的相機參數
        if cam_id == 'l1': camera_params[cam_id] = (mtx_l1, dist_l1)
        elif cam_id == 'l2': camera_params[cam_id] = (mtx_l2, dist_l2)
        elif cam_id == 'l3': camera_params[cam_id] = (mtx_l3, dist_l3)
        elif cam_id == 'c': camera_params[cam_id] = (mtx_c, dist_c)
        elif cam_id == 'r1': camera_params[cam_id] = (mtx_r1, dist_r1)
        elif cam_id == 'r2': camera_params[cam_id] = (mtx_r2, dist_r2)
        elif cam_id == 'r3': camera_params[cam_id] = (mtx_r3, dist_r3)
        elif cam_id == 'r4': camera_params[cam_id] = (mtx_r4, dist_r4)
    
    # 進行多視圖三維重建（只使用有檢測到的相機）
    points_3d = triangulate_points_multi_cameras(
        poses,  # 只包含成功檢測的相機的2D點
        camera_params,  # 只包含成功檢測的相機的參數
        cam_matrices  # 只包含成功檢測的相機的投影矩陣
    )
    
    # 在圖像上標註檢測到的關鍵點
    for cam_id, pose in poses.items():
        for mark_i in range(33):
            mark_coord = (int(pose[mark_i,0]), int(pose[mark_i,1]))
            cv2.circle(images[cam_id], mark_coord, 10, (0, 0, 255), -1)
    
    return points_3d, images

def triangulate_points_multi_cameras(poses, camera_params, projection_matrices):
    """
    使用多個相機進行三角測量
    Args:
        poses: 字典，包含每個相機的2D點
        camera_params: 字典，包含每個相機的內參和畸變係數
        projection_matrices: 字典，包含每個相機的投影矩陣
    Returns:
        points_3d: 三維點雲 (33, 3)
    """
    points_3d = []
    num_points = 33  # MediaPipe姿態關鍵點數量
    
    # 對每個關鍵點進行三角測量
    for pt_idx in range(num_points):
        # 構建DLT方程組
        A = []
        
        for cam_id, pose in poses.items():
            # 獲取相機參數
            mtx, dist = camera_params[cam_id]
            P = projection_matrices[cam_id]
            
            # 去畸變
            pt = pose[pt_idx].reshape(1, 1, 2)
            pt_undist = cv2.undistortPoints(pt, mtx, dist)
            
            # 添加DLT約束
            x, y = pt_undist[0, 0]
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
        
        # 求解最小二乘問題
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        point_4d = Vt[-1]
        point_3d = (point_4d / point_4d[3])[:3]
        
        points_3d.append(point_3d)
    
    return np.array(points_3d)


    
# 5. 定义处理视频的主循环
def process_videos(video_paths, start_frame=0):
    """
    處理八個相機的視頻
    Args:
        video_paths: 字典，包含所有相機的視頻路徑
            {
                'l1': 左側相機1路徑, 'l2': 左側相機2路徑, 'l3': 左側相機3路徑,
                'c': 中心相機路徑,
                'r1': 右側相機1路徑, 'r2': 右側相機2路徑, 'r3': 右側相機3路徑, 'r4': 右側相機4路徑
            }
        start_frame: 起始幀（默認為0）
    Returns:
        all_points_3d: 所有幀的3D骨骼點
        aruco_axis: ArUco坐標系
        cam_axes: 所有相機坐標系的字典
    """
    ## 0. 導入相機內參"
    camera_params_path = r"C:\Users\user\Desktop\Dropbox\camera_8\eight_camera_calibration.npz"
    if not os.path.exists(camera_params_path):
        print(f'相機參數文件不存在: {camera_params_path}')
        print('請先運行相機標定程序生成參數文件')
        return
    
    # 載入八個相機的參數
    cams_params = load_camera_params(camera_params_path)
    
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

    ## 2. 設置MediaPipe姿態估計模型
    model_asset_path = r"C:\Users\user\Desktop\pose_landmarker_full.task"
    if not os.path.exists(model_asset_path):
        print('model_asset_path is error.')
        return
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.7,    # 提高到0.7（原為0.3）
        min_pose_presence_confidence=0.7,     # 提高到0.7（原為0.3）
        min_tracking_confidence=0.7,          # 提高到0.7（原為0.3）
        num_poses=1                          # 限制檢測人數為1
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    ## 3. 讀取視頻
    print(f"開始處理視頻從幀 {start_frame} 開始...")
    
    # 初始化所有相機的視頻捕獲
    caps = {}
    cam_order = ['l1', 'l2', 'l3', 'c', 'r1', 'r2', 'r3', 'r4']
    for cam_id in cam_order:
        caps[cam_id] = cv2.VideoCapture(video_paths[cam_id])
        if not caps[cam_id].isOpened():
            raise ValueError(f"無法打開{cam_id}相機的視頻文件")
        caps[cam_id].set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 讀取第一幀來初始化並獲取 ArUco 外參
    frames = {}
    for cam_id, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            print(f"無法從 {cam_id} 相機讀取第一幀")
            return None, None, None
        frames[cam_id] = frame
        # 重置視頻位置到起始幀
        caps[cam_id].set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 從第一幀獲取 ArUco 外參
    result = get_aruco_axis(
        img_c=frames['c'],
        img_l1=frames['l1'],
        img_l2=frames['l2'],
        img_l3=frames['l3'],
        img_r1=frames['r1'],
        img_r2=frames['r2'],
        img_r3=frames['r3'],
        img_r4=frames['r4'], 
        aruco_detector=aruco_detector, 
        board_coord=board_coord, 
        cams_params=cams_params)
    
    if result is None:
        print("無法從第一幀獲取 ArUco 外參")
        return None, None, None
        
    # 解包結果並構建相機投影矩陣字典
    R_results, t_results, _ = result
    cam_P = {}
    for cam_id in cam_order:
        cam_P[f'R_aruco2{cam_id}'] = R_results[f'R_aruco2{cam_id}']
        cam_P[f't_aruco2{cam_id}'] = t_results[f't_aruco2{cam_id}']
    
    # 初始化圖像字典用於顯示
    images = {cam_id: frame.copy() for cam_id, frame in frames.items()}
    
    # 初始化數據收集列表
    all_points_3d = []  # 3d骨骼點
    aruco_axis = []     # aruco坐標系
    cam_axes = {cam_id: [] for cam_id in cam_order}  # 各相機坐標系
    frame_count = start_frame
    
    # 坐標系的4個點
    axis_coord = np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ], dtype=np.float32) * 200
    
    # 開始循環讀取每一幀視頻圖像
    while True:
        frames = {}
        all_ret = True
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                all_ret = False
                break
            frames[cam_id] = frame
        if not all_ret:
            break
            
        # 初始化圖像字典用於顯示
        images = {cam_id: frame.copy() for cam_id, frame in frames.items()}
        
        print(f"處理第 {frame_count + 1} 幀")
        if frame_count == 912:
            break
            
        # 處理姿態估計
        points_3d, updated_images = process_frame(detector, frames, images, cams_params, cam_P)
        
        # 更新坐標系（使用第一幀的變換矩陣）
        if frame_count == 0:
            # 第一幀：初始化坐標系
            first_aruco_axis = axis_coord.copy()
            first_cam_axes = {}
            for cam_id in cam_order:
                R = R_results[f'R_{cam_id}2aruco']
                t = t_results[f't_{cam_id}2aruco']
                first_cam_axes[cam_id] = np.dot(R, axis_coord.T).T + t.T
        
        # 每一幀都使用相同的坐標系（相對於第一幀）
        aruco_axis.append(first_aruco_axis)
        for cam_id in cam_order:
            cam_axes[cam_id].append(first_cam_axes[cam_id])
        
        if points_3d is not None:
            # 將點雲轉換到世界坐標系（ArUco坐標系）
            all_points_3d.append(points_3d)
        else:
            all_points_3d.append(np.zeros((33, 3)))
        
        # 顯示圖像
        # 將8個相機的圖像排列為2x4的網格
        top_row = np.hstack([
            cv2.resize(updated_images['l1'], (320, 240)),
            cv2.resize(updated_images['l2'], (320, 240)),
            cv2.resize(updated_images['l3'], (320, 240)),
            cv2.resize(updated_images['c'], (320, 240))
        ])
        bottom_row = np.hstack([
            cv2.resize(updated_images['r1'], (320, 240)),
            cv2.resize(updated_images['r2'], (320, 240)),
            cv2.resize(updated_images['r3'], (320, 240)),
            cv2.resize(updated_images['r4'], (320, 240))
        ])
        combined_img = np.vstack([top_row, bottom_row])
        
        cv2.imshow('Eight Camera Views', combined_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"已處理 {frame_count} 幀")
    
    # 釋放資源
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    
    print(f"視頻處理完成。共處理了 {frame_count - start_frame} 幀")
    
    return (np.array(all_points_3d), 
            first_aruco_axis,  # 只返回第一幀的 ArUco 坐標系
            first_cam_axes)    # 只返回第一幀的相機坐標系

def visualize_3d_animation_eight_cameras(points, aruco_axis, cam_axes, title='3D Visualization'):
    """
    八相機系統的3D點雲動畫可視化
    Args:
        points: 3D關鍵點序列 [frames, 33, 3]
        aruco_axis: 第一幀的ArUco坐標系 [4, 3]
        cam_axes: 所有相機在第一幀的坐標系 {'l1': [4, 3], 'l2': [4, 3], ...}
        title: 視窗標題
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 計算所有點的範圍（只使用人體關鍵點）
    min_vals = np.min(points.reshape(-1, 3), axis=0)
    max_vals = np.max(points.reshape(-1, 3), axis=0)
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
    
    # 定義骨架連接
    connections = [
        # 頭部
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (3, 6),
        # 頸部
        (9, 10),
        # 軀幹
        (11, 12), (11, 23), (12, 24), (23, 24),
        # 左臂
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (19, 21),
        # 右臂
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        (18, 20), (20, 22),
        # 左腿
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
        # 右腿
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
    ]
    
    # 為不同部位設置顏色
    colors = {
        'head': 'purple',
        'spine': 'blue',
        'arms': 'green',
        'legs': 'red',
        'hands': 'orange'
    }
    
    # 定義每個連接的顏色
    connection_colors = []
    for start, end in connections:
        if start <= 8 or end <= 8:  # 頭部
            connection_colors.append(colors['head'])
        elif start in [9, 10, 11] or end in [9, 10, 11]:  # 脊椎
            connection_colors.append(colors['spine'])
        elif (start in [13, 14, 15, 16] or end in [13, 14, 15, 16]):  # 手臂
            connection_colors.append(colors['arms'])
        elif start >= 17 or end >= 17:  # 手部
            connection_colors.append(colors['hands'])
        else:  # 腿部
            connection_colors.append(colors['legs'])
    
    # 創建線條
    lines = []
    for color in connection_colors:
        line, = ax.plot([], [], [], color=color, alpha=0.8, linewidth=2)
        lines.append(line)
    
    # 繪製固定的坐標系
    # ArUco坐標系
    for i in range(3):
        ax.plot3D([aruco_axis[0,0], aruco_axis[i+1,0]],
                 [aruco_axis[0,1], aruco_axis[i+1,1]],
                 [aruco_axis[0,2], aruco_axis[i+1,2]],
                 color=['r','g','b'][i], linewidth=2, alpha=0.7)
    
    # 相機坐標系
    cam_order = ['l1', 'l2', 'l3', 'c', 'r1', 'r2', 'r3', 'r4']
    for cam_id in cam_order:
        for i in range(3):
            ax.plot3D([cam_axes[cam_id][0,0], cam_axes[cam_id][i+1,0]],
                     [cam_axes[cam_id][0,1], cam_axes[cam_id][i+1,1]],
                     [cam_axes[cam_id][0,2], cam_axes[cam_id][i+1,2]],
                     color=['r','g','b'][i], linewidth=2, alpha=0.7)
    
    def update(frame):
        # 更新骨骼點
        point_cloud = points[frame]
        scatter._offsets3d = (point_cloud[:,0], point_cloud[:,1], point_cloud[:,2])
        
        # 更新骨架線條
        for i, ((start, end), line) in enumerate(zip(connections, lines)):
            line.set_data_3d([point_cloud[start,0], point_cloud[end,0]],
                           [point_cloud[start,1], point_cloud[end,1]],
                           [point_cloud[start,2], point_cloud[end,2]])
        
        # 更新標題顯示當前幀
        ax.set_title(f'{title} - Frame: {frame}')
        
        return [scatter] + lines
    
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
    # # 确保视频路径正确
    video_paths = {
        'l1': r"E:\20250116\recordings\part03\port_6-01162025125609-0000.avi",#"D:\20250116\recordings\part03\port_6-01162025125609-0000.avi"
        'l2': r"E:\20250116\recordings\part03\port_7-01162025125620-0000.avi",#"D:\20250116\recordings\part03\port_7-01162025125620-0000.avi"
        'l3': r"E:\20250116\recordings\part03\port_8-01162025125607-0000.avi",#"D:\20250116\recordings\part03\port_8-01162025125607-0000.avi"
        'c': r"E:\20250116\recordings\part03\port_5-01162025125613-0000.avi",#"D:\20250116\recordings\part03\port_5-01162025125613-0000.avi"
        'r1': r"E:\20250116\recordings\part03\port_4-01162025125615-0000.avi",#"D:\20250116\recordings\part03\port_4-01162025125615-0000.avi"
        'r2': r"E:\20250116\recordings\part03\port_3-01162025125611-0000.avi",#"D:\20250116\recordings\part03\port_3-01162025125611-0000.avi"
        'r3': r"E:\20250116\recordings\part03\port_2-01162025125617-0000.avi",#"D:\20250116\recordings\part03\port_2-01162025125617-0000.avi"
        'r4': r"E:\20250116\recordings\part03\port_1-01162025125622-0000.avi"#""D:\20250116\recordings\part03\port_1-01162025125622-0000.avi"
    }
    
    points_3d, aruco_axis, cam_axes = process_videos(video_paths)

    if len(points_3d) == 0:
        print("未能从视频中提取任何3D关键点。")
        return

    # 可視化3D點雲
    print("開始3D可視化...")
    visualize_3d_animation_eight_cameras(
        points_3d,
        aruco_axis,
        cam_axes,
        title='Eight Camera Motion Capture'
    )



if __name__ == "__main__":
    main()

