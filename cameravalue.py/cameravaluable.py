import cv2
import numpy as np
import os
import glob
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
# 導入所需的庫
import cv2  # OpenCV庫,用於圖像處理
import numpy as np  # 數值計算庫
import mediapipe as mp  # MediaPipe庫,用於姿態估計
from mediapipe.tasks import python  # MediaPipe Python任務庫
import matplotlib.pyplot as plt  # 繪圖庫
from mediapipe.tasks.python import vision  # MediaPipe視覺任務庫
from scipy.signal import savgol_filter  # 信號平滑濾波器
from scipy.optimize import minimize  # 最佳化函數
import time  # 時間處理
import os  # 操作系統接口
import pandas as pd  # 數據分析庫
import pyvista as pv  # 3D可視化庫
from pyvistaqt import BackgroundPlotter  # 交互式3D繪圖
import math  # 數學函數
from matplotlib.animation import FuncAnimation  # 動畫製作
# 設置ffmpeg路徑
os.environ['PATH'] += ';D:\\software\\ffmpeg-7.1-essentials_build'
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = 'D:\\software\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe'
from collections import defaultdict  # 預設字典
import numpy as np  # 再次導入numpy(可刪除)
import cv2  # 再次導入cv2(可刪除)
import glob  # 文件路徑匹配
import os  # 再次導入os(可刪除)
import seaborn as sns  # 統計數據可視化
from config import *  # 導入配置文件中的所有變量

# 初始化全局變量
error_data = []  # 用於存儲重投影誤差數據

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
                          right_1_folder, right_2_folder, right_3_folder, right_4_folder,
                          pattern_size=(9,6), square_size=30.0):
    """
    八相機標定主函數
    Args:
        left_1_folder: 左側相機1標定圖片文件夾
        left_2_folder: 左側相機2標定圖片文件夾
        left_3_folder: 左側相機3標定圖片文件夾
        center_folder: 中心相機標定圖片文件夾
        right_1_folder: 右側相機1標定圖片文件夾
        right_2_folder: 右側相機2標定圖片文件夾
        right_3_folder: 右側相機3標定圖片文件夾
        right_4_folder: 右側相機4標定圖片文件夾
        pattern_size: 棋盤格內角點數量(寬,高)，默認(9,6)
        square_size: 棋盤格方格實際尺寸(mm)，默認30.0
    """
    print("開始左側相機1標定...")
    ret_l1, mtx_l1, dist_l1, rvecs_l1, tvecs_l1 = calibrate_single_camera(left_1_folder, pattern_size, square_size)
    
    print("\n開始左側相機2標定...")
    ret_l2, mtx_l2, dist_l2, rvecs_l2, tvecs_l2 = calibrate_single_camera(left_2_folder, pattern_size, square_size)
    
    print("\n開始左側相機3標定...")
    ret_l3, mtx_l3, dist_l3, rvecs_l3, tvecs_l3 = calibrate_single_camera(left_3_folder, pattern_size, square_size)
    
    print("\n開始中心相機標定...")
    ret_c, mtx_c, dist_c, rvecs_c, tvecs_c = calibrate_single_camera(center_folder, pattern_size, square_size)
    
    print("\n開始右側相機1標定...")
    ret_r1, mtx_r1, dist_r1, rvecs_r1, tvecs_r1 = calibrate_single_camera(right_1_folder, pattern_size, square_size)
    
    print("\n開始右側相機2標定...")
    ret_r2, mtx_r2, dist_r2, rvecs_r2, tvecs_r2 = calibrate_single_camera(right_2_folder, pattern_size, square_size)
    
    print("\n開始右側相機3標定...")
    ret_r3, mtx_r3, dist_r3, rvecs_r3, tvecs_r3 = calibrate_single_camera(right_3_folder, pattern_size, square_size)
    
    print("\n開始右側相機4標定...")
    ret_r4, mtx_r4, dist_r4, rvecs_r4, tvecs_r4 = calibrate_single_camera(right_4_folder, pattern_size, square_size)
    
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
                
                # 使用solvePnPGeneric解決PnP問題
                image_C = np.ascontiguousarray(img_coord[:,:2]).reshape((-1,1,2))
                ret, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(
                    tem_coord, 
                    image_C, 
                    cam_data['mtx'], 
                    cam_data['dist'],
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    reprojectionError=True,
                    useExtrinsicGuess=False,
                    rvec=None,
                    tvec=None
                )
                
                if ret > 0:
                    min_error_idx = np.argmin(reprojectionError)
                    # 添加調試打印
                    print(f"Camera {cam_id}, Marker {cam_data['ids'][i][0]}, Error: {reprojectionError[min_error_idx]}")
                    
                    # 確保誤差值是有效的數字
                    if not np.isnan(reprojectionError[min_error_idx]):
                        error_data.append(
                            f"Camera {cam_id}, Marker {cam_data['ids'][i][0]}, "
                            f"Reprojection Error: {reprojectionError[min_error_idx]}"
                        )
                    
                    R, _ = cv2.Rodrigues(rvecs[min_error_idx])
                    
                    # 計算ArUco點在相機坐標系中的位置
                    aruco_p = np.dot(R, tem_coord.T).T + tvecs[min_error_idx].T
                    aruco_ps.append(aruco_p)
                else:
                    print(f"solvePnPGeneric failed for camera {cam_id}, marker {cam_data['ids'][i][0]}")
        
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
                np.ones((len(results[cam_id]['img_coords']), 1))
            ))
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
        # 計算相機到ArUco的旋轉矩陣 - 旋轉矩陣的逆等於其轉置
        R_cam2aruco = R_aruco2cam.T  
        
        # 計算相機到ArUco的平移向量 - 根據坐標系變換公式:
        # t_cam2aruco = -(R_cam2aruco @ t_aruco2cam)
        # 因為 R_cam2aruco = R_aruco2cam.T,所以可以寫成:
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
        
        print(f"相機 {cam_id} 使用了 {len(results[cam_id]['point_coords'])} 個點進行外部參數求解")
    
    # 可視化重投影誤差
    if error_data:
        visualize_reprojection_errors(error_data)
    
    return (R_results, t_results, cameras)

# 3. 定义处理单帧的函数
def calculate_rmse(points_2d, points_3d, camera_params, projection_matrices):
    """
    計算重投影誤差的RMSE
    Args:
        points_2d: 字典，包含每個相機檢測到的2D點
        points_3d: 三角測量得到的3D點 (33, 3)
        camera_params: 字典，包含每個相機的內參和畸變係數
        projection_matrices: 字典，包含每個相機的投影矩陣
    Returns:
        rmse: 總體RMSE值
        rmse_per_camera: 每個相機的RMSE值字典
    """
    squared_errors = []
    errors_per_camera = {cam_id: [] for cam_id in points_2d.keys()}
    
    for cam_id, pose_2d in points_2d.items():
        mtx, dist = camera_params[cam_id]
        P = projection_matrices[cam_id]
        R = P[:, :3]
        t = P[:, 3:]
        
        # 將3D點投影到2D
        rvec, _ = cv2.Rodrigues(R)
        projected_points, _ = cv2.projectPoints(points_3d, rvec, t, mtx, dist)
        projected_points = projected_points.reshape(-1, 2)
        
        # 計算每個點的誤差
        errors = np.linalg.norm(pose_2d - projected_points, axis=1)
        valid_errors = errors[~np.isnan(errors)]  # 排除無效值
        
        if len(valid_errors) > 0:
            squared_errors.extend(valid_errors ** 2)
            errors_per_camera[cam_id] = np.sqrt(np.mean(valid_errors ** 2))
    
    # 計算總體RMSE
    if squared_errors:
        rmse = np.sqrt(np.mean(squared_errors))
    else:
        rmse = float('inf')
    
    # 添加RMSE閾值過濾
    RMSE_THRESHOLD = 10.0  # 設置閾值
    for cam_id in errors_per_camera:
        if errors_per_camera[cam_id] > RMSE_THRESHOLD:
            print(f"警告: {cam_id}相機RMSE過大: {errors_per_camera[cam_id]:.2f}")
            # 可以選擇忽略該相機的數據
            errors_per_camera[cam_id] = float('inf')
    
    return rmse, errors_per_camera

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
    
    # 姿態檢測和處理
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
    
    # 修改姿態檢測部分，添加可見度信息
    poses = {}
    visibilities = {}  # 新增可見度字典
    failed_cameras = []
    
    for cam_id in cam_order:
        result = detection_results[cam_id]
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            # 存儲2D坐標
            poses[cam_id] = np.array([
                [landmark.x * frames[cam_id].shape[1], 
                 landmark.y * frames[cam_id].shape[0]]
                for landmark in result.pose_landmarks[0]
            ])
            # 存儲可見度
            visibilities[cam_id] = np.array([
                landmark.visibility
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
    camera_params_detected = {}
    for cam_id in poses.keys():  # 只處理成功檢測的相機
        R = cam_P[f'R_aruco2{cam_id}']
        t = cam_P[f't_aruco2{cam_id}']
        cam_matrices[cam_id] = np.hstack((R, t))
        
        # 獲取對應的相機參數
        if cam_id == 'l1': camera_params_detected[cam_id] = (mtx_l1, dist_l1)
        elif cam_id == 'l2': camera_params_detected[cam_id] = (mtx_l2, dist_l2)
        elif cam_id == 'l3': camera_params_detected[cam_id] = (mtx_l3, dist_l3)
        elif cam_id == 'c': camera_params_detected[cam_id] = (mtx_c, dist_c)
        elif cam_id == 'r1': camera_params_detected[cam_id] = (mtx_r1, dist_r1)
        elif cam_id == 'r2': camera_params_detected[cam_id] = (mtx_r2, dist_r2)
        elif cam_id == 'r3': camera_params_detected[cam_id] = (mtx_r3, dist_r3)
        elif cam_id == 'r4': camera_params_detected[cam_id] = (mtx_r4, dist_r4)
    
    # 修改三角測量調用，添加可見度參數
    points_3d = triangulate_points_multi_cameras(
        poses,  
        camera_params_detected,  
        cam_matrices,
        visibilities  # 新增可見度參數
    )
    
    # 在圖像上標註檢測到的關鍵點
    for cam_id, pose in poses.items():
        visibility = visibilities[cam_id]  # 獲取當前相機的可見度
        for mark_i in range(33):
            mark_coord = (int(pose[mark_i,0]), int(pose[mark_i,1]))
            # 根據可見度調整顏色
            visibility_color = int(255 * visibility[mark_i])
            cv2.circle(images[cam_id], mark_coord, 10, 
                      (0, visibility_color, 255-visibility_color), -1)
    
    # 在三角測量之後添加RMSE計算
    if len(poses) >= 3:  # 確保有足夠的相機檢測到姿態
        # 計算RMSE
        rmse, rmse_per_camera = calculate_rmse(
            poses,  # 2D檢測結果
            points_3d,  # 三角測量得到的3D點
            camera_params_detected,  # 相機參數
            cam_matrices  # 投影矩陣
        )
        
        # 在圖像上顯示RMSE
        for cam_id in poses.keys():
            if cam_id in rmse_per_camera:
                text = f'RMSE: {rmse_per_camera[cam_id]:.2f}'
                cv2.putText(images[cam_id], text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 打印總體RMSE
        print(f'Total RMSE: {rmse:.2f}')
        print('Per-camera RMSE:')
        for cam_id, error in rmse_per_camera.items():
            print(f'{cam_id}: {error:.2f}')
    
    return points_3d, images

def triangulate_points_multi_cameras(poses, camera_params, projection_matrices, visibilities):
    """
    使用多個相機進行三角測量，考慮關鍵點可見度
    Args:
        poses: 字典，包含每個相機的2D點
        camera_params: 字典，包含每個相機的內參和畸變係數
        projection_matrices: 字典，包含每個相機的投影矩陣
        visibilities: 字典，包含每個相機檢測到的關鍵點可見度
    Returns:
        points_3d: 三維點雲 (33, 3)
    """
    points_3d = []
    num_points = 33  # MediaPipe姿態關鍵點數量
    
    # 對每個關鍵點進行三角測量
    for pt_idx in range(num_points):
        # 構建加權DLT方程組
        A = []
        weights = []
        valid_cameras = []  # 新增：記錄參與重建的相機
        
        for cam_id, pose in poses.items():
            # 獲取相機參數
            mtx, dist = camera_params[cam_id]
            P = projection_matrices[cam_id]
            
            # 獲取當前點的可見度作為權重
            visibility = visibilities[cam_id][pt_idx]
            
            if visibility < 0.8:
                continue
                
            valid_cameras.append(cam_id)  # 記錄有效相機
            
            # 獲取相機參數
            mtx, dist = camera_params[cam_id]
            P = projection_matrices[cam_id]
            
            # 去畸變
            pt = pose[pt_idx].reshape(1, 1, 2)
            pt_undist = cv2.undistortPoints(pt, mtx, dist)
            
            # 添加DLT約束，並考慮權重
            x, y = pt_undist[0, 0]
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
            # 每個點對應兩個方程，所以添加兩次權重
            weights.extend([visibility, visibility])
        
        # 輸出診斷信息
        print(f"Point {pt_idx}: Using {len(valid_cameras)} cameras: {valid_cameras}")
        
        if len(A) < 4:
            print(f"Warning: Point {pt_idx} has insufficient observations")
            points_3d.append(np.zeros(3))
            continue
            
        # 將權重轉換為對角矩陣
        A = np.array(A)
        W = np.diag(weights)
        
        # 求解加權最小二乘問題：min ||WA·X||^2
        weighted_A = W @ A
        _, _, Vt = np.linalg.svd(weighted_A)
        point_4d = Vt[-1]
        point_3d = (point_4d / point_4d[3])[:3]
        
        points_3d.append(point_3d)
    
    return np.array(points_3d)

def define_aruco_parameters():
    """
    定義 ArUco 標記板的參數
    Returns:
        board_coord: ArUco 標記板的坐標字典
        axis_coord: 坐標軸點
    """
    # 從配置文件獲取參數
    board_length = ARUCO_CONFIG['board_length']
    board_gap = ARUCO_CONFIG['board_gap']
    
    # 基礎坐標
    base_coord = np.array([[0,0],[0,1],[1,1],[1,0]])
    
    # 定義6個標記的位置
    board_coord = {
        0: base_coord * board_length + [0,0],
        1: base_coord * board_length + [board_length+board_gap,0],
        2: base_coord * board_length + [0,board_length+board_gap],
        3: base_coord * board_length + [board_length+board_gap,board_length+board_gap],
        4: base_coord * board_length + [0,(board_length+board_gap)*2],
        5: base_coord * board_length + [board_length+board_gap,(board_length+board_gap)*2],
    }
    
    # 定義坐標軸點
    axis_coord = np.array([
        [0,0,0],
        [200,0,0],  # X軸，紅色
        [0,200,0],  # Y軸，綠色
        [0,0,200]   # Z軸，藍色
    ], dtype=np.float32)
    
    return board_coord, axis_coord

def setup_aruco_detector():
    """
    設置 ArUco 檢測器
    Returns:
        aruco_detector: ArUco 檢測器實例
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    return aruco_detector

# 5. 定义处理视频的主循环
def process_videos(video_paths, start_frame=0):
    """
    處理八個相機的視頻
    """
    ## 0. 驗證所有路徑
    validate_paths()
    
    ## 1. 載入相機參數
    cams_params = load_camera_params(CAMERA_PARAMS_PATH)
    
    ## 2. 設置aruco參數
    aruco_detector = setup_aruco_detector()
    board_coord, axis_coord = define_aruco_parameters()

    ## 3. 設置MediaPipe姿態估計模型
    base_options = python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        min_pose_detection_confidence=MEDIAPIPE_CONFIG['min_pose_detection_confidence'],
        min_pose_presence_confidence=MEDIAPIPE_CONFIG['min_pose_presence_confidence'],
        min_tracking_confidence=MEDIAPIPE_CONFIG['min_tracking_confidence']
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    ## 4. 讀取視頻
    print(f"開始處理視頻從幀 {start_frame} 開始...")
    
    # 初始化所有相機的視頻捕獲
    caps = {}
    for cam_id in CAM_ORDER:
        caps[cam_id] = cv2.VideoCapture(video_paths[cam_id])
        if not caps[cam_id].isOpened():
            raise ValueError(f"無法打開{CAM_NAMES[cam_id]}的視頻文件")
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
    for cam_id in CAM_ORDER:
        cam_P[f'R_aruco2{cam_id}'] = R_results[f'R_aruco2{cam_id}']
        cam_P[f't_aruco2{cam_id}'] = t_results[f't_aruco2{cam_id}']
    
    # 初始化圖像字典用於顯示
    images = {cam_id: frame.copy() for cam_id, frame in frames.items()}
    
    # 初始化數據收集列表
    all_points_3d = []  # 3d骨骼點
    aruco_axis = []     # aruco坐標系
    cam_axes = {cam_id: [] for cam_id in CAM_ORDER}  # 各相機坐標系
    frame_count = start_frame
    
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
            for cam_id in CAM_ORDER:
                R = R_results[f'R_{cam_id}2aruco']
                t = t_results[f't_{cam_id}2aruco']
                first_cam_axes[cam_id] = np.dot(R, axis_coord.T).T + t.T
        
        # 每一幀都使用相同的坐標系（相對於第一幀）
        aruco_axis.append(first_aruco_axis)
        for cam_id in CAM_ORDER:
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
        elif start in [13, 14, 15, 16] or end in [13, 14, 15, 16]:  # 手臂
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
    for cam_id in CAM_ORDER:
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
    """
    主程序
    """
    # # 确保视频路径正确
    video_paths = {
        'l1': r"D:\20250116\recordings\part03\port_6-01162025125609-0000.avi",#"D:\20250116\recordings\part03\port_6-01162025125609-0000.avi"
        'l2': r"D:\20250116\recordings\part03\port_7-01162025125620-0000.avi",#"D:\20250116\recordings\part03\port_7-01162025125620-0000.avi"
        'l3': r"D:\20250116\recordings\part03\port_8-01162025125607-0000.avi",#"D:\20250116\recordings\part03\port_8-01162025125607-0000.avi"
        'c': r"D:\20250116\recordings\part03\port_5-01162025125613-0000.avi",#"D:\20250116\recordings\part03\port_5-01162025125613-0000.avi"
        'r1': r"D:\20250116\recordings\part03\port_4-01162025125615-0000.avi",#"D:\20250116\recordings\part03\port_4-01162025125615-0000.avi"
        'r2': r"D:\20250116\recordings\part03\port_3-01162025125611-0000.avi",#"D:\20250116\recordings\part03\port_3-01162025125611-0000.avi"
        'r3': r"D:\20250116\recordings\part03\port_2-01162025125617-0000.avi",#"D:\20250116\recordings\part03\port_2-01162025125617-0000.avi"
        'r4': r"D:\20250116\recordings\part03\port_1-01162025125622-0000.avi"#""D:\20250116\recordings\part03\port_1-01162025125622-0000.avi"
    }
    
    # 處理視頻
    points_3d, aruco_axis, cam_axes = process_videos(video_paths)
    
    if len(points_3d) == 0:
        print("未能從視頻中提取任何3D關鍵點。")
        return
    
    # 可視化3D點雲
    print("開始3D可視化...")
    visualize_3d_animation_eight_cameras(
        points_3d,
        aruco_axis,
        cam_axes,
        title='Eight Camera Motion Capture'
    )

def visualize_reprojection_errors(error_data):
    """
    可視化重投影誤差
    Args:
        error_data: 包含相機ID、Marker ID和重投影誤差的列表
    """
    # 整理數據
    cameras = ['l1', 'l2', 'l3', 'c', 'r1', 'r2', 'r3', 'r4']
    markers = list(range(6))  # 0-5號標記
    
    # 創建誤差矩陣
    error_matrix = np.zeros((len(cameras), len(markers)))
    error_matrix.fill(np.nan)  # 初始化為NaN，表示缺失數據
    
    # 解析數據
    for line in error_data:
        parts = line.split(', ')
        cam = parts[0].split(' ')[1]
        marker = int(parts[1].split(' ')[1])
        error = float(parts[2].split(': ')[1].strip('[]'))
        
        # 填充誤差矩陣
        cam_idx = cameras.index(cam)
        error_matrix[cam_idx, marker] = error
    
    # 創建圖表，調整大小和子圖間距
    plt.figure(figsize=(15, 12))  # 增加圖表高度
    plt.subplots_adjust(hspace=0.3)  # 增加子圖之間的間距
    
    # 1. 熱力圖
    plt.subplot(211)
    sns.heatmap(error_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=markers,
                yticklabels=cameras,
                cbar_kws={'label': 'Reprojection Error (pixels)'})
    plt.title('重投影誤差熱力圖')
    plt.xlabel('Marker ID')
    plt.ylabel('Camera ID')
    
    # 2. 箱型圖
    plt.subplot(212)
    data_for_box = []
    labels = []
    for i, cam in enumerate(cameras):
        valid_errors = error_matrix[i, ~np.isnan(error_matrix[i, :])]
        data_for_box.append(valid_errors)
        labels.append(f'{cam}\n(mean={np.mean(valid_errors):.3f})')
    
    plt.boxplot(data_for_box, labels=labels)
    plt.title('各相機重投影誤差分布')
    plt.ylabel('Reprojection Error (pixels)')
    plt.grid(True, alpha=0.3)
    
    # 調整統計信息的位置，移到左上角
    overall_mean = np.nanmean(error_matrix)
    overall_std = np.nanstd(error_matrix)
    plt.figtext(0.02, 0.95, 
                f'Overall Statistics:\n'
                f'Mean Error: {overall_mean:.3f} pixels\n'
                f'Std Dev: {overall_std:.3f} pixels\n'
                f'Max Error: {np.nanmax(error_matrix):.3f} pixels\n'
                f'Min Error: {np.nanmin(error_matrix):.3f} pixels',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()  # 自動調整布局
    plt.show()

def check_license():
    """
    添加許可證檢查機制
    """
    license_key = "YOUR-LICENSE-KEY"
    expiration_date = "2025-01-24"
    
    if not verify_license(license_key, expiration_date):
        raise RuntimeError("未授權使用")

def verify_license(license_key, expiration_date):
    """
    驗證許可證
    Args:
        license_key: 許可證密鑰
        expiration_date: 過期日期
    Returns:
        bool: 許可證是否有效
    """
    # 簡單的許可證檢查
    if license_key == "YOUR-LICENSE-KEY":
        from datetime import datetime
        expiry = datetime.strptime(expiration_date, "%Y-%m-%d")
        return datetime.now() <= expiry
    return False




if __name__ == "__main__":
    main()
# 假設需要的變數
CAMERA_CONFIG = {
    'left': 3,  # 左側相機數量
    'right': 4,  # 右側相機數量
    'center': 1,  # 中心相機數量
    'all': 8  # 總共相機數量
}

# 根據相機配置自動設置相機文件夾路徑
def get_camera_folders(base_folder, cam_config):
    camera_folders = {
        'left': [os.path.join(base_folder, f'left{i+1}') for i in range(cam_config['left'])],
        'right': [os.path.join(base_folder, f'right{i+1}') for i in range(cam_config['right'])],
        'center': [os.path.join(base_folder, f'center{i+1}') for i in range(cam_config['center'])]
    }
    return camera_folders

# 單個相機標定函數
def calibrate_single_camera(images_folder, pattern_size=(9,6), square_size=30.0):
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp = objp * square_size
    
    objpoints = []
    imgpoints = []
    
    images = glob.glob(os.path.join(images_folder, '*.jpeg'))
    if not images:
        images = glob.glob(os.path.join(images_folder, '*.png'))
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow('Corners', cv2.resize(img, (800,600)))
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"平均重投影誤差: {mean_error/len(objpoints)}")
    
    return ret, mtx, dist, rvecs, tvecs

# 相機標定的處理函數
def calibrate_cameras(camera_folders, pattern_size=(9,6), square_size=30.0):
    # 假設有三種類型的相機：左側、右側和中心
    for camera_type, folders in camera_folders.items():
        for folder in folders:
            print(f"開始對 {folder} 進行標定...")
            # 調用單個相機標定函數
            ret, mtx, dist, rvecs, tvecs = calibrate_single_camera(folder, pattern_size, square_size)
            # 保存標定結果
            save_calibration_result(camera_type, folder, mtx, dist, rvecs, tvecs)

# 保存標定結果
def save_calibration_result(camera_type, folder, mtx, dist, rvecs, tvecs):
    # 假設這裡需要將標定結果保存到文件中
    result_file = f'{camera_type}_calibration_results.npz'
    np.savez(result_file, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print(f"已保存標定結果到 {result_file}")

# 動態處理每個相機進行三維重建
def process_reconstruction(cameras, detector, frames, camera_params, cam_P):
    cam_order = list(cameras.keys())  # 按相機類型動態排列
    for cam_id in cam_order:
        print(f"處理相機 {cam_id} 的三維重建")
        # 繪製每個相機的坐標系
        R = cam_P[f'R_aruco2{cam_id}']
        t = cam_P[f't_aruco2{cam_id}']
        # 進行其他處理...
    
    return points_3d, updated_images  # 返回更新的3D點雲和圖片

# 3D可視化動畫
def visualize_3d_animation_eight_cameras(points, aruco_axis, cam_axes, title='3D Visualization'):
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
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (3, 6),
        (9, 10),
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (19, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        (18, 20), (20, 22),
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
    ]
    
    # 創建線條
    lines = []
    for _ in connections:
        line, = ax.plot([], [], [], color='r', alpha=0.8, linewidth=2)
        lines.append(line)
    
    def update(frame):
        point_cloud = points[frame]
        scatter._offsets3d = (point_cloud[:,0], point_cloud[:,1], point_cloud[:,2])
        
        for i, (start, end) in enumerate(connections):
            lines[i].set_data_3d([point_cloud[start,0], point_cloud[end,0]],
                                 [point_cloud[start,1], point_cloud[end,1]],
                                 [point_cloud[start,2], point_cloud[end,2]])
        
        return [scatter] + lines
    
    anim = FuncAnimation(
        fig,
        update,
        frames=len(points),
        interval=50,
        blit=False,
        repeat=True
    )
    
    plt.show()

# 主程序，進行動態相機數量的支持
def main():
    base_folder = r'./20250116/intrinsic'
    camera_folders = get_camera_folders(base_folder, CAMERA_CONFIG)
    
    # 開始標定相機
    calibrate_cameras(camera_folders)
    
    # 開始處理視頻
    points_3d, aruco_axis, cam_axes = process_videos(video_paths)
    
    if len(points_3d) == 0:
        print("未能從視頻中提取任何3D關鍵點。")
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
