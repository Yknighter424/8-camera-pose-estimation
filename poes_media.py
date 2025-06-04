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
    # 創建一個 pattern_size[0]*pattern_size[1] 行、3列的零矩陣，用於存儲標定板角點的世界坐標
    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    # 使用 np.mgrid 生成網格坐標，並重塑為 n×2 的矩陣，賦值給前兩列(x,y坐標)，z坐標保持為0
    objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
    objp = objp * square_size  # 轉換為實際尺寸
    
    # 存儲所有圖像的3D點和2D點
    objpoints = [] # 3D點
    imgpoints = [] # 2D點
    
    # 獲取所有校正圖片
    # 使用glob查找所有.jpeg格式的標定圖片
    images = glob.glob(os.path.join(images_folder, '*.jpeg'))
    # 如果沒有找到jpeg格式的圖片,則查找png格式的圖片
    if not images:
        images = glob.glob(os.path.join(images_folder, '*.png'))
    
    # 打印找到的圖片數量
    print(f"在 {images_folder} 中找到 {len(images)} 張圖片")
    
    # 遍歷每張標定圖片
    for image_filename in images:  # fname是file name的縮寫,表示圖片文件名
        # 讀取圖片
        img = cv2.imread(image_filename)
        # 將圖片轉換為灰度圖
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 在灰度圖中查找棋盤格角點
        # 使用cv2.findChessboardCorners()查找棋盤格角點
        # gray: 輸入的灰度圖像
        # pattern_size: 棋盤格內角點的數量(寬,高)
        # None: 可選的標誌參數
        # ret: 是否成功找到所有角點的布爾值
        # corners: 檢測到的角點坐標數組
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        # 如果成功找到所有角點
        if ret:
            # 設置亞像素精確化的終止條件
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # 對角點進行亞像素精確化
            # 使用cv2.cornerSubPix()对角点进行亚像素精确化
            # gray: 输入的灰度图像
            # corners: 初始的角点坐标(由findChessboardCorners得到)
            # (11,11): 搜索窗口大小,必须是奇数
            # (-1,-1): 死区大小,(-1,-1)表示没有死区
            # criteria: 迭代终止条件(最大迭代次数和精度)
            # 返回精确化后的角点坐标
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            # 將世界坐標系中的點加入到objpoints
            objpoints.append(objp)
            # 將圖像平面的點加入到imgpoints
            imgpoints.append(corners2)
            
            # 在圖片上繪製檢測到的棋盤格角點
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            # 顯示帶有角點的圖片(縮放到800x600)
            cv2.imshow('Corners', cv2.resize(img, (800,600)))
            # 等待500毫秒
            cv2.waitKey(500)
    
    # 關閉所有OpenCV視窗
    cv2.destroyAllWindows()
    
    # 執行相機標定
    # 使用cv2.calibrateCamera()執行相機標定
    # objpoints: 世界坐標系中的三維點
    # imgpoints: 對應的圖像平面二維點
    # gray.shape[::-1]: 圖像尺寸(寬,高)
    # 返回:
    # ret: 標定是否成功的標誌
    # mtx: 相機內參矩陣
    # dist: 畸變係數
    # rvecs: 旋轉向量
    # tvecs: 平移向量
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # 在OpenCV中，圖像的shape屬性返回格式為(高度,寬度)
    # 但是calibrateCamera函數需要的圖像尺寸格式為(寬度,高度)
    # 所以我們使用[::-1]來反轉shape元組的順序
    # 這樣就能將(高度,寬度)轉換為(寬度,高度)
    # 如果不進行轉換，標定結果會出現錯誤
    
    # 計算重投影誤差
    # 初始化平均誤差為0
    mean_error = 0
    # 對每組標定點進行重投影誤差計算
    for i in range(len(objpoints)):
        # 使用cv2.projectPoints將三維點投影到圖像平面
        # objpoints[i]: 當前幀的三維點
        # rvecs[i], tvecs[i]: 當前幀的旋轉和平移向量
        # mtx: 相機內參矩陣
        # dist: 畸變係數
        # 返回投影後的二維點坐標
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        # 計算投影點與實際檢測點之間的歐氏距離
        # cv2.NORM_L2: 使用L2範數(歐氏距離)
        # 將總誤差除以點的數量得到平均誤差
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        # 累加每幀的平均誤差
        mean_error += error
    
    # 輸出所有幀的平均重投影誤差
    print(f"平均重投影誤差: {mean_error/len(objpoints)}")
    
    # 返回標定得到的所有參數
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

'''def test_calibration(image_path, camera_params):
    """
    測試標定結果
    
    Args:
        image_path: 測試圖片路徑
        camera_params: 相機參數(mtx, dist)
    """
    # 從camera_params中解包相機矩陣和畸變係數
    mtx, dist = camera_params
    # 讀取測試圖片
    img = cv2.imread(image_path)
    # 獲取圖片的高度和寬度
    h, w = img.shape[:2]
    
    # 使用cv2.getOptimalNewCameraMatrix獲取優化後的相機矩陣和ROI區域
    # 參數1表示alpha值,用於控制視野大小,(w,h)指定輸出圖像大小
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    # 使用cv2.undistort對圖像進行校正
    # None表示不使用可選的R矩陣,newcameramtx用於優化輸出
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # 從ROI中獲取裁剪區域的座標和大小
    x, y, w, h = roi
    # 根據ROI對校正後的圖像進行裁剪
    dst = dst[y:y+h, x:x+w]
    
    # 顯示原始圖像,調整大小為800x600
    cv2.imshow('Original', cv2.resize(img, (800,600)))
    # 顯示校正後的圖像,調整大小為800x600
    cv2.imshow('Calibrated', cv2.resize(dst, (800,600)))
    # 等待按鍵輸入
    cv2.waitKey(0)
    # 關閉所有視窗
    cv2.destroyAllWindows()'''

if __name__ == "__main__":
 # 这段代码是相机标定的测试代码
 # while False 表示这段代码默认不会执行
 while False:
    # 设置三个相机标定图片的文件夹路径
    # cam1_left 是左相机的标定图片路径
    cam1_left = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\intrinsic\port_3left"
    # cam2_center 是中间相机的标定图片路径
    cam2_center = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\intrinsic\port_1_mid" 
    # cam3_right 是右相机的标定图片路径
    cam3_right = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\intrinsic\port_2_right"
    
    # 调用 calibrate_three_cameras 函数进行三相机标定
    # 函数返回保存标定参数的 npz 文件路径
    npz_path = calibrate_three_cameras(cam1_left, cam2_center, cam3_right)
    
    # 尝试加载生成的标定文件,验证文件是否正确保存
    try:
        # 加载 npz 文件中的标定数据
        calib_data = np.load(npz_path)
        print("\n成功加载标定文件,包含以下数据:")
        # 打印文件中包含的所有数据项名称
        print(calib_data.files)
    except Exception as e:
        # 如果加载出错,打印错误信息
        print(f"\n加载标定文件时出错: {e}")









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
    axis_coord = axis_coord * 500  # 放大坐标轴显示
    
    
    # 在三个相机图像中检测ArUco标记
    # corners: 检测到的标记角点坐标
    # ids: 检测到的标记ID
    # rejectedImgPoints: 被拒绝的候选标记
    corners_L, ids_L, rejectedImgPoints_L = aruco_detector.detectMarkers(img_L)
    corners_M, ids_M, rejectedImgPoints_M = aruco_detector.detectMarkers(img_M)
    corners_R, ids_R, rejectedImgPoints_R = aruco_detector.detectMarkers(img_R)

    # 对左相机图像进行亚像素角点精化处理
    if ids_L is not None and len(ids_L) > 0:  # 检查是否检测到ArUco标记
        # 将BGR彩色图像转换为灰度图,如果输入已经是灰度图则直接使用
        # 亚像素角点检测需要灰度图,因为:
        # 1. 灰度图只有一个通道,计算量更小,处理更快
        # 2. 角点检测主要依赖于图像的强度变化,而不是颜色信息
        # 3. 灰度值的梯度计算更简单且足以表达边缘和角点特征
        # cv2.cvtColor: OpenCV颜色空间转换函数
        # COLOR_BGR2GRAY: 从BGR转为灰度图的转换代码
        # len(img_L.shape) == 3: 检查图像是否有3个维度(高、宽、通道),判断是否为彩色图
        # 如果是彩色图则转换,否则保持原样
        gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY) if len(img_L.shape) == 3 else img_L
        # 设置亚像素精化的终止条件:
        # TERM_CRITERIA_EPS:精度达到0.001时停止
        # TERM_CRITERIA_MAX_ITER:最大迭代次数30次
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 对每个检测到的角点进行精化
        for i in range(len(corners_L)):
            # cornerSubPix函数使用3x3像素的邻域窗口来计算亚像素精度
            # 3x3窗口意味着以当前角点为中心,取周围8个相邻像素点
            # 通过分析这9个像素点的灰度值分布来精确定位角点位置
            refined_corners = cv2.cornerSubPix(gray_L, corners_L[i], (3, 3), (-1, -1), criteria)
            # 用精化后的坐标替换原始角点坐标
            corners_L[i] = refined_corners

    # 对中间相机图像进行相同的亚像素角点精化处理
    if ids_M is not None and len(ids_M) > 0:  # 检查中间相机是否检测到ArUco标记
        # 转换为灰度图
        gray_M = cv2.cvtColor(img_M, cv2.COLOR_BGR2GRAY) if len(img_M.shape) == 3 else img_M
        # 对每个检测到的角点进行精化
        for i in range(len(corners_M)):
            # 同样使用3x3像素窗口进行亚像素精化
            # 窗口中心是初始检测的角点位置
            corners_M[i] = cv2.cornerSubPix(gray_M, corners_M[i], (3, 3), (-1, -1), criteria)

    # 对右相机图像进行相同的亚像素角点精化处理
    if ids_R is not None and len(ids_R) > 0:  # 检查右相机是否检测到ArUco标记
        # 转换为灰度图
        gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY) if len(img_R.shape) == 3 else img_R
        # 对每个检测到的角点进行精化
        for i in range(len(corners_R)):
            # 同样使用3x3像素窗口进行亚像素精化
            # 通过分析局部灰度分布提高角点定位精度
            corners_R[i] = cv2.cornerSubPix(gray_R, corners_R[i], (3, 3), (-1, -1), criteria)
    
    # 初始化存储列表
    # img_coords_L: 存储左相机图像中检测到的ArUco标记角点的2D坐标 [图像平面坐标]
    # point_coords_L: 存储ArUco标记在标定板坐标系中的3D坐标 [世界坐标系] 
    # aruco_ps_L_2camL: 存储ArUco标记点在左相机坐标系中的3D位置 [通过PnP算法求解]
    img_coords_L, point_coords_L, aruco_ps_L_2camL = [], [], []
    img_coords_M, point_coords_M, aruco_ps_M_2camM = [], [], []
    img_coords_R, point_coords_R, aruco_ps_R_2camR = [], [], []

    # 处理左相机图像
    # 如果检测到标记
    # 检查左相机是否检测到ArUco标记
    # ids_L: 存储检测到的ArUco标记ID的数组
    # is not None: 确保ids_L不是空值
    if ids_L is not None:
        # 遍历每个检测到的标记
        # len(ids_L): 检测到的标记总数
        # range(): 生成从0到标记总数-1的序列
        # i: 当前遍历的标记索引
        for i in range(len(ids_L)):
            # 检查当前标记ID是否在标定板坐标字典中
            # ids_L[i]: 获取第i个标记的ID数组
            # ids_L[i][0]: 获取ID值(因为ID存储在1x1数组中)
            # board_coord.keys(): 获取标定板坐标字典中所有有效的ID
            # not in: 检查ID是否不在有效ID列表中
            if ids_L[i][0] not in board_coord.keys():
                continue  # 如果ID无效,跳过当前标记
            
            # 获取当前标记的角点坐标
            # corners_L: 存储所有标记角点坐标的数组
            # corners_L[i]: 获取第i个标记的角点数组
            # corners_L[i][0]: 获取实际的角点坐标(因为坐标存储在嵌套数组中)
            # tmp_marker: 临时变量,存储当前标记的4个角点坐标
            tmp_marker = corners_L[i][0]
            
            # OpenCV的绘图函数要求整数像素坐标
            # 因为图像是离散的像素网格,不能在小数位置绘制
            # tmp_marker是一个4x2的数组,存储了ArUco标记的4个角点坐标
            # tmp_marker[0] 是左上角点的坐标 [x,y]
            # tmp_marker[1] 是右上角点的坐标 [x,y] 
            # tmp_marker[2] 是右下角点的坐标 [x,y]
            # tmp_marker[3] 是左下角点的坐标 [x,y]
            # 所以tmp_marker[0][0]表示左上角的x坐标,tmp_marker[0][1]表示左上角的y坐标
            tmp_marker_tl = (int(tmp_marker[0][0]), int(tmp_marker[0][1]))  # 左上角 (x,y)
            tmp_marker_tr = (int(tmp_marker[1][0]), int(tmp_marker[1][1]))  # 右上角 (x,y)
            tmp_marker_br = (int(tmp_marker[2][0]), int(tmp_marker[2][1]))  # 右下角 (x,y)
            tmp_marker_bl = (int(tmp_marker[3][0]), int(tmp_marker[3][1]))  # 左下角 (x,y)
            # 在图像上标记角点
            # 在左上角(tl)画一个红色圆圈
            # cv2.circle参数: 图像, 圆心坐标, 半径, 颜色(BGR格式), 填充(-1表示填充)
            cv2.circle(img_L, tmp_marker_tl, 10, (0, 0, 255), -1)  # (0,0,255)是红色
            
            # 在右上角(tr)画一个绿色圆圈
            cv2.circle(img_L, tmp_marker_tr, 10, (0, 255, 0), -1)  # (0,255,0)是绿色
            
            # 在右下角(br)画一个蓝色圆圈
            cv2.circle(img_L, tmp_marker_br, 10, (255, 0, 0), -1)  # (255,0,0)是蓝色
            
            # 在左下角(bl)画一个橙色圆圈
            cv2.circle(img_L, tmp_marker_bl, 10, (0, 170, 255), -1)  # (0,170,255)是橙色
            
            # 在标记点附近添加ID文本
            # cv2.putText参数:
            # img_L: 目标图像
            # f"ID: {ids_L[i][0]}": 要显示的文本(ArUco标记的ID)
            # (int(...), int(...)): 文本位置(左上角点右上方10像素)
            # cv2.FONT_HERSHEY_SIMPLEX: 字体类型
            # 1: 字体大小
            # (0,0,255): 字体颜色(红色)
            # 1: 字体粗细
            # cv2.LINE_AA: 抗锯齿类型
            # 在标记点附近添加ID文本标注
            cv2.putText(img_L, f"ID: {ids_L[i][0]}", (int(tmp_marker_tl[0] + 10), int(tmp_marker_tl[1] + 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            
            # 构建角点坐标数组 - 将4个角点坐标组成一个数组
            img_coord = np.array([tmp_marker[j] for j in range(4)])  # 4x2数组,每行是一个角点坐标[x,y]
            img_coords_L.append(np.squeeze(img_coord))  # 添加到左相机的角点坐标列表中
            
            # 创建3D坐标 - 将2D板坐标扩展为3D坐标(z=0)
            # tem_coord是ArUco标记在标定板坐标系中的3D坐标
            # board_coord[ids_L[i][0]]获取当前ArUco标记ID对应的2D坐标(x,y)
            # np.zeros(...) 创建一个全0数组作为z坐标
            # np.hstack()将x,y坐标和z坐标水平拼接,得到(x,y,0)形式的3D坐标
            # 最终tem_coord是一个4x3的数组,存储标记4个角点的3D坐标
            tem_coord = np.hstack((board_coord[ids_L[i][0]], np.zeros(len(board_coord[ids_L[i][0]]))[:,None]))  # 添加z坐标列
            point_coords_L.append(tem_coord)  # 添加到左相机的3D坐标列表中
            '''标定板的角点顺序是固定的：
            索引0：左上角 [0,0]
            索引1：右上角 [0,1]
            索引2：右下角 [1,1]
            索引3：左下角 [1,0]
            '''
            # 解决PnP(Perspective-n-Point)问题 - 计算相机姿态
            # image_C_L: 左相机图像坐标
            # np.ascontiguousarray(): 将数组转换为内存连续的数组,提高计算效率
            # img_coord: 4x2的数组,存储4个角点的图像坐标
            # [:,:2]: 取所有行和前2列,即x,y坐标
            # reshape((-1,1,2)): 重塑为(-1,1,2)形状,符合OpenCV要求
            # -1表示自动计算该维度大小,这里是4
            
            image_C_L = np.ascontiguousarray(img_coord[:,:2]).reshape((-1,1,2))
            # 将图像坐标转换为OpenCV所需的格式
            # np.ascontiguousarray(): 确保数组在内存中连续存储,提高计算效率
            # img_coord[:,:2]: 取所有角点的x,y坐标(不包含z坐标)
            # reshape((-1,1,2)): 重塑数组维度为(n,1,2),其中n是角点数量
            # -1表示自动计算该维度的大小(这里是4个角点)
            # 最终得到的image_C_L形状为(4,1,2),存储4个角点的图像坐标
            
            # ret_L: 求解是否成功的布尔值
            # rvec_L: 旋转向量,描述ArUco标记相对相机的旋转
            # tvec_L: 平移向量,描述ArUco标记相对相机的平移
            # tem_coord: 标记在标定板坐标系中的3D坐标(4x3)
            # mtx1: 左相机内参矩阵
            # dist1: 左相机畸变系数
            ret_L, rvec_L, tvec_L = cv2.solvePnP(tem_coord, image_C_L, mtx1, dist1)
            
            # R_aruco2camL: 从ArUco标记坐标系到左相机坐标系的旋转矩阵(3x3)
            # _: 忽略返回的雅可比矩阵
            # cv2.Rodrigues(): 将旋转向量(3x1)转换为旋转矩阵(3x3)
            R_aruco2camL, _ = cv2.Rodrigues(rvec_L)
            
            # t_aruco2camL: 从ArUco标记坐标系到左相机坐标系的平移向量(3x1)
            t_aruco2camL = tvec_L
            # 计算ArUco标记点在相机坐标系中的3D位置
            aruco_p_L_2camL = np.dot(R_aruco2camL, tem_coord.T).T + t_aruco2camL.T  # 坐标变换:R*X + t
            aruco_ps_L_2camL.append(aruco_p_L_2camL)  # 添加到左相机的ArUco点3D位置列表中

    # 处理中间相机图像
    # 如果中间相机检测到了ArUco标记
    if ids_M is not None:
        # 遍历每个检测到的标记
        for i in range(len(ids_M)):
            # 如果检测到的标记ID不在预定义的board_coord中,跳过该标记
            if ids_M[i][0] not in board_coord.keys():
                continue
            # 获取当前标记的四个角点坐标
            tmp_marker = corners_M[i][0]
            
            # 将角点坐标转换为整数,便于在图像上绘制
            tmp_marker_tl = (int(tmp_marker[0][0]), int(tmp_marker[0][1]))  # 左上角坐标
            tmp_marker_tr = (int(tmp_marker[1][0]), int(tmp_marker[1][1]))  # 右上角坐标
            tmp_marker_br = (int(tmp_marker[2][0]), int(tmp_marker[2][1]))  # 右下角坐标
            tmp_marker_bl = (int(tmp_marker[3][0]), int(tmp_marker[3][1]))  # 左下角坐标
            
            # 在图像上用不同颜色的圆圈标记四个角点
            cv2.circle(img_M, tmp_marker_tl, 10, (0, 0, 255), -1)    # 左上角红色
            cv2.circle(img_M, tmp_marker_tr, 10, (0, 255, 0), -1)    # 右上角绿色
            cv2.circle(img_M, tmp_marker_br, 10, (255, 0, 0), -1)    # 右下角蓝色
            cv2.circle(img_M, tmp_marker_bl, 10, (0, 170, 255), -1)  # 左下角橙色
            # 在标记左上角附近添加ID文本标注
            cv2.putText(img_M, f"ID: {ids_M[i][0]}", (int(tmp_marker_tl[0] + 10), int(tmp_marker_tl[1] + 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            
            # 将四个角点坐标组成一个数组
            # 构建角点坐标数组 - 将4个角点坐标组成一个数组
            # j是循环变量,从0到3,用于访问tmp_marker中的4个角点坐标
            # tmp_marker[j]获取第j个角点的[x,y]坐标
            # range(4)生成[0,1,2,3],对应左上、右上、右下、左下四个角点
            img_coord = np.array([tmp_marker[j] for j in range(4)])  # 4x2数组,每行是一个角点坐标[x,y]
            img_coords_M.append(np.squeeze(img_coord))  # 添加到中间相机的角点坐标列表中
            
            # 将2D板坐标扩展为3D坐标(添加z=0)
            # 将2D板坐标扩展为3D坐标(z=0)
            # board_coord[ids_M[i][0]] 获取当前标记ID对应的2D坐标(x,y)
            # np.zeros(...) 创建一个全0数组作为z坐标
            # np.hstack() 将x,y坐标和z坐标水平拼接,得到3D坐标(x,y,0)
            
            # 创建3D坐标 - 将2D板坐标扩展为3D坐标(z=0)
            # tem_coord: 临时坐标变量
            # np.hstack(): 水平堆叠数组
            # board_coord[ids_M[i][0]]: 从board_coord字典中获取当前标记ID对应的2D坐标
            # np.zeros(): 创建全0数组作为z坐标
            # len(): 获取数组长度
            # [:,None]: 增加新的维度,使数组变为列向量
            tem_coord = np.hstack((board_coord[ids_M[i][0]], np.zeros(len(board_coord[ids_M[i][0]]))[:,None]))  # 添加z坐标列
            
            # point_coords_M: 存储中间相机检测到的所有3D坐标的列表
            # append(): 将新的坐标添加到列表末尾
            point_coords_M.append(tem_coord)  # 添加到中间相机的3D坐标列表中
            
            # 解决PnP问题,计算相机姿态
            image_C_M = np.ascontiguousarray(img_coord[:,:2]).reshape((-1,1,2))  # 重塑图像坐标为OpenCV需要的格式
            ret_M, rvec_M, tvec_M = cv2.solvePnP(tem_coord, image_C_M, mtx2, dist2)  # 求解相机位姿
            R_aruco2camM, _ = cv2.Rodrigues(rvec_M)  # 将旋转向量转换为旋转矩阵
            t_aruco2camM = tvec_M  # 获取平移向量
            
            # 计算ArUco标记点在中间相机坐标系中的3D位置
            aruco_p_M_2camM = np.dot(R_aruco2camM, tem_coord.T).T + t_aruco2camM.T  # 坐标变换:R*X + t
            aruco_ps_M_2camM.append(aruco_p_M_2camM)  # 添加到中间相机的ArUco点3D位置列表

    # 如果右相机检测到了ArUco标记
    if ids_R is not None:
        # 遍历每个检测到的标记
        for i in range(len(ids_R)):
            # 如果检测到的标记ID不在预定义的board_coord中,跳过该标记
            if ids_R[i][0] not in board_coord.keys():
                continue
            # 获取当前标记的四个角点坐标
            tmp_marker = corners_R[i][0]
            
            # 将角点坐标转换为整数,便于在图像上绘制
            tmp_marker_tl = (int(tmp_marker[0][0]), int(tmp_marker[0][1]))  # 左上角坐标
            tmp_marker_tr = (int(tmp_marker[1][0]), int(tmp_marker[1][1]))  # 右上角坐标
            tmp_marker_br = (int(tmp_marker[2][0]), int(tmp_marker[2][1]))  # 右下角坐标
            tmp_marker_bl = (int(tmp_marker[3][0]), int(tmp_marker[3][1]))  # 左下角坐标
            
            # 在图像上用不同颜色的圆圈标记四个角点
            cv2.circle(img_R, tmp_marker_tl, 10, (0, 0, 255), -1)    # 左上角红色
            cv2.circle(img_R, tmp_marker_tr, 10, (0, 255, 0), -1)    # 右上角绿色
            cv2.circle(img_R, tmp_marker_br, 10, (255, 0, 0), -1)    # 右下角蓝色
            cv2.circle(img_R, tmp_marker_bl, 10, (0, 170, 255), -1)  # 左下角橙色
            # 在标记左上角附近添加ID文本标注
            cv2.putText(img_R, f"ID: {ids_R[i][0]}", (int(tmp_marker_tl[0] + 10), int(tmp_marker_tl[1] + 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            
            # 将四个角点坐标组成一个数组
            img_coord = np.array([tmp_marker[j] for j in range(4)])
            img_coords_R.append(np.squeeze(img_coord))  # 添加到右相机的角点坐标列表
            
            # 将2D板坐标扩展为3D坐标(添加z=0)
            # 将2D板坐标扩展为3D坐标:
            # board_coord[ids_R[i][0]]: 从board_coord字典中获取当前标记ID对应的2D坐标
            # np.zeros(len(board_coord[ids_R[i][0]])): 创建与2D坐标等长的全0数组作为z坐标
            # [:,None]: 将z坐标数组增加一个维度,变成列向量形状
            # np.hstack(): 水平堆叠2D坐标和z坐标,形成完整的3D坐标
            tem_coord = np.hstack((board_coord[ids_R[i][0]], np.zeros(len(board_coord[ids_R[i][0]]))[:,None]))
            point_coords_R.append(tem_coord)  # 添加到右相机的3D坐标列表
            
            # 解决PnP问题,计算相机姿态
            image_C_R = np.ascontiguousarray(img_coord[:,:2]).reshape((-1,1,2))  # 重塑图像坐标为OpenCV需要的格式
            ret_R, rvec_R, tvec_R = cv2.solvePnP(tem_coord, image_C_R, mtx3, dist3)  # 求解相机位姿
            R_aruco2camR, _ = cv2.Rodrigues(rvec_R)  # 将旋转向量转换为旋转矩阵
            t_aruco2camR = tvec_R  # 获取平移向量
            
            # 计算ArUco标记点在右相机坐标系中的3D位置
            aruco_p_R_2camR = np.dot(R_aruco2camR, tem_coord.T).T + t_aruco2camR.T  # 坐标变换:R*X + t
            aruco_ps_R_2camR.append(aruco_p_R_2camR)  # 添加到右相机的ArUco点3D位置列表

    # 合并检测到的点
    # 合并左相机检测到的所有ArUco角点的2D坐标,如果没有检测到则返回空数组
    img_coords_L = np.concatenate(img_coords_L, axis=0) if img_coords_L else np.array([])
    # 合并中间相机检测到的所有ArUco角点的2D坐标,如果没有检测到则返回空数组
    img_coords_M = np.concatenate(img_coords_M, axis=0) if img_coords_M else np.array([])
    # 合并右相机检测到的所有ArUco角点的2D坐标,如果没有检测到则返回空数组
    img_coords_R = np.concatenate(img_coords_R, axis=0) if img_coords_R else np.array([])
    
    # 为左相机的2D坐标添加齐次坐标(添加值为1的第三维),用于后续投影变换计算
    img_coords_L = np.hstack((img_coords_L, np.ones((len(img_coords_L), 1)))) if len(img_coords_L) > 0 else np.array([])
    # 为中间相机的2D坐标添加齐次坐标(添加值为1的第三维),用于后续投影变换计算
    img_coords_M = np.hstack((img_coords_M, np.ones((len(img_coords_M), 1)))) if len(img_coords_M) > 0 else np.array([])
    # 为右相机的2D坐标添加齐次坐标(添加值为1的第三维),用于后续投影变换计算
    img_coords_R = np.hstack((img_coords_R, np.ones((len(img_coords_R), 1)))) if len(img_coords_R) > 0 else np.array([])
    
    # 合并左相机检测到的所有ArUco标记的3D世界坐标,如果没有检测到则返回空数组
    point_coords_L = np.concatenate(point_coords_L, axis=0) if point_coords_L else np.array([])
    # 合并中间相机检测到的所有ArUco标记的3D世界坐标,如果没有检测到则返回空数组
    point_coords_M = np.concatenate(point_coords_M, axis=0) if point_coords_M else np.array([])
    # 合并右相机检测到的所有ArUco标记的3D世界坐标,如果没有检测到则返回空数组
    point_coords_R = np.concatenate(point_coords_R, axis=0) if point_coords_R else np.array([])
    
    # 合并左相机坐标系下所有ArUco标记点的3D位置,如果没有检测到则返回空数组
    aruco_ps_L_2camL = np.concatenate(aruco_ps_L_2camL, axis=0) if aruco_ps_L_2camL else np.array([])
    # 合并中间相机坐标系下所有ArUco标记点的3D位置,如果没有检测到则返回空数组
    aruco_ps_M_2camM = np.concatenate(aruco_ps_M_2camM, axis=0) if aruco_ps_M_2camM else np.array([])
    # 合并右相机坐标系下所有ArUco标记点的3D位置,如果没有检测到则返回空数组
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
    # 初始化左相機的聚類數據結構
    clusters_L = defaultdict(list)  # 存儲每個聚類的點
    cluster_ids_L = []  # 存儲聚類ID列表
    cluster_indx_L = {}  # 存儲每個聚類包含的點的索引
    
    # 遍歷左相機檢測到的每個ArUco角點
    # 聚類邏輯:
    # 1. 遍歷每個檢測到的ArUco角點
    # 2. 計算該點與已有聚類中所有點的距離
    # 3. 如果與某個聚類中的任一點距離小於棋盤格最大邊長,則歸入該聚類
    # 4. 如果與所有已有聚類的點距離都大於閾值,則創建新聚類
    # 這樣可以將空間位置相近的ArUco角點歸為一組,代表同一個標定板
    # 遍歷左相機檢測到的每個ArUco角點及其索引
    for i, point in enumerate(aruco_ps_L_2camL):
        # 初始化標記,假設當前點需要創建新聚類
        new_cluster = True  
        
        # 遍歷所有現有的聚類組
        for cluster_id, cluster_points in clusters_L.items():
            # 遍歷當前聚類中的所有點
            for cluster_point in cluster_points:
                # 計算當前點與聚類中點的歐氏距離
                # board_coord[5][3,1]表示標定板的最大邊長,作為聚類閾值
                # 如果距離小於閾值,說明兩點屬於同一標定板
                
                # 计算当前点与聚类中点的欧氏距离
                # board_coord[5] 表示第6个标定板的坐标数据
                # board_coord[5][3,1] 
                # 这个值代表标定板的最大边长,用作聚类阈值
                # 如果两点间距离小于标定板最大边长,说明它们属于同一个标定板
                # np.linalg.norm计算两点间的欧氏距离
                # point是当前遍历到的点,cluster_point是已有聚类中的点
                if np.linalg.norm(point - cluster_point) <= board_coord[5][3,1]:
                    # 將當前點加入到已有聚類中
                    clusters_L[cluster_id].append(point)
                    # 記錄該點在原始數組中的索引,用於後續處理
                    cluster_indx_L[cluster_id].append(i)
                    # 標記已找到所屬聚類,不需要創建新聚類
                    new_cluster = False
                    # 找到所屬聚類後跳出內層循環
                    break
            
            # 如果已經將點分配到某個聚類,跳出外層循環
            # 避免重複將同一點分配給多個聚類
            if not new_cluster:
                break
        # 如果點不屬於任何現有聚類,則創建新聚類
        if new_cluster:
            cluster_id = len(cluster_ids_L)  # 生成新的聚類ID
            cluster_ids_L.append(cluster_id)  # 添加新聚類ID
            clusters_L[cluster_id] = [point]  # 創建新聚類並添加點
            cluster_indx_L[cluster_id] = [i]  # 記錄點的索引

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

    # 右相机聚类    #一个默认字典
    clusters_R = defaultdict(list)
    #一个列表，存储所有聚类的ID
    cluster_ids_R = []
    #一个字典，key是聚类的ID，value是该聚类中所有点的原始数组中的索引
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

    # 获取最大聚类的索引 - 对每个相机找到包含最多点的聚类的索引列表
    cluster_max_indxs_L = max(cluster_indx_L.values(), key=len) if cluster_indx_L else [] # 如果左相机有聚类,获取最大聚类的索引,否则返回空列表
    cluster_max_indxs_M = max(cluster_indx_M.values(), key=len) if cluster_indx_M else [] # 同上,处理中间相机
    cluster_max_indxs_R = max(cluster_indx_R.values(), key=len) if cluster_indx_R else [] # 同上,处理右相机

    # 对每个相机的最大聚类索引进行排序,确保点的顺序一致
    cluster_max_indxs_L.sort() # 对左相机最大聚类的索引排序
    cluster_max_indxs_M.sort() # 对中间相机最大聚类的索引排序 
    cluster_max_indxs_R.sort() # 对右相机最大聚类的索引排序

    # 使用最大聚类的索引选择对应的图像坐标点
    img_coords_L = img_coords_L[cluster_max_indxs_L] # 选择左相机最大聚类对应的图像坐标
    img_coords_M = img_coords_M[cluster_max_indxs_M] # 选择中间相机最大聚类对应的图像坐标
    img_coords_R = img_coords_R[cluster_max_indxs_R] # 选择右相机最大聚类对应的图像坐标

    # 使用最大聚类的索引选择对应的3D点坐标
    point_coords_L = point_coords_L[cluster_max_indxs_L] # 选择左相机最大聚类对应的3D坐标
    point_coords_M = point_coords_M[cluster_max_indxs_M] # 选择中间相机最大聚类对应的3D坐标
    point_coords_R = point_coords_R[cluster_max_indxs_R] # 选择右相机最大聚类对应的3D坐标

    # 使用选定的点解决每个相机的PnP问题,计算相机姿态
    # 左相机PnP求解
    image_C_L = np.ascontiguousarray(img_coords_L[:,:2]).reshape((-1,1,2)) # 重塑图像坐标为OpenCV需要的格式
    ret_L, rvec_L, tvec_L = cv2.solvePnP(point_coords_L, image_C_L, mtx1, dist1) # 求解左相机位姿
    
    # 中间相机PnP求解
    image_C_M = np.ascontiguousarray(img_coords_M[:,:2]).reshape((-1,1,2)) # 重塑图像坐标为OpenCV需要的格式
    ret_M, rvec_M, tvec_M = cv2.solvePnP(point_coords_M, image_C_M, mtx2, dist2) # 求解中间相机位姿
    
    # 右相机PnP求解
    image_C_R = np.ascontiguousarray(img_coords_R[:,:2]).reshape((-1,1,2)) # 重塑图像坐标为OpenCV需要的格式
    ret_R, rvec_R, tvec_R = cv2.solvePnP(point_coords_R, image_C_R, mtx3, dist3) # 求解右相机位姿

    # 在每个相机图像上绘制坐标轴,可视化相机姿态
    # 左相机坐标轴绘制
    image_points, _ = cv2.projectPoints(axis_coord, rvec_L, tvec_L, mtx1, dist1) # 将3D坐标轴投影到左相机图像平面
    image_points = image_points.reshape(-1, 2).astype(np.int16) # 转换为整数坐标
    cv2.line(img_L, (image_points[0,0], image_points[0,1]), (image_points[1,0], image_points[1,1]), (0,0,255), 5) # 绘制X轴(红色)
    cv2.line(img_L, (image_points[0,0], image_points[0,1]), (image_points[2,0], image_points[2,1]), (0,255,0), 5) # 绘制Y轴(绿色)
    cv2.line(img_L, (image_points[0,0], image_points[0,1]), (image_points[3,0], image_points[3,1]), (255,0,0), 5) # 绘制Z轴(蓝色)

    # 中间相机坐标轴绘制
    image_points, _ = cv2.projectPoints(axis_coord, rvec_M, tvec_M, mtx2, dist2) # 将3D坐标轴投影到中间相机图像平面
    image_points = image_points.reshape(-1, 2).astype(np.int16) # 转换为整数坐标
    cv2.line(img_M, (image_points[0,0], image_points[0,1]), (image_points[1,0], image_points[1,1]), (0,0,255), 5) # 绘制X轴(红色)
    cv2.line(img_M, (image_points[0,0], image_points[0,1]), (image_points[2,0], image_points[2,1]), (0,255,0), 5) # 绘制Y轴(绿色)
    cv2.line(img_M, (image_points[0,0], image_points[0,1]), (image_points[3,0], image_points[3,1]), (255,0,0), 5) # 绘制Z轴(蓝色)

    # 右相机坐标轴绘制
    image_points, _ = cv2.projectPoints(axis_coord, rvec_R, tvec_R, mtx3, dist3) # 将3D坐标轴投影到右相机图像平面
    # image_points: 从cv2.projectPoints返回的投影点坐标数组
    # reshape(-1, 2): 将数组重塑为Nx2的形状,其中N是点的数量,-1表示自动计算行数
    # astype(np.int16): 将浮点数坐标转换为16位整数类型,便于在图像上绘制
    image_points = image_points.reshape(-1, 2).astype(np.int16) # 转换为整数坐标
    cv2.line(img_R, (image_points[0,0], image_points[0,1]), (image_points[1,0], image_points[1,1]), (0,0,255), 5) # 绘制X轴(红色)
    cv2.line(img_R, (image_points[0,0], image_points[0,1]), (image_points[2,0], image_points[2,1]), (0,255,0), 5) # 绘制Y轴(绿色)
    cv2.line(img_R, (image_points[0,0], image_points[0,1]), (image_points[3,0], image_points[3,1]), (255,0,0), 5) # 绘制Z轴(蓝色)

    # 计算最终的旋转矩阵和变换矩阵
    R_aruco2camL, _ = cv2.Rodrigues(rvec_L) # 将左相机旋转向量转换为旋转矩阵
    t_aruco2camL = tvec_L # 左相机平移向量
    R_aruco2camM, _ = cv2.Rodrigues(rvec_M) # 将中间相机旋转向量转换为旋转矩阵
    t_aruco2camM = tvec_M # 中间相机平移向量
    R_aruco2camR, _ = cv2.Rodrigues(rvec_R) # 将右相机旋转向量转换为旋转矩阵
    t_aruco2camR = tvec_R # 右相机平移向量

    # 计算从相机坐标系到ArUco坐标系的变换(逆变换)
    R_camL2aruco = R_aruco2camL.T # 左相机旋转矩阵的转置
    t_camL2aruco = -R_aruco2camL.T @ t_aruco2camL # 左相机平移向量的逆变换
    R_camM2aruco = R_aruco2camM.T # 中间相机旋转矩阵的转置
    t_camM2aruco = -R_aruco2camM.T @ t_aruco2camM # 中间相机平移向量的逆变换
    R_camR2aruco = R_aruco2camR.T # 右相机旋转矩阵的转置
    t_camR2aruco = -R_aruco2camR.T @ t_aruco2camR # 右相机平移向量的逆变换

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
        points_3d: 3D點坐標 (N, 3)  # N個點的三維坐標
        points_2d: 2D點坐標 (N, 2)  # N個點的二維圖像坐標
        P: 投影矩陣 (3, 4)  # 3x4的投影矩陣,用於將3D點投影到2D平面
        mtx: 相機內參矩陣  # 3x3的相機內參矩陣,包含焦距和主點坐標
        dist: 畸變係數  # 相機鏡頭的畸變係數,用於校正畸變
    Returns:
        rmse: 均方根誤差  # 重投影誤差的均方根值
    """
    # 將3D點轉換到相機坐標系
    # np.hstack(): 水平堆疊數組
    # np.ones(): 創建全1數組
    # points_3d.shape[0]: 獲取點的數量
    # points_3d_homo: 齊次坐標形式的3D點 (N, 4)
    points_3d_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # np.dot(): 矩陣乘法
    # P.T: 投影矩陣的轉置
    # points_cam: 相機坐標系下的點坐標 (N, 3)
    points_cam = np.dot(points_3d_homo, P.T)
    
    # 進行投影
    # [:, :2]: 取前兩列(x,y坐標)
    # [:, 2:]: 取第三列(z坐標)用於歸一化
    # points_proj: 投影後的2D點坐標 (N, 2)
    points_proj = points_cam[:, :2] / points_cam[:, 2:]
    
    # 應用相機畸變
    # cv2.projectPoints(): OpenCV的投影函數
    # P[:, :3]: 取投影矩陣的旋轉部分
    # P[:, 3]: 取投影矩陣的平移部分
    # reshape(-1, 2): 重塑為Nx2的形狀
    # points_proj_dist: 考慮畸變後的投影點坐標 (N, 2)
    points_proj_dist = cv2.projectPoints(points_3d, P[:, :3], P[:, 3], mtx, dist)[0].reshape(-1, 2)
    
    # 計算誤差
    # np.linalg.norm(): 計算向量範數(歐氏距離)
    # axis=1: 沿第1軸計算範數,即對每個點的x和y坐標分量計算歐氏距離
    # 例如對於點集:
    # [[x1,y1],
    #  [x2,y2],
    #  [x3,y3]]
    # axis=1會分別計算sqrt(x1^2+y1^2), sqrt(x2^2+y2^2), sqrt(x3^2+y3^2)
    # error: 每個點的投影誤差 (N,)
    error = np.linalg.norm(points_2d - points_proj_dist, axis=1)
    
    # np.mean(): 計算平均值
    # np.sqrt(): 計算平方根
    # rmse: 均方根誤差(標量)
    rmse = np.sqrt(np.mean(error ** 2))
    
    return rmse

def process_frame(detector,  # 姿態檢測器對象
                 frame_left,  # 左相機的幀圖像
                 frame_middle,  # 中間相機的幀圖像 
                 frame_right,  # 右相機的幀圖像
                 img_L,  # 左相機的輸出圖像
                 img_M,  # 中間相機的輸出圖像
                 img_R,  # 右相機的輸出圖像
                 cams_params,  # 相機參數元組(mtx1,dist1,mtx2,dist2,mtx3,dist3)
                 cam_P):  # 相機投影矩陣元組(R_L,t_L,R_M,t_M,R_R,t_R)
    """
    處理三個相機的單幀圖像並計算重投影誤差
    """
    # mp: MediaPipe庫
    # Image: MediaPipe的圖像類
    # image_format: 指定圖像格式
    # SRGB: sRGB顏色空間
    # data: 圖像數據
    # cvtColor: OpenCV的顏色空間轉換函數
    # BGR2RGB: 從BGR轉換到RGB顏色空間
    # frame: 輸入的相機幀
    mp_images = [
        mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for frame in [frame_left, frame_middle, frame_right]  # 對三個相機幀進行循環
    ]
    
    # detector.detect(): 使用檢測器進行姿態檢測
    # img: MediaPipe圖像對象
    detection_results = [detector.detect(img) for img in mp_images]  # 對每個圖像進行檢測
    
    # poses: 存儲檢測到的姿態關鍵點列表
    # frames: 原始相機幀列表
    # cam_names: 相機名稱列表
    poses = []  # 初始化空列表存儲姿態
    frames = [frame_left, frame_middle, frame_right]  # 相機幀列表
    cam_names = ["左側", "中間", "右側"]  # 相機名稱列表
    
    # detection_failed: 檢測失敗標誌
    # failed_cameras: 檢測失敗的相機列表
    detection_failed = False  # 初始化檢測失敗標誌為False
    failed_cameras = []  # 初始化空列表存儲失敗的相機
    
    # enumerate(): 枚舉函數,返回索引和值
    # result: 每個相機的檢測結果
    for i, result in enumerate(detection_results):
        # pose_landmarks: 姿態關鍵點結果
        # len(): 計算長度
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            # np.array(): 創建NumPy數組
            # landmark.x/y: 關鍵點的x/y坐標(0-1範圍)
            # shape[1]/[0]: 圖像的寬度/高度
            pose = np.array([[landmark.x * frames[i].shape[1], landmark.y * frames[i].shape[0]]  # np.array: 創建NumPy數組
                            for landmark in result.pose_landmarks[0]])  # landmark: 關鍵點對象
                                                                      # x,y: 關鍵點的歸一化坐標(0-1)
                                                                      # frames[i]: 當前幀圖像
                                                                      # shape[1]: 圖像寬度
                                                                      # shape[0]: 圖像高度
                                                                      # *: 乘法運算符,將歸一化坐標轉換為像素坐標
            poses.append(pose)  # 添加到姿態列表
        else:
            detection_failed = True  # 設置檢測失敗標誌
            failed_cameras.append(cam_names[i])  # 添加失敗的相機名稱
    
    # join(): 連接字符串
    # f-string: 格式化字符串
    if detection_failed:
        print(f"以下相機未檢測到人體姿勢: {', '.join(failed_cameras)}")  # 打印失敗信息
        # np.zeros(): 創建全零數組
        # (33,3): 33個關鍵點,每個點3個坐標
        # (0,0,0): 三個相機的零誤差值
        return np.zeros((33, 3)), img_L, img_M, img_R, (0, 0, 0)
    
    # 解包相機參數和投影矩陣
    (mtx1, dist1, mtx2, dist2, mtx3, dist3) = cams_params  # 相機內參和畸變係數
    (R_aruco2camL, t_aruco2camL, R_aruco2camM, t_aruco2camM, R_aruco2camR, t_aruco2camR) = cam_P  # 旋轉矩陣和平移向量
    
    # triangulate_points_three_cameras(): 三相機三角測量函數
    # np.hstack(): 水平堆疊數組
    points_3d = triangulate_points_three_cameras(
        poses[0], poses[1], poses[2],  # 三個相機的2D關鍵點
        mtx1, dist1, mtx2, dist2, mtx3, dist3,  # 相機參數
        np.hstack((R_aruco2camL, t_aruco2camL)),  # 左相機投影矩陣
        np.hstack((R_aruco2camM, t_aruco2camM)),  # 中間相機投影矩陣
        np.hstack((R_aruco2camR, t_aruco2camR))  # 右相機投影矩陣
    )
    
    # 在圖像上標註關鍵點
    images = [img_L, img_M, img_R]  # 輸出圖像列表
    for i, pose in enumerate(poses):  # 遍歷每個相機的姿態
        for mark_i in range(33):  # 遍歷33個關鍵點
            # int(): 轉換為整數
            mark_coord = (int(pose[mark_i,0]), int(pose[mark_i,1]))  # 關鍵點坐標
            # cv2.circle(): 繪製圓圈
            # 10: 圓的半徑
            # (0,0,255): 紅色(BGR格式)
            # -1: 填充圓
            cv2.circle(images[i], mark_coord, 10, (0, 0, 255), -1)  # 繪製紅色圓圈
    
    # calculate_reprojection_error(): 計算重投影誤差函數
    rmse_left = calculate_reprojection_error(
        points_3d, poses[0],  # 3D點和左相機2D點
        np.hstack((R_aruco2camL, t_aruco2camL)),  # 左相機投影矩陣
        mtx1, dist1  # 左相機參數
    )
    
    rmse_middle = calculate_reprojection_error(
        points_3d, poses[1],  # 3D點和中間相機2D點
        np.hstack((R_aruco2camM, t_aruco2camM)),  # 中間相機投影矩陣
        mtx2, dist2  # 中間相機參數
    )
    
    rmse_right = calculate_reprojection_error(
        points_3d, poses[2],  # 3D點和右相機2D點
        np.hstack((R_aruco2camR, t_aruco2camR)),  # 使用np.hstack()將旋轉矩陣和平移向量水平拼接成投影矩陣
                                                  # R_aruco2camR是3x3的旋轉矩陣
                                                  # t_aruco2camR是3x1的平移向量
                                                  # 拼接後得到3x4的投影矩陣
        mtx3, dist3  # mtx3是右相機的3x3內參矩陣,包含焦距和主點坐標
                     # dist3是右相機的畸變係數,用於校正鏡頭畸變
    )
    
    # return: 返回值
    # points_3d: 三維重建點雲
    # img_L,img_M,img_R: 標註後的圖像
    # (rmse_left,rmse_middle,rmse_right): 三個相機的重投影誤差
    return points_3d, img_L, img_M, img_R, (rmse_left, rmse_middle, rmse_right)

def triangulate_points_three_cameras(points_left, points_middle, points_right,
                                   mtx1, dist1, mtx2, dist2, mtx3, dist3,
                                   P1, P2, P3):
    """
    使用三個相機進行三角測量
    
    數學推導:
    對於空間中的一個3D點X=[X,Y,Z,1]^T,其在相機i中的投影點為x_i=[u_i,v_i]^T
    根據針孔相機模型:
    s_i * [u_i,v_i,1]^T = P_i * [X,Y,Z,1]^T
    其中P_i為第i個相機的投影矩陣,s_i為尺度因子
    
    消除尺度因子可得線性方程組:
    u_i * (P_i^3 * X) - (P_i^1 * X) = 0
    v_i * (P_i^3 * X) - (P_i^2 * X) = 0
    其中P_i^j表示P_i的第j行
    
    將三個相機的6個方程組合可得:
    AX = 0
    其中A為6x4矩陣,X為待求的3D點坐標
    
    使用SVD分解求解最小二乘解:
    A = UΣV^T
    X為V的最後一列(對應最小奇異值)
    
    Args:
        points_left/middle/right: 三個相機的2D點
        mtx1/2/3, dist1/2/3: 相機內參和畸變係數
        P1/2/3: 投影矩陣
    Returns:
        points_3d: 三維點
    """
    points_3d = []
    
    # 對每個關鍵點進行三角測量
    # 同時遍歷三個相機的對應點坐標
    # points_left/middle/right: 每個相機的2D點坐標列表
    # zip(): 將三個列表打包成元組,每次迭代返回三個相機的對應點
    for pt_left, pt_middle, pt_right in zip(points_left, points_middle, points_right):
        # 對左相機點進行去畸變
        # pt_left.reshape(1,1,2): 將點坐標重塑為(1,1,2)形狀,符合OpenCV要求
        # mtx1: 左相機內參矩陣,包含焦距(fx,fy)和主點坐標(cx,cy)
        # dist1: 左相機畸變係數[k1,k2,p1,p2,k3]
        # 數學公式: x' = x(1 + k1r^2 + k2r^4 + k3r^6) + 2p1xy + p2(r^2 + 2x^2)
        #          y' = y(1 + k1r^2 + k2r^4 + k3r^6) + p1(r^2 + 2y^2) + 2p2xy
        # 其中(x,y)為歸一化坐標,(x',y')為去畸變後的歸一化坐標,r^2 = x^2 + y^2
        pt_left_undist = cv2.undistortPoints(pt_left.reshape(1, 1, 2), mtx1, dist1)
        
        # 對中間相機點進行去畸變,原理同上
        # mtx2: 中間相機內參矩陣
        # dist2: 中間相機畸變係數
        pt_middle_undist = cv2.undistortPoints(pt_middle.reshape(1, 1, 2), mtx2, dist2)
        
        # 對右相機點進行去畸變,原理同上
        # mtx3: 右相機內參矩陣
        # dist3: 右相機畸變係數
        pt_right_undist = cv2.undistortPoints(pt_right.reshape(1, 1, 2), mtx3, dist3)
        
        # 構建線性方程組的係數矩陣A
        A = np.zeros((6, 4))
        
        # 左相机的两个方程:
        # u_1(P_1^3X) - P_1^1X = 0  (x坐标方程)
        # v_1(P_1^3X) - P_1^2X = 0  (y坐标方程)
        # pt_left_undist[0,0,0]：第一组的第一个点的x坐标
        # pt_left_undist[0,0,1]：第一组的第一个点的y坐标
        # P1[2]：左相机投影矩阵的第3行
        # P1[0]：左相机投影矩阵的第1行
        # P1[1]：左相机投影矩阵的第2行
        A[0] = pt_left_undist[0, 0, 0] * P1[2] - P1[0]  # x坐标方程
        A[1] = pt_left_undist[0, 0, 1] * P1[2] - P1[1]  # y坐标方程
        
        # 中間相機的兩個方程:
        # u_2(P_2^3X) - P_2^1X = 0
        # v_2(P_2^3X) - P_2^2X = 0
        A[2] = pt_middle_undist[0, 0, 0] * P2[2] - P2[0]
        A[3] = pt_middle_undist[0, 0, 1] * P2[2] - P2[1]
        
        # 右相機的兩個方程:
        # u_3(P_3^3X) - P_3^1X = 0
        # v_3(P_3^3X) - P_3^2X = 0
        A[4] = pt_right_undist[0, 0, 0] * P3[2] - P3[0]
        A[5] = pt_right_undist[0, 0, 1] * P3[2] - P3[1]
        
        # 使用SVD(奇异值分解)求解AX=0
        # SVD将矩阵A分解为A = U·Σ·V^T
        # Vt是V的转置，Vt[-1]是V的最后一列，对应最小奇异值
        # 这一列就是方程AX=0的最优解
        _, _, Vt = np.linalg.svd(A)
        point_4d = Vt[-1]
        # 齐次坐标归一化：将[X,Y,Z,W]转换为[X/W,Y/W,Z/W]
        # point_4d[3]是W分量
        # [:3]表示只保留前三个元素，即[X/W,Y/W,Z/W]
        point_3d = (point_4d / point_4d[3])[:3]
        # 将计算得到的3D点添加到结果列表
        points_3d.append(point_3d)   
    # 返回所有3D点组成的数组
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
    # 創建ArUco碼字典,使用原始ArUco碼集
    # cv2.aruco.DICT_ARUCO_ORIGINAL: 使用原始ArUco碼集,包含1024個獨特的標記
    # getPredefinedDictionary(): 獲取預定義的ArUco碼字典
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

    # 創建ArUco碼檢測器參數對象
    # DetectorParameters(): 創建默認的檢測器參數,包含閾值、邊緣細化等設置
    aruco_params = cv2.aruco.DetectorParameters()

    # 創建ArUco碼檢測器
    # ArucoDetector(): 結合字典和參數創建檢測器
    # aruco_dict: 用於識別標記的字典
    # aruco_params: 檢測過程中使用的參數
    # 创建ArUco码检测器对象，使用之前定义的字典和参数
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # 定义ArUco标记板的基本参数
    board_length = 160  # 每个ArUco标记的边长(单位:)
    board_gap = 30     # 相邻标记之间的间隔(单位:)

    # 定义基础坐标，表示单个ArUco标记的四个角点相对位置
    base_coord = np.array([[0,0],[0,1],[1,1],[1,0]])  # 顺时针方向的四个角点

    # 定义整个标记板上6个ArUco标记的位置
    # 使用字典存储每个标记的坐标位置
    board_coord = {
        # 标记ID 0: 左上角标记
        0: base_coord * board_length + [0,0],
        # 标记ID 1: 右上角标记
        1: base_coord * board_length + [board_length+board_gap,0],
        # 标记ID 2: 左中标记
        2: base_coord * board_length + [0,board_length+board_gap],
        # 标记ID 3: 右中标记
        3: base_coord * board_length + [board_length+board_gap,board_length+board_gap],
        # 标记ID 4: 左下角标记
        4: base_coord * board_length + [0,(board_length+board_gap)*2],
        # 标记ID 5: 右下角标记
        5: base_coord * board_length + [board_length+board_gap,(board_length+board_gap)*2],
    }

    ## 2. 设置MediaPipe姿态估计模型
    # 指定模型文件路径
    model_asset_path = r"C:\Users\user\Desktop\pose_landmarker_full.task"
    # 检查模型文件是否存在
    if not os.path.exists(model_asset_path):
        print('model_asset_path is error.')
        return
    # 创建基础选项，设置模型路径
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    # 创建姿态检测器选项，启用分割遮罩输出
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
    # 根据选项创建姿态检测器
    detector = vision.PoseLandmarker.create_from_options(options)
    # 导入MediaPipe姿态估计模块
    mp_pose = mp.solutions.pose

    
    ## 4. 读取视频
    print(f"开始处理视频从帧 {start_frame} 开始...")
    # 打开三个摄像头的视频文件
    cap_left = cv2.VideoCapture(video_path_left)
    cap_center = cv2.VideoCapture(video_path_center)
    cap_right = cv2.VideoCapture(video_path_right)
    
    # 检查视频文件是否成功打开
    if not cap_left.isOpened() or not cap_center.isOpened() or not cap_right.isOpened():
        raise ValueError("无法打开视频文件")
        
    # 设置视频起始帧位置
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_center.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 初始化各种存储列表
    all_points_3d = []  # 存储所有帧的3D关键点
    all_rmse = []      # 存储所有帧的均方根误差
    aruco_axis = []    # 存储ArUco坐标系
    camL_axis = []     # 存储左相机坐标系
    camM_axis = []     # 存储中间相机坐标系
    camR_axis = []     # 存储右相机坐标系
    frame_count = start_frame  # 帧计数器
    
    # 定义坐标系的四个点(原点和三个轴向量端点)
    # 定义一个3D坐标系的四个关键点:
    # - 第一个点[0,0,0]是坐标系原点
    # - 第二个点[1,0,0]表示X轴正方向的单位向量端点
    # - 第三个点[0,1,0]表示Y轴正方向的单位向量端点  
    # - 第四个点[0,0,1]表示Z轴正方向的单位向量端点
    # 注意:这是真实世界的坐标系(单位:毫米)
    axis_coord = np.array([
        [0,0,0],  # 原点
        [1,0,0],  # X轴端点
        [0,1,0],  # Y轴端点
        [0,0,1]   # Z轴端点
    ],dtype=np.float32)
    # 将坐标系放大200倍(200mm=20cm),使其在3D空间中更容易可视化
    axis_coord = axis_coord * 200  # 缩放坐标系大小为200mm
    
    # 创建用于存储检测历史记录的默认字典
    global_detection_history = defaultdict(list)
    
    # 开始主循环，逐帧处理视频
    while True:
        # 读取三个摄像头的当前帧
        ret_left, frame_left = cap_left.read()
        ret_center, frame_center = cap_center.read()
        ret_right, frame_right = cap_right.read()

        # 如果任一视频读取失败，退出循环
        if not ret_left or not ret_center or not ret_right:
            break
            
        # 打印当前处理的帧号 ({frame_count + 1} 表示当前帧号)
        print(f"处理第 {frame_count + 1} 帧")
        
        # 在第912帧停止处理 (912是预设的停止帧数)
        if frame_count == 912:
            break
            
        # 调用get_aruco_axis()函数处理ArUco标记并获取坐标系变换
        # frame_left: 左相机当前帧图像
        # frame_center: 中间相机当前帧图像  
        # frame_right: 右相机当前帧图像
        # aruco_detector: 用于检测ArUco标记的检测器对象
        # board_coord: 包含ArUco标记板上每个标记ID及其3D坐标的字典
        # cams_params: 包含三个相机内参和畸变系数的元组(mtx1,dist1,mtx2,dist2,mtx3,dist3)
        # result: 返回包含旋转矩阵、平移向量和处理后图像的元组
        result = get_aruco_axis(frame_left, frame_center, frame_right, aruco_detector, board_coord, cams_params)
        
        # 检查是否成功检测到ArUco标记
        # result[0]是返回元组中的第一个元素,即R_aruco2camL旋转矩阵
        # is None: 判断是否为None值,表示检测失败
        if result[0] is None:
            # 使用f-string格式化字符串打印检测失败信息
            # frame_count从0开始计数,但通常我们习惯从1开始计数帧号
            # 所以这里+1转换为人类习惯的帧号显示方式
            print(f"第 {frame_count + 1} 帧未检测到 ArUco 标记")
            
            # np.zeros(): 创建指定形状的全零数组
            # (33,3): 33行3列,表示33个关键点,每个点有xyz三个坐标
            # append(): 将数组添加到列表末尾
            all_points_3d.append(np.zeros((33, 3)))  # 添加全零的人体3D关键点数据
            
            # (4,3): 4行3列,表示坐标系的4个点(原点和3个轴向量端点)
            # 每个点有xyz三个坐标
            aruco_axis.append(np.zeros((4,3)))  # 添加全零的ArUco标记板坐标系数据
            
            # 为三个相机坐标系分别添加全零数据
            # 每个相机坐标系同样用4个点表示
            camL_axis.append(np.zeros((4,3)))  # 左相机坐标系
            camM_axis.append(np.zeros((4,3)))  # 中间相机坐标系  
            camR_axis.append(np.zeros((4,3)))  # 右相机坐标系
            
            # += 运算符: 将frame_count增加1
            frame_count += 1  # 帧计数器递增
            
            # continue语句: 跳过当前循环的剩余部分
            # 直接开始处理下一帧
            continue
        # 从result中解包所有返回值 (R_*: 旋转矩阵, t_*: 平移向量, img_*: 处理后的图像)
        (R_aruco2camL, t_aruco2camL, 
         R_aruco2camM, t_aruco2camM,
         R_aruco2camR, t_aruco2camR,
         R_camL2aruco, t_camL2aruco,
         R_camM2aruco, t_camM2aruco,
         R_camR2aruco, t_camR2aruco,
         img_L, img_M, img_R) = result
         
        # 更新各个坐标系的数据
        # 添加ArUco标记板坐标系 (axis_coord: 原始坐标系点)
        aruco_axis.append(axis_coord)
        # 添加左相机坐标系 (np.dot(): 矩阵乘法, .T: 矩阵转置)
        camL_axis.append(np.dot(R_camL2aruco, (axis_coord).T).T + t_camL2aruco.T)
        # 添加中间相机坐标系
        camM_axis.append(np.dot(R_camM2aruco, (axis_coord).T).T + t_camM2aruco.T)
        # 添加右相机坐标系
        camR_axis.append(np.dot(R_camR2aruco, (axis_coord).T).T + t_camR2aruco.T)

        # 整理相机姿态参数 (cam_P: 包含所有相机的旋转和平移参数的元组)
        cam_P = (R_aruco2camL, t_aruco2camL, R_aruco2camM, t_aruco2camM, R_aruco2camR, t_aruco2camR)
        # 处理当前帧 (detector: 人体关键点检测器, 返回3D点云、处理后的图像和重投影误差)
        points_3d, img_L, img_M, img_R, rmse = process_frame(detector, frame_left, frame_center, frame_right, 
                                                      img_L, img_M, img_R, cams_params, cam_P)
                                                      
        # 如果成功获取3D点云数据 (points_3d不为None表示处理成功)
        if points_3d is not None:
            # 保存3D点云数据 (append: 添加到列表末尾)
            all_points_3d.append(points_3d)
            # 保存重投影误差 (rmse: root mean square error)
            all_rmse.append(rmse)
            
            # 打印当前帧的重投影误差 ({:.3f}: 保留3位小数)
            print(f"Frame {frame_count} RMSE - Left: {rmse[0]:.3f}, Middle: {rmse[1]:.3f}, Right: {rmse[2]:.3f}")
        
        # 将三个相机的图像水平拼接并显示 (np.hstack: 水平堆叠数组)
        combined_img = np.hstack((
            cv2.resize(img_L, (426, 320)),  # 调整左相机图像大小为426x320
            cv2.resize(img_M, (426, 320)),  # 调整中间相机图像大小为426x320
            cv2.resize(img_R, (426, 320))   # 调整右相机图像大小为426x320
        ))
        # 显示拼接后的图像 ('Three Camera Views': 窗口标题)
        cv2.imshow('Three Camera Views', combined_img)
        # 检查是否按下'q'键退出 (waitKey(1): 等待1ms, &0xFF: 取最后8位, ord('q'): 'q'的ASCII码)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 增加帧计数器 (+=1: 计数器加1)
        frame_count += 1
        # 每处理50帧打印一次进度 (%: 取余运算符)
        if frame_count % 50 == 0:
            print(f"已處理 {frame_count} 幀")

    # 释放所有视频捕获对象 (release(): 释放视频捕获资源)
    cap_left.release()
    cap_center.release()
    cap_right.release()
    # 关闭所有OpenCV窗口 (destroyAllWindows(): 关闭所有cv2创建的窗口)
    cv2.destroyAllWindows()

    # 打印处理完成信息 (frame_count - start_frame: 计算处理的总帧数)
    print(f"視頻處理完成。共處理了 {frame_count - start_frame} 幀")
    
    # 计算并显示平均重投影误差
    # 将误差数据转换为numpy数组 (np.array: 创建numpy数组)
    all_rmse = np.array(all_rmse)
    # 计算每个相机的平均误差 (np.mean: 计算平均值, axis=0: 沿第0轴计算)
    mean_rmse = np.mean(all_rmse, axis=0)
    # 打印误差统计信息 (\n: 换行符)
    print("\n平均重投影誤差(RMSE):")
    print(f"左相機: {mean_rmse[0]:.3f} pixels")
    print(f"中間相機: {mean_rmse[1]:.3f} pixels")
    print(f"右相機: {mean_rmse[2]:.3f} pixels")
    print(f"總體平均: {np.mean(mean_rmse):.3f} pixels")
    
    # 返回所有处理结果 (np.array: 将列表转换为numpy数组)
    return (np.array(all_points_3d), np.array(aruco_axis), np.array(camL_axis), np.array(camM_axis), np.array(camR_axis), np.array(all_rmse))

def visualize_3d_animation_three_cameras(points, aruco_axis, camL_axis, camM_axis, camR_axis, title='3D Visualization'):
    """
    三相機系統的3D點雲動畫可視化
    Args:
        points: 3D關鍵點序列 [frames, 33, 3]  # frames:幀數, 33:關鍵點數, 3:xyz座標
        aruco_axis: ArUco坐標系  # ArUco標記的坐標系
        camL_axis: 左相機坐標系  # 左相機的坐標系
        camM_axis: 中間相機坐標系  # 中間相機的坐標系
        camR_axis: 右相機坐標系  # 右相機的坐標系
        title: 視窗標題  # 可視化視窗的標題
    """
    # 創建一個新的圖形視窗,大小為12x8英寸
    fig = plt.figure(figsize=(12, 8))  # figsize=(寬度,高度)
    # 添加3D子圖,111表示1x1網格的第1個位置
    ax = fig.add_subplot(111, projection='3d')  # projection='3d':指定為3D圖
    
    # 計算所有點的範圍
    # np.vstack:垂直堆疊數組
    # reshape(-1,3):將數組重塑為nx3的形狀,其中n自動計算
    all_points = np.vstack((
        points.reshape(-1, 3),  # 關鍵點
        aruco_axis.reshape(-1, 3),  # ArUco坐標系點
        camL_axis.reshape(-1, 3),  # 左相機坐標系點
        camM_axis.reshape(-1, 3),  # 中間相機坐標系點
        camR_axis.reshape(-1, 3)  # 右相機坐標系點
    ))
    # 計算所有點在每個維度的最小值
    min_vals = np.min(all_points, axis=0)  # axis=0:沿第0軸計算
    # 計算所有點在每個維度的最大值
    max_vals = np.max(all_points, axis=0)  # axis=0:沿第0軸計算
    # 計算數值範圍(最大值-最小值)
    range_vals = max_vals - min_vals
    
    # 設置坐標軸範圍,添加邊距
    margin = 0.1 * range_vals  # 邊距為範圍的10%
    # 設置x軸範圍
    # min_vals[0] - margin[0]: x軸最小值減去10%的邊距作為下限
    # max_vals[0] + margin[0]: x軸最大值加上10%的邊距作為上限
    ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
    # 設置y軸範圍
    # min_vals[1] - margin[1]: y軸最小值減去10%的邊距作為下限
    # max_vals[1] + margin[1]: y軸最大值加上10%的邊距作為上限
    ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
    # 設置z軸範圍
    # min_vals[2]: 所有3D點在z軸方向上的最小值,通過np.min()計算得到
    # margin[2]: z軸方向上的邊距,為z軸數值範圍的10%
    # min_vals[2] - margin[2]: z軸最小值減去邊距作為下限,使圖像有留白
    # max_vals[2] + margin[2]: z軸最大值加上邊距作為上限,使圖像有留白
    ax.set_zlim(min_vals[2] - margin[2], max_vals[2] + margin[2])
    
    # 設置坐標軸標籤
    ax.set_xlabel('X')  # x軸標籤
    ax.set_ylabel('Y')  # y軸標籤
    ax.set_zlabel('Z')  # z軸標籤
    
    # 設置視角(elev:仰角,azim:方位角)
    ax.view_init(elev=10, azim=-60)
    
    # 添加地板
    floor_y = min_vals[1]  # 地板y座標為最小y值
    # 地板x範圍
    x_floor = np.array([min_vals[0] - margin[0], max_vals[0] + margin[0]])
    # 地板z範圍
    z_floor = np.array([min_vals[2] - margin[2], max_vals[2] + margin[2]])
    # 創建地板網格
    X_floor, Z_floor = np.meshgrid(x_floor, z_floor)
    # 創建地板y值陣列(全部相同)
    Y_floor = np.full(X_floor.shape, floor_y)
    # 繪製地板表面
    ax.plot_surface(X_floor, Y_floor, Z_floor, alpha=0.2, color='gray')
    
    # 初始化散點圖(s:點大小,c:顏色,alpha:透明度)
    scatter = ax.scatter([], [], [], s=20, c='r', alpha=0.6)
    
    # 定義骨架連接(每個元組表示兩個關鍵點的索引)
    connections = [
        # 頭部連接
        (0, 1), (1, 2), (2, 3), (3, 7),  # 左側頭部
        (0, 4), (4, 5), (5, 6), (6, 8),  # 右側頭部
        (3, 6),  # 頭部橫向連接
        # 頸部連接
        (9, 10),  # 頸椎
        # 軀幹連接
        (11, 12), (11, 23), (12, 24), (23, 24),  # 胸腔框架
        # 左臂連接
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # 左臂主要骨架
        (17, 19), (19, 21),  # 左手指連接
        # 右臂連接
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # 右臂主要骨架
        (18, 20), (20, 22),  # 右手指連接
        # 左腿連接
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # 左腿骨架
        # 右腿連接
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)  # 右腿骨架
    ]
    
    # 為不同部位設置顏色
    colors = {
        'head': 'purple',    # 頭部:紫色
        'spine': 'blue',     # 脊椎:藍色
        'arms': 'green',     # 手臂:綠色
        'legs': 'red',       # 腿部:紅色
        'hands': 'orange'    # 手部:橙色
    }
    
    # 定義每個連接的顏色
    connection_colors = []
    for start, end in connections:
        # 根據連接點的索引決定顏色
        if start <= 8 or end <= 8:  # 頭部點的索引範圍
            connection_colors.append(colors['head'])
        elif start in [9, 10, 11] or end in [9, 10, 11]:  # 脊椎點的索引
            connection_colors.append(colors['spine'])
        elif (start in [13, 14, 15, 16] or end in [13, 14, 15, 16]):  # 手臂點的索引
            connection_colors.append(colors['arms'])
        elif start >= 17 or end >= 17:  # 手部點的索引
            connection_colors.append(colors['hands'])
        else:  # 其餘點為腿部
            connection_colors.append(colors['legs'])
    
    # 創建線條對象列表
    lines = []
    for color in connection_colors:
        # 為每個連接創建一條線(color:顏色,alpha:透明度,linewidth:線寬)
        line, = ax.plot([], [], [], color=color, alpha=0.8, linewidth=2)
        lines.append(line)
    
    # 添加坐標系線條
    coord_lines = []
    for _ in range(12):  # 4個坐標系(ArUco,左,中,右) × 3個軸(x,y,z)
        # 創建坐標軸線條(lw:線寬,alpha:透明度)
        line, = ax.plot([], [], [], '-', lw=2, alpha=0.7)
        coord_lines.append(line)
    
    def update(frame):
        """
        更新動畫的每一幀
        Args:
            frame: 當前幀索引
        """
        # 更新骨骼點位置
        point_cloud = points[frame]  # 獲取當前幀的點雲數據
        # scatter._offsets3d: 散點圖的3D位置屬性
        # point_cloud[:,0]: 所有點的x座標
        # point_cloud[:,1]: 所有點的y座標 
        # point_cloud[:,2]: 所有點的z座標
        scatter._offsets3d = (point_cloud[:,0], point_cloud[:,1], point_cloud[:,2])
        
        # 更新骨架線條
        # zip(): 將connections和lines打包成元組進行遍歷
        # enumerate(): 生成索引和元素對
        # i: 當前迭代的索引
        # start,end: 連接的起點和終點索引
        # line: 對應的線條對象
        for i, ((start, end), line) in enumerate(zip(connections, lines)):
            # line.set_data_3d(): 設置線條的3D數據
            # point_cloud[start,0]: 起點的x座標
            # point_cloud[end,0]: 終點的x座標
            # point_cloud[start,1]: 起點的y座標
            # point_cloud[end,1]: 終點的y座標
            # point_cloud[start,2]: 起點的z座標
            # point_cloud[end,2]: 終點的z座標
            line.set_data_3d([point_cloud[start,0], point_cloud[end,0]],  # x座標
                           [point_cloud[start,1], point_cloud[end,1]],    # y座標
                           [point_cloud[start,2], point_cloud[end,2]])    # z座標
        
        # 更新坐標系
        # ArUco坐標系
        # range(3): 遍歷x,y,z三個軸
        for i in range(3):
            # coord_lines[i]: 第i個坐標軸線條
            # aruco_axis[frame,0,0/1/2]: 原點座標
            # aruco_axis[frame,i+1,0/1/2]: 軸向量端點座標
            coord_lines[i].set_data_3d([aruco_axis[frame,0,0], aruco_axis[frame,i+1,0]],  # x座標
                                     [aruco_axis[frame,0,1], aruco_axis[frame,i+1,1]],    # y座標
                                     [aruco_axis[frame,0,2], aruco_axis[frame,i+1,2]])    # z座標
            # ['r','g','b'][i]: 根據索引選擇顏色(紅綠藍)
            coord_lines[i].set_color(['r','g','b'][i])  # 設置顏色
        
        # 左相機坐標系
        # i+3: 左相機坐標軸的索引從3開始
        for i in range(3):
            # camL_axis: 左相機坐標系數據
            coord_lines[i+3].set_data_3d([camL_axis[frame,0,0], camL_axis[frame,i+1,0]],
                                       [camL_axis[frame,0,1], camL_axis[frame,i+1,1]],
                                       [camL_axis[frame,0,2], camL_axis[frame,i+1,2]])
            coord_lines[i+3].set_color(['r','g','b'][i])  # 設置顏色
        
        # 中間相機坐標系
        # i+6: 中間相機坐標軸的索引從6開始
        for i in range(3):
            # camM_axis: 中間相機坐標系數據
            coord_lines[i+6].set_data_3d([camM_axis[frame,0,0], camM_axis[frame,i+1,0]],
                                       [camM_axis[frame,0,1], camM_axis[frame,i+1,1]],
                                       [camM_axis[frame,0,2], camM_axis[frame,i+1,2]])
            coord_lines[i+6].set_color(['r','g','b'][i])  # 設置顏色
        
        # 右相機坐標系
        # i+9: 右相機坐標軸的索引從9開始
        for i in range(3):
            # camR_axis: 右相機坐標系數據
            coord_lines[i+9].set_data_3d([camR_axis[frame,0,0], camR_axis[frame,i+1,0]],
                                       [camR_axis[frame,0,1], camR_axis[frame,i+1,1]],
                                       [camR_axis[frame,0,2], camR_axis[frame,i+1,2]])
            coord_lines[i+9].set_color(['r','g','b'][i])  # 設置顏色
        
        # ax.set_title(): 設置圖形標題
        # f'...': 格式化字符串
        # {title}: 插入標題變量
        # {frame}: 插入當前幀索引
        ax.set_title(f'{title} - Frame: {frame}')
        
        # 返回所有更新的圖形對象
        # [scatter]: 散點圖對象列表
        # lines: 骨架線條列表
        # coord_lines: 坐標軸線條列表
        return [scatter] + lines + coord_lines
    # 創建動畫
    anim = FuncAnimation(
        fig,                # 圖形對象
        update,            # 更新函數
        frames=len(points), # 總幀數
        interval=50,       # 幀間隔(毫秒)
        blit=False,        # 是否使用blitting優化
        repeat=True        # 是否重複播放
    )
    
    # 顯示動畫
    plt.show()
    # 返回動畫對象
    return anim

# 13. 主程
def main():
    # 确保视频路径正确
    video_path_left = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\recordings\port_3\port_3-01082025180809-0000.avi"
    video_path_center = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\recordings\port_1\port_1-01082025180808-0000.avi"
    video_path_right = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\recordings\port_2\port_2-01082025180810-0000.avi"

    # 处理所有帧
    all_points_3d_original, aruco_axises, camL_axises, camM_axises, camR_axises, all_rmse = process_videos(video_path_left, video_path_center, video_path_right)

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

    # 繪製RMSE隨時間變化的圖表
    plt.figure(figsize=(10, 6))
    frames = range(len(all_rmse))
    plt.plot(frames, all_rmse[:, 0], label='Left Camera')
    plt.plot(frames, all_rmse[:, 1], label='Middle Camera')
    plt.plot(frames, all_rmse[:, 2], label='Right Camera')
    plt.xlabel('Frame')
    plt.ylabel('RMSE (pixels)')
    plt.title('Reprojection Error Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

