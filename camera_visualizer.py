from PyQt5.QtWidgets import QApplication
import sys

# 確保在導入pyqtgraph之前創建QApplication實例
app = QApplication(sys.argv)

import numpy as np
import pyqtgraph.opengl as gl
import cv2

class CameraMesh:
    """
    用於創建相機的3D網格模型
    """
    def __init__(self, R=None, t=None, size=1.0, color=(1,1,1,1)):
        self.size = size
        self.color = color
        
        # 相機視錐體的頂點 - 調整為更直觀的形狀
        self.vertices = np.array([
            [0, 0, 0],          # 相機中心
            [-size/2, -size/2, size],   # 前方左下
            [size/2, -size/2, size],    # 前方右下
            [size/2, size/2, size],     # 前方右上
            [-size/2, size/2, size],    # 前方左上
        ])
        
        # 如果提供了旋轉和平移，應用變換
        if R is not None and t is not None:
            self.apply_transform(R, t)
            
        # 定義面的索引（全部使用三角形）
        self.faces = np.array([
            [0, 1, 2],  # 底面三角形1
            [0, 2, 3],  # 底面三角形2
            [0, 3, 4],  # 左面三角形
            [0, 4, 1],  # 右面三角形
            [1, 2, 3],  # 前面三角形1
            [1, 3, 4]   # 前面三角形2
        ])
        
        # 創建網格項 - 調整透明度和邊緣顯示
        self.mesh = gl.GLMeshItem(
            vertexes=self.vertices,
            faces=self.faces,
            smooth=False,
            drawEdges=True,
            edgeColor=self.color,
            color=(*self.color[:3], 0.2)  # 更透明的填充
        )
        
    def apply_transform(self, R, t):
        """應用旋轉和平移變換"""
        # 創建變換矩陣
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        # 將頂點轉換為齊次坐標
        vertices_h = np.hstack((self.vertices, np.ones((self.vertices.shape[0], 1))))
        
        # 應用變換
        transformed_vertices_h = vertices_h @ T.T
        
        # 轉回3D坐標
        self.vertices = transformed_vertices_h[:, :3]

class CameraArrayVisualizer:
    """
    用於可視化多相機系統的類
    """
    def __init__(self):
        # 使用全局app實例
        self.app = app
        
        # 創建3D視圖窗口
        self.view = gl.GLViewWidget()
        self.view.setWindowTitle('相機陣列可視化')
        self.view.setCameraPosition(distance=2000, elevation=30, azimuth=45)  # 調整視角
        self.view.opts['distance'] = 2000  # 調整默認視距
        
        # 添加坐標軸並調整大小
        self.axis = gl.GLAxisItem()
        self.axis.setSize(x=500, y=500, z=500)  # 調整坐標軸大小
        self.view.addItem(self.axis)
        
        # 存儲相機網格
        self.camera_meshes = {}
        
    def add_camera(self, camera_id, R, t, color=(1,1,1,1), size=200):  # 調整相機大小
        """添加一個相機到場景中"""
        camera_mesh = CameraMesh(R, t, size, color)
        self.camera_meshes[camera_id] = camera_mesh
        self.view.addItem(camera_mesh.mesh)
        
    def show(self):
        """顯示可視化窗口"""
        self.view.show()
        
    def add_capture_volume(self, points, color=(1,0,0,1), size=5):
        """添加捕獲體積的點雲"""
        scatter = gl.GLScatterPlotItem(
            pos=points,
            color=color,
            size=size,
            pxMode=False
        )
        self.view.addItem(scatter)

    def start(self):
        """啟動應用程序主循環"""
        self.show()
        return self.app.exec_()

if __name__ == "__main__":
    # 導入相機參數
    camera_params_path = r"C:\Users\user\Desktop\wenfeng_caliscope\three_camera_calibration.npz"
    calib_data = np.load(camera_params_path)
    mtx1, dist1, mtx2, dist2, mtx3, dist3 = (
        calib_data['mtx1'], calib_data['dist1'],
        calib_data['mtx2'], calib_data['dist2'],
        calib_data['mtx3'], calib_data['dist3']
    )
    cams_params = (mtx1, dist1, mtx2, dist2, mtx3, dist3)

    # 設置ArUco參數
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # 定義ArUco標記板的坐標
    board_length = 160
    board_gap = 30
    base_coord = np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]])  # 添加Z坐標為0
    board_coord = {
        0: base_coord * [board_length, board_length, 1] + [0,0,0],
        1: base_coord * [board_length, board_length, 1] + [board_length+board_gap,0,0],
        2: base_coord * [board_length, board_length, 1] + [0,board_length+board_gap,0],
        3: base_coord * [board_length, board_length, 1] + [board_length+board_gap,board_length+board_gap,0],
        4: base_coord * [board_length, board_length, 1] + [0,(board_length+board_gap)*2,0],
        5: base_coord * [board_length, board_length, 1] + [board_length+board_gap,(board_length+board_gap)*2,0],
    }

    # 讀取視頻第一幀
    video_path_left = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\recordings\port_3\port_3-01082025180809-0000.avi"
    video_path_center = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\recordings\port_1\port_1-01082025180808-0000.avi"
    video_path_right = r"C:\Users\user\Desktop\Dropbox\Camera_passion changes lives\calibration0108\recordings\port_2\port_2-01082025180810-0000.avi"

    cap_left = cv2.VideoCapture(video_path_left)
    cap_center = cv2.VideoCapture(video_path_center)
    cap_right = cv2.VideoCapture(video_path_right)

    ret_left, frame_left = cap_left.read()
    ret_center, frame_center = cap_center.read()
    ret_right, frame_right = cap_right.read()

    if not (ret_left and ret_center and ret_right):
        print("無法讀取視頻幀")
        sys.exit(1)

    # 處理ArUco標記
    def get_aruco_pose(frame, aruco_detector, board_coord, mtx, dist):
        corners, ids, _ = aruco_detector.detectMarkers(frame)
        if ids is None:
            print("未檢測到任何ArUco標記")
            return None, None

        print(f"檢測到 {len(ids)} 個ArUco標記")
        print(f"標記ID: {ids.flatten()}")

        # 收集所有檢測到的角点和對應的3D坐標
        img_points = []
        obj_points = []
        for i in range(len(ids)):
            marker_id = ids[i][0]
            if marker_id not in board_coord:
                print(f"跳過未知ID: {marker_id}")
                continue
            print(f"處理標記ID {marker_id}")
            # 只使用每個標記的四个角點
            marker_corners = corners[i][0]
            marker_obj_points = board_coord[marker_id]
            
            img_points.extend(marker_corners)
            obj_points.extend(marker_obj_points)

        if not img_points:
            print("沒有有效的點對應關係")
            return None, None

        print(f"總共收集到 {len(img_points)} 個點對應關係")
        
        # 轉換為numpy數組並確保數據類型正確
        img_points = np.array(img_points, dtype=np.float32)
        obj_points = np.array(obj_points, dtype=np.float32)

        print(f"圖像點形狀: {img_points.shape}")
        print(f"物體點形狀: {obj_points.shape}")

        try:
            # 使用SOLVEPNP_ITERATIVE方法，這是最穩定的方法
            ret, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                mtx,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            R, _ = cv2.Rodrigues(rvec)
            print("成功計算相機姿態")
            return R, tvec
        except cv2.error as e:
            print(f"PnP解算失敗: {e}")
            return None, None

    # 获取三个相機的位姿
    print("\n處理左相機...")
    R1, t1 = get_aruco_pose(frame_left, aruco_detector, board_coord, mtx1, dist1)
    print("\n處理中间相機...")
    R2, t2 = get_aruco_pose(frame_center, aruco_detector, board_coord, mtx2, dist2)
    print("\n處理右相機...")
    R3, t3 = get_aruco_pose(frame_right, aruco_detector, board_coord, mtx3, dist3)

    # 释放視頻捕獲
    cap_left.release()
    cap_center.release()
    cap_right.release()

    # 檢查是否所有相機姿態都成功計算
    if R1 is None or t1 is None or R2 is None or t2 is None or R3 is None or t3 is None:
        print("無法在所有相機中檢測到ArUco標記")
        sys.exit(1)

    # 準備相機參數
    camera_params = {
        0: {'R': R1, 't': t1},  # 左相機
        1: {'R': R2, 't': t2},  # 中间相機
        2: {'R': R3, 't': t3}   # 右相機
    }

    print("\n相機姿態計算完成:")
    for cam_id, params in camera_params.items():
        print(f"\n相機 {cam_id}:")
        print("旋轉矩陣 R:")
        print(params['R'])
        print("平移向量 t:")
        print(params['t'])

    # 創建可視化器
    visualizer = CameraArrayVisualizer()
    
    # 添加每个相機
    colors = {
        0: (1,0,0,1),  # 紅色
        1: (0,1,0,1),  # 绿色
        2: (0,0,1,1),  # 蓝色
    }
    
    for camera_id, params in camera_params.items():
        color = colors.get(camera_id, (1,1,1,1))
        visualizer.add_camera(camera_id, params['R'], params['t'], color)
    
    # 啟動應用程序
    sys.exit(visualizer.start()) 