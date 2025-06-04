"""
配置文件：存儲所有外部參數
"""

import os

# 文件路徑配置
CAMERA_PARAMS_PATH = r"C:\Users\godli\Dropbox\camera_8\eight_camera_calibration.npz"
MODEL_ASSET_PATH = r"C:\Users\godli\OneDrive\Desktop\pose_landmarker_full.task"

# 視頻路徑配置
#VIDEO_PATHS = {
#    'l1': r"D:\20250116\recordings\part03\port_6-01162025125609-0000.avi",
#    'l2': r"D:\20250116\recordings\part03\port_7-01162025125620-0000.avi",
#    'l3': r"D:\20250116\recordings\part03\port_8-01162025125607-0000.avi",
#    'c': r"D:\20250116\recordings\part03\port_5-01162025125613-0000.avi",
#    'r1': r"D:\20250116\recordings\part03\port_4-01162025125615-0000.avi",
#    'r2': r"D:\20250116\recordings\part03\port_3-01162025125611-0000.avi",

#}
# ArUco 參數配置
ARUCO_CONFIG = {
    'board_length': 160,  # 標記板邊長
    'board_gap': 30,      # 標記板間距
}

# MediaPipe 配置
MEDIAPIPE_CONFIG = {
    'min_pose_detection_confidence': 0.8,
    'min_pose_presence_confidence': 0.8,
    'min_tracking_confidence': 0.8
}

# 相機順序配置
CAM_ORDER = ['l1', 'l2', 'l3', 'c', 'r1', 'r2', 'r3', 'r4']

# 相機名稱映射
CAM_NAMES = {
    'l1': "左側相機1", 'l2': "左側相機2", 'l3': "左側相機3",
    'c': "中心相機",
    'r1': "右側相機1", 'r2': "右側相機2", 'r3': "右側相機3", 'r4': "右側相機4"
}

# 檢查配置的文件路徑是否存在
def validate_paths():
    """驗證所有配置的文件路徑"""
    if not os.path.exists(CAMERA_PARAMS_PATH):
        raise FileNotFoundError(f"相機參數文件不存在: {CAMERA_PARAMS_PATH}")
    
    if not os.path.exists(MODEL_ASSET_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_ASSET_PATH}") 