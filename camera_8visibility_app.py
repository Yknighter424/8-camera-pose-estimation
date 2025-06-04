from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                           QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QTextEdit, QFileDialog, QGroupBox, QGridLayout, 
                           QSpinBox, QDoubleSpinBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QImage
import os
import sys
import cv2
import numpy as np
from datetime import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvas

# 導入相機處理相關函數
from camera_8visibility import (calibrate_eight_cameras, process_videos, 
                              visualize_3d_animation_eight_cameras)

class CameraVisibilityApp(QMainWindow):
    """八相機系統的主應用程序窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("八相機系統分析工具")
        self.setGeometry(100, 100, 1600, 900)
        
        # 初始化成員變量
        self.video_paths = {}
        self.caps = {}
        self.current_frame = 0
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 創建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 創建主布局
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 創建標籤頁
        self.create_tabs()
        
        # 初始化UI
        self.init_ui()
    
    def create_tabs(self):
        """創建標籤頁"""
        self.tabs = QTabWidget()
        
        # 相機標定標籤頁
        self.calibration_tab = QWidget()
        self.calibration_layout = QVBoxLayout(self.calibration_tab)
        self.create_calibration_widgets()
        self.tabs.addTab(self.calibration_tab, "相機標定")
        
        # 視頻顯示標籤頁
        self.video_tab = QWidget()
        self.video_layout = QVBoxLayout(self.video_tab)
        self.create_video_widgets()
        self.tabs.addTab(self.video_tab, "視頻顯示")
        
        # 3D可視化標籤頁
        self.visualization_tab = QWidget()
        self.visualization_layout = QVBoxLayout(self.visualization_tab)
        self.create_visualization_widgets()
        self.tabs.addTab(self.visualization_tab, "3D可視化")
        
        # 數據分析標籤頁
        self.analysis_tab = QWidget()
        self.analysis_layout = QVBoxLayout(self.analysis_tab)
        self.create_analysis_widgets()
        self.tabs.addTab(self.analysis_tab, "數據分析")
        
        self.main_layout.addWidget(self.tabs)
    
    def create_video_widgets(self):
        """創建視頻顯示相關的部件"""
        # 控制按鈕組
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()
        
        # 添加視頻文件按鈕
        self.load_btn = QPushButton("加載視頻文件")
        self.load_btn.clicked.connect(self.load_videos)
        
        # 播放控制按鈕
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        
        # 重置按鈕
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_playback)
        self.reset_btn.setEnabled(False)
        
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.reset_btn)
        control_group.setLayout(control_layout)
        
        # 視頻顯示區域
        self.video_displays = {}
        video_grid = QHBoxLayout()
        
        # 創建左側4個相機的顯示
        left_layout = QVBoxLayout()
        for cam_id in ['l1', 'l2', 'l3', 'c']:
            display = QLabel()
            display.setMinimumSize(380, 285)
            display.setAlignment(Qt.AlignCenter)
            display.setStyleSheet("border: 1px solid black")
            self.video_displays[cam_id] = display
            left_layout.addWidget(display)
        
        # 創建右側4個相機的顯示
        right_layout = QVBoxLayout()
        for cam_id in ['r1', 'r2', 'r3', 'r4']:
            display = QLabel()
            display.setMinimumSize(380, 285)
            display.setAlignment(Qt.AlignCenter)
            display.setStyleSheet("border: 1px solid black")
            self.video_displays[cam_id] = display
            right_layout.addWidget(display)
        
        video_grid.addLayout(left_layout)
        video_grid.addLayout(right_layout)
        
        # 添加到主布局
        self.video_layout.addWidget(control_group)
        self.video_layout.addLayout(video_grid)
    
    def create_calibration_widgets(self):
        """創建相機標定相關的部件"""
        # 標定控制組
        control_group = QGroupBox("標定控制")
        control_layout = QVBoxLayout()
        
        # 添加標定圖片文件夾按鈕組
        folders_group = QGroupBox("選擇標定圖片文件夾")
        folders_layout = QGridLayout()
        
        self.folder_paths = {}
        self.folder_labels = {}
        camera_names = {
            'l1': "左側相機1", 'l2': "左側相機2", 'l3': "左側相機3",
            'c': "中心相機",
            'r1': "右側相機1", 'r2': "右側相機2", 'r3': "右側相機3", 'r4': "右側相機4"
        }
        
        row = 0
        col = 0
        for cam_id, cam_name in camera_names.items():
            # 創建按鈕和標籤
            btn = QPushButton(f"選擇{cam_name}文件夾")
            label = QLabel("未選擇")
            label.setWordWrap(True)
            
            # 設置按鈕點擊事件
            btn.clicked.connect(lambda checked, cid=cam_id: self.select_calibration_folder(cid))
            
            # 保存標籤引用
            self.folder_labels[cam_id] = label
            
            # 添加到布局
            folders_layout.addWidget(btn, row, col*2)
            folders_layout.addWidget(label, row, col*2+1)
            
            col += 1
            if col == 2:
                col = 0
                row += 1
        
        folders_group.setLayout(folders_layout)
        
        # 標定參數設置組
        params_group = QGroupBox("標定參數")
        params_layout = QGridLayout()
        
        # 棋盤格尺寸設置
        params_layout.addWidget(QLabel("棋盤格內角點數量:"), 0, 0)
        self.pattern_size_w = QSpinBox()
        self.pattern_size_w.setValue(9)
        self.pattern_size_h = QSpinBox()
        self.pattern_size_h.setValue(6)
        params_layout.addWidget(self.pattern_size_w, 0, 1)
        params_layout.addWidget(QLabel("x"), 0, 2)
        params_layout.addWidget(self.pattern_size_h, 0, 3)
        
        # 方格實際尺寸設置
        params_layout.addWidget(QLabel("方格實際尺寸(mm):"), 1, 0)
        self.square_size = QDoubleSpinBox()
        self.square_size.setValue(30.0)
        params_layout.addWidget(self.square_size, 1, 1)
        
        params_group.setLayout(params_layout)
        
        # 開始標定按鈕
        self.calibrate_btn = QPushButton("開始標定")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        
        # 標定結果顯示
        self.calibration_result = QTextEdit()
        self.calibration_result.setReadOnly(True)
        
        # 添加所有控件到布局
        control_layout.addWidget(folders_group)
        control_layout.addWidget(params_group)
        control_layout.addWidget(self.calibrate_btn)
        control_group.setLayout(control_layout)
        
        self.calibration_layout.addWidget(control_group)
        self.calibration_layout.addWidget(self.calibration_result)
    
    def select_calibration_folder(self, cam_id):
        """選擇標定圖片文件夾"""
        folder = QFileDialog.getExistingDirectory(self, f"選擇{cam_id}相機的標定圖片文件夾")
        if folder:
            self.folder_paths[cam_id] = folder
            self.folder_labels[cam_id].setText(folder)
            self.folder_labels[cam_id].setToolTip(folder)
    
    def start_calibration(self):
        """開始相機標定"""
        if len(self.folder_paths) != 8:
            self.calibration_result.append("錯誤：請先選擇所有相機的標定圖片文件夾")
            return
        
        pattern_size = (self.pattern_size_w.value(), self.pattern_size_h.value())
        square_size = self.square_size.value()
        
        try:
            # 執行標定
            save_path = calibrate_eight_cameras(
                self.folder_paths['l1'], self.folder_paths['l2'], 
                self.folder_paths['l3'], self.folder_paths['c'],
                self.folder_paths['r1'], self.folder_paths['r2'], 
                self.folder_paths['r3'], self.folder_paths['r4'],
                pattern_size=pattern_size,
                square_size=square_size
            )
            
            self.calibration_result.append(f"標定完成！結果保存在：{save_path}")
            
        except Exception as e:
            self.calibration_result.append(f"標定失敗：{str(e)}")
    
    def create_visualization_widgets(self):
        """創建3D可視化相關的部件"""
        # 控制按鈕組
        control_group = QGroupBox("3D視圖控制")
        control_layout = QHBoxLayout()
        
        # 添加視圖控制按鈕
        self.start_3d_btn = QPushButton("開始3D重建")
        self.start_3d_btn.clicked.connect(self.start_3d_reconstruction)
        self.start_3d_btn.setEnabled(False)
        
        control_layout.addWidget(self.start_3d_btn)
        control_group.setLayout(control_layout)
        
        # 3D視圖容器
        self.plotter_widget = QWidget()
        
        # 添加到主布局
        self.visualization_layout.addWidget(control_group)
        self.visualization_layout.addWidget(self.plotter_widget)
    
    def start_3d_reconstruction(self):
        """開始3D重建"""
        if not hasattr(self, 'video_paths') or not self.video_paths:
            QMessageBox.warning(self, "警告", "請先加載視頻文件")
            return
        
        try:
            # 處理視頻並獲取3D點雲
            points_3d, aruco_axis, cam_axes = process_videos(self.video_paths)
            
            if points_3d is not None and len(points_3d) > 0:
                # 創建3D動畫
                anim = visualize_3d_animation_eight_cameras(
                    points_3d,
                    aruco_axis,
                    cam_axes,
                    title='Eight Camera Motion Capture'
                )
                
                # 將動畫嵌入到應用程序窗口中
                canvas = FigureCanvas(anim._fig)
                layout = QVBoxLayout(self.plotter_widget)
                layout.addWidget(canvas)
                
            else:
                QMessageBox.warning(self, "錯誤", "3D重建失敗：未能獲取有效的3D點雲數據")
                
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"3D重建過程中發生錯誤：{str(e)}")
    
    def create_analysis_widgets(self):
        """創建數據分析相關的部件"""
        # TODO: 添加數據分析相關的部件
        pass
    
    def init_ui(self):
        """初始化UI的其他部分"""
        self.statusBar().showMessage('就緒')
        self.show()
    
    def load_videos(self):
        """加載視頻文件"""
        # 定義相機ID和對應的友好名稱
        camera_names = {
            'l1': "左側相機1", 'l2': "左側相機2", 'l3': "左側相機3",
            'c': "中心相機",
            'r1': "右側相機1", 'r2': "右側相機2", 'r3': "右側相機3", 'r4': "右側相機4"
        }
        
        # 為每個相機選擇視頻文件
        for cam_id, cam_name in camera_names.items():
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                f"選擇{cam_name}的視頻文件",
                "",
                "視頻文件 (*.avi *.mp4 *.mkv);;所有文件 (*.*)"
            )
            if file_path:
                self.video_paths[cam_id] = file_path
            else:
                self.statusBar().showMessage(f'未選擇{cam_name}的視頻文件')
                return
        
        # 初始化視頻捕獲
        for cam_id, path in self.video_paths.items():
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.statusBar().showMessage(f'無法打開{camera_names[cam_id]}的視頻文件')
                return
            self.caps[cam_id] = cap
        
        # 啟用控制按鈕
        self.play_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.statusBar().showMessage('視頻文件加載完成')
        
        # 啟用3D重建按鈕
        self.start_3d_btn.setEnabled(True)
        
        # 顯示第一幀
        self.update_frame()
    
    def update_frame(self):
        """更新視頻幀"""
        frames = {}
        all_ret = True
        
        # 讀取所有相機的當前幀
        for cam_id, cap in self.caps.items():
            ret, frame = cap.read()
            if not ret:
                all_ret = False
                break
            frames[cam_id] = frame
        
        # 如果有任何一個視頻結束，停止播放
        if not all_ret:
            self.stop_playback()
            return
        
        # 更新顯示
        for cam_id, frame in frames.items():
            # 調整圖像大小
            frame = cv2.resize(frame, (380, 285))
            
            # 創建一個標籤區域
            label_height = 30
            label_bg = np.zeros((label_height, frame.shape[1], 3), dtype=np.uint8)
            
            # 在標籤區域添加相機ID和名稱
            camera_name = {
                'l1': "左側相機1", 'l2': "左側相機2", 'l3': "左側相機3",
                'c': "中心相機",
                'r1': "右側相機1", 'r2': "右側相機2", 'r3': "右側相機3", 'r4': "右側相機4"
            }[cam_id]
            label = f"{cam_id}: {camera_name}"
            
            # 在標籤背景上繪製文字
            cv2.putText(label_bg, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 將標籤區域與視頻幀垂直拼接
            frame_with_label = np.vstack((label_bg, frame))
            
            # 轉換顏色空間
            frame_with_label = cv2.cvtColor(frame_with_label, cv2.COLOR_BGR2RGB)
            
            # 創建QImage
            h, w, ch = frame_with_label.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_with_label.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 顯示圖像
            self.video_displays[cam_id].setPixmap(QPixmap.fromImage(qt_image))
            self.video_displays[cam_id].setMinimumSize(380, 315)  # 調整大小以適應標籤
        
        self.current_frame += 1
        self.statusBar().showMessage(f'當前幀: {self.current_frame}')
    
    def toggle_play(self):
        """切換播放/暫停狀態"""
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """開始播放"""
        self.is_playing = True
        self.play_btn.setText("暫停")
        self.timer.start(33)  # 約30 FPS
    
    def stop_playback(self):
        """停止播放"""
        self.is_playing = False
        self.play_btn.setText("播放")
        self.timer.stop()
    
    def reset_playback(self):
        """重置播放"""
        self.stop_playback()
        self.current_frame = 0
        for cap in self.caps.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.update_frame()
    
    def closeEvent(self, event):
        """關閉應用程序時的處理"""
        # 釋放視頻捕獲資源
        for cap in self.caps.values():
            cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = CameraVisibilityApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

