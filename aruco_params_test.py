import cv2
import numpy as np
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt

class ArucoParamsTester:
    def __init__(self):
        # 基礎參數設置
        self.base_params = {
            'adaptiveThreshConstant': 7,
            'minMarkerPerimeterRate': 0.08,
            'maxMarkerPerimeterRate': 0.3,
            'polygonalApproxAccuracyRate': 0.02,
            'minCornerDistanceRate': 0.04,
            'minMarkerDistanceRate': 0.1,
            'minDistanceToBorder': 3,
        }
        
        # 參數測試範圍
        self.param_ranges = {
            'adaptiveThreshConstant': [5, 6, 7, 8, 9],
            'minMarkerPerimeterRate': [0.06, 0.07, 0.08, 0.09, 0.1],
            'maxMarkerPerimeterRate': [0.25, 0.275, 0.3, 0.325, 0.35],
            'polygonalApproxAccuracyRate': [0.01, 0.015, 0.02, 0.025, 0.03],
            'minCornerDistanceRate': [0.03, 0.035, 0.04, 0.045, 0.05]
        }
        
        self.results = {}
        
    def setup_detector(self, params):
        """設置 ArUco 檢測器"""
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        detector_params = cv2.aruco.DetectorParameters()
        
        for param_name, value in params.items():
            if hasattr(detector_params, param_name):
                setattr(detector_params, param_name, value)
                
        return cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    
    def test_single_image(self, image_path, params, show_result=False):
        """測試單張圖像的檢測效果"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"無法讀取圖像: {image_path}")
            
        detector = self.setup_detector(params)
        
        # 記錄檢測時間
        start_time = time.time()
        corners, ids, rejected = detector.detectMarkers(img)
        detection_time = time.time() - start_time
        
        # 計算檢測結果
        result = {
            'detected_count': len(corners) if ids is not None else 0,
            'detection_time': detection_time,
            'perimeter_ratios': [],
            'corner_accuracy': []
        }
        
        if ids is not None:
            # 計算周長比率和角點精度
            for corner in corners:
                perimeter = cv2.arcLength(corner[0], True)
                img_perimeter = 2 * (img.shape[0] + img.shape[1])
                ratio = perimeter / img_perimeter
                result['perimeter_ratios'].append(ratio)
                
                # 計算角點精度（使用角點之間的距離標準差）
                corner_points = corner[0]
                distances = []
                for i in range(4):
                    next_i = (i + 1) % 4
                    dist = np.linalg.norm(corner_points[i] - corner_points[next_i])
                    distances.append(dist)
                result['corner_accuracy'].append(np.std(distances))
        
        # 可視化結果
        if show_result:
            vis_img = img.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
                for i, corner in enumerate(corners):
                    # 添加檢測信息
                    text_pos = tuple(corner[0][0].astype(int))
                    ratio = result['perimeter_ratios'][i]
                    cv2.putText(vis_img, f"ID:{ids[i][0]}, Ratio:{ratio:.3f}",
                              text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            cv2.imshow('Detection Result', vis_img)
            cv2.waitKey(1)
        
        return result
    
    def test_parameter_set(self, image_paths, param_name, param_values):
        """測試特定參數的不同值"""
        results = []
        base_params = self.base_params.copy()
        
        for value in param_values:
            base_params[param_name] = value
            test_results = []
            
            for img_path in image_paths:
                result = self.test_single_image(img_path, base_params)
                test_results.append(result)
            
            # 計算平均結果
            avg_result = {
                'value': value,
                'avg_detected': np.mean([r['detected_count'] for r in test_results]),
                'avg_time': np.mean([r['detection_time'] for r in test_results]),
                'avg_perimeter_ratio': np.mean([np.mean(r['perimeter_ratios']) 
                                              if r['perimeter_ratios'] else 0 
                                              for r in test_results]),
                'avg_corner_accuracy': np.mean([np.mean(r['corner_accuracy'])
                                              if r['corner_accuracy'] else float('inf')
                                              for r in test_results])
            }
            results.append(avg_result)
            
        return results
    
    def plot_parameter_results(self, param_name, results):
        """繪製參數測試結果"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        values = [r['value'] for r in results]
        
        # 檢測數量
        ax1.plot(values, [r['avg_detected'] for r in results], 'b-o')
        ax1.set_title('Average Detected Markers')
        ax1.set_xlabel(param_name)
        ax1.set_ylabel('Count')
        
        # 檢測時間
        ax2.plot(values, [r['avg_time'] for r in results], 'r-o')
        ax2.set_title('Average Detection Time')
        ax2.set_xlabel(param_name)
        ax2.set_ylabel('Time (s)')
        
        # 周長比率
        ax3.plot(values, [r['avg_perimeter_ratio'] for r in results], 'g-o')
        ax3.set_title('Average Perimeter Ratio')
        ax3.set_xlabel(param_name)
        ax3.set_ylabel('Ratio')
        
        # 角點精度
        ax4.plot(values, [r['avg_corner_accuracy'] for r in results], 'm-o')
        ax4.set_title('Corner Accuracy (lower is better)')
        ax4.set_xlabel(param_name)
        ax4.set_ylabel('Std Dev')
        
        plt.tight_layout()
        plt.show()
    
    def run_full_test(self, image_folder):
        """運行完整的參數測試"""
        image_paths = list(Path(image_folder).glob('*.jpg')) + list(Path(image_folder).glob('*.png'))
        if not image_paths:
            raise ValueError(f"在 {image_folder} 中未找到圖像文件")
        
        print(f"找到 {len(image_paths)} 張測試圖像")
        
        for param_name, values in self.param_ranges.items():
            print(f"\n測試參數: {param_name}")
            results = self.test_parameter_set(image_paths, param_name, values)
            self.results[param_name] = results
            self.plot_parameter_results(param_name, results)
            
        # 保存測試結果
        with open('aruco_params_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
            
        return self.results

def main():
    # 使用示例
    tester = ArucoParamsTester()
    
    # 設置測試圖像文件夾路徑
    image_folder = "./test_images"  # 替換為你的測試圖像文件夾路徑
    
    try:
        results = tester.run_full_test(image_folder)
        print("\n測試完成！結果已保存到 aruco_params_test_results.json")
        
        # 顯示最佳參數建議
        for param_name, param_results in results.items():
            best_result = max(param_results, key=lambda x: x['avg_detected'])
            print(f"\n{param_name} 建議值: {best_result['value']}")
            print(f"平均檢測數量: {best_result['avg_detected']:.2f}")
            print(f"平均檢測時間: {best_result['avg_time']:.3f}s")
            
    except Exception as e:
        print(f"測試過程中出現錯誤: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 