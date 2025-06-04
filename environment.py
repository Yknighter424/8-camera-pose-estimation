# 檢查系統環境
import sys
import platform
import os
import torch
import cv2
import numpy as np

def check_environment():
    """檢查並打印運行環境信息"""
    
    print("=== 基本系統信息 ===")
    print(f"操作系統: {platform.platform()}")
    print(f"Python版本: {sys.version}")
    
    print("\n=== CUDA環境 ===")
    # 檢查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"當前GPU設備: {torch.cuda.get_device_name(0)}")
        print(f"GPU數量: {torch.cuda.device_count()}")
        print(f"當前GPU內存使用情況: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        
        # GPU 性能測試
        try:
            # 創建測試用張量
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            
            # 計時器
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            # 執行測試
            start.record()
            z = torch.matmul(x, y)
            end.record()
            
            # 同步並獲取時間
            torch.cuda.synchronize()
            print(f"GPU 矩陣乘法測試時間：{start.elapsed_time(end):.2f} 毫秒")
        except Exception as e:
            print(f"GPU 測試失敗: {str(e)}")
    else:
        print("CUDA 不可用，請檢查：")
        print("1. NVIDIA 驅動是否正確安裝")
        print("2. CUDA 工具包是否正確安裝")
        print("3. PyTorch 是否為 CUDA 版本")
        print("4. 環境變量是否正確設置")
    
    print("\n=== 主要庫版本 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"OpenCV版本: {cv2.__version__}")
    print(f"NumPy版本: {np.__version__}")
    
    print("\n=== 環境變量 ===")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '未設置')}")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', '未設置')}")
    
    # 檢查可用內存
    try:
        import psutil
        vm = psutil.virtual_memory()
        print("\n=== 系統內存 ===")
        print(f"總內存: {vm.total/1024**3:.2f} GB")
        print(f"可用內存: {vm.available/1024**3:.2f} GB")
        print(f"內存使用率: {vm.percent}%")
    except ImportError:
        print("\n無法獲取系統內存信息 (需要安裝 psutil)")

if __name__ == "__main__":
    check_environment()