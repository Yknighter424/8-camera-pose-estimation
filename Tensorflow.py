import sys
import subprocess

# 檢查已安裝的包
def check_tensorflow():
    # 獲取已安裝的包列表
    installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'list'])
    print("已安裝的包：")
    print(installed_packages.decode())
    
    # 特別檢查 tensorflow
    try:
        import tensorflow as tf
        print(f"\nTensorFlow 版本: {tf.__version__}")
        print("TensorFlow 已正確安裝")
    except ImportError:
        print("\nTensorFlow 未安裝")

check_tensorflow()