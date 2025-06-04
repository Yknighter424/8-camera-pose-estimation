print("測試基本庫...")
import cv2
import numpy as np
import matplotlib.pyplot as plt
print("基本庫導入成功！")

print("\n測試 TensorFlow...")
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
print("TensorFlow 導入成功！")

print("\n測試 SciPy...")
from scipy.signal import savgol_filter
from scipy.optimize import minimize
print("SciPy 導入成功！")

print("\n測試 Pandas...")
import pandas as pd
print("Pandas 導入成功！")

print("\n測試 PyVista...")
import pyvista as pv
from pyvistaqt import BackgroundPlotter
print("PyVista 導入成功！")

print("\n測試 MediaPipe...")
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
print("MediaPipe 導入成功！")

print("\n所有庫都已成功導入！") 