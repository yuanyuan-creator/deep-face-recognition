# utils.py
"""
功能：通用工具模块
作用：提供共享工具函数和依赖管理
关联：
  - 被所有其他模块调用
  - 提供目录创建、人脸检测器获取等基础功能
  - 实现依赖自动检查和安装
  - 包含人脸标注框绘制函数
  - 提供训练历史可视化功能
"""


import os
import sys
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt
from importlib.metadata import version


def create_directory(path):
    """创建目录（如果不存在）"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建目录: {path}")
    return path


def get_face_detector():
    """获取人脸检测器"""
    possible_paths = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_default.xml"
    ]

    for haar_path in possible_paths:
        if os.path.exists(haar_path):
            try:
                detector = cv2.CascadeClassifier(haar_path)
                # 验证检测器是否可用
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                faces = detector.detectMultiScale(test_img)
                # 只要不抛出异常就算成功
                print(f"已加载人脸检测器: {haar_path}")
                return detector
            except Exception as e:
                print(f"警告: 检测器 {haar_path} 初始化失败: {str(e)}")
                continue

    raise FileNotFoundError("错误: 找不到可用的Haar级联分类器文件\n"
                            "请从以下位置下载: "
                            "https://github.com/opencv/opencv/tree/master/data/haarcascades")


def draw_prediction(frame, x, y, w, h, text):
    """在检测到的人脸上绘制边界框和标签"""
    # 绘制人脸矩形
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 计算文本背景大小
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

    # 绘制文本背景
    cv2.rectangle(frame,
                  (x, y - text_height - 10),
                  (x + text_width, y),
                  (0, 255, 0), -1)

    # 绘制文本
    cv2.putText(frame, text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def check_and_install_dependencies():
    """检查并安装必要的依赖"""
    required_mapping = {
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'tensorflow': 'tensorflow>=2.0.0',
        'matplotlib': 'matplotlib',
        'importlib_metadata': 'importlib-metadata'
    }

    missing_packages = []

    # 检查模块是否存在
    for module, package in required_mapping.items():
        try:
            if module == 'importlib_metadata':
                # 特殊处理
                __import__('importlib.metadata')
            else:
                __import__(module)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"缺少依赖: {', '.join(missing_packages)}")
        install = input("是否自动安装? (y/n): ").strip().lower()
        if install == 'y':
            print("正在安装...")
            try:
                command = [sys.executable, '-m', 'pip', 'install'] + missing_packages
                subprocess.check_call(command)
                print("依赖安装成功! 请重启程序")
                sys.exit(0)
            except Exception as e:
                print(f"安装失败: {str(e)}")
                print("请手动安装: "
                      f"pip install {' '.join(missing_packages)}")
                sys.exit(1)
        else:
            print("请手动安装依赖后重新运行程序")
            sys.exit(1)


def plot_training_history(history):
    """绘制训练历史曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    plt.figure(figsize=(12, 5))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.ylabel('准确率')
    plt.xlabel('训练轮次')
    plt.legend()
    plt.grid(True)

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('训练轮次')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # 保存图像
    output_path = "output/training_history.png"
    plt.savefig(output_path)
    print(f"训练历史曲线已保存到 {output_path}")
    plt.close()
