# recognize.py
"""
功能：单张图片识别模块
作用：对输入图片进行人脸检测和识别
关联：
  - 被main.py的选项3调用
  - 加载train_model.py生成的模型和标签映射
  - 使用utils.py的人脸检测器和绘制函数
  - 与train_model.py共享IMG_SIZE参数
  - 结果保存到output目录
"""


import os
import cv2
import numpy as np
import tensorflow as tf
from utils import get_face_detector, draw_prediction
from train_model import IMG_SIZE  # 导入训练时使用的尺寸


def recognize_image(image_path):
    """识别单张图片中的人脸"""
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 {image_path}")
        return

    # 加载资源
    try:
        model_path = "models/face_recognition_model.h5"
        label_map_path = "models/label_map.npy"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"标签映射文件不存在: {label_map_path}")

        model = tf.keras.models.load_model(model_path)
        label_map = np.load(label_map_path, allow_pickle=True).item()
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        print("请先训练模型")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"错误: 无法读取图像 {image_path}")
        return

    # 创建原始图像的副本
    result_frame = frame.copy()

    # 人脸检测
    face_detector = get_face_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

    if len(faces) == 0:
        print("警告: 未检测到人脸，保存原始图片")
        # 保存结果
        output_path = "output/result.jpg"
        cv2.imwrite(output_path, result_frame)
        print(f"结果已保存: {output_path}")
        return

    # 识别每个人脸
    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        # 使用训练时相同的尺寸
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_img = np.expand_dims(face_img, axis=0) / 255.0

        preds = model.predict(face_img, verbose=0)
        pred_label = np.argmax(preds)
        confidence = np.max(preds) * 100

        name = label_map.get(pred_label, "Unknown")
        draw_prediction(result_frame, x, y, w, h, f"{name} {confidence:.1f}%")

    # 保存结果
    output_path = "output/result.jpg"
    cv2.imwrite(output_path, result_frame)
    print(f"结果已保存: {output_path}")

    # 显示结果
    cv2.imshow('识别结果', result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
