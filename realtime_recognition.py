import cv2
import numpy as np
import tensorflow as tf
from utils import get_face_detector, draw_prediction
from train_model import IMG_SIZE


def realtime_recognition():
    """实时摄像头人脸识别"""
    # 加载资源
    try:
        model = tf.keras.models.load_model("models/face_recognition_model.h5")
        label_map = np.load("models/label_map.npy", allow_pickle=True).item()
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        print("请先训练模型")
        return

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法访问摄像头")
        return

    face_detector = get_face_detector()

    print("\n实时识别已启动...")
    print("按 'ESC' 键退出")  # 修改这里的提示文字

    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取摄像头画面")
            break

        # 创建显示画面的副本
        display_frame = frame.copy()

        # 在画面左上角添加退出提示
        cv2.putText(display_frame, "Press ESC to exit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

        # 识别每个人脸
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            face_img = np.expand_dims(face_img, axis=0) / 255.0

            preds = model.predict(face_img, verbose=0)
            pred_label = np.argmax(preds)
            confidence = np.max(preds) * 100

            name = label_map.get(pred_label, "Unknown")
            draw_prediction(display_frame, x, y, w, h, f"{name} {confidence:.1f}%")

        # 显示画面
        cv2.imshow('人脸识别 - 实时模式', display_frame)

        # 检测退出键 (ESC)
        key = cv2.waitKey(1)
        if key == 27 :  # 27 是 ESC 键
            print("检测到退出按键")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("实时识别已结束")
