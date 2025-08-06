import os
import glob
import cv2
from utils import create_directory, get_face_detector


def capture_known_faces():
    """从原始图像中提取人脸样本"""
    name = input("请输入姓名: ").strip()
    input_dir = os.path.join("data", "raw_images", name)
    output_dir = os.path.join("data", "known_faces", name)

    create_directory(output_dir)

    if not os.path.exists(input_dir):
        print(f"错误: 目录 {input_dir} 不存在")
        print("请先放入原始图片")
        return

    try:
        face_detector = get_face_detector()
    except Exception as e:
        print(f"人脸检测器初始化失败: {str(e)}")
        return

    # 兼容不同格式的图片
    image_types = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in image_types:
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))

    if not image_paths:
        print(f"错误: {input_dir}中没有图片")
        return

    count = 0
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

        if len(faces) == 0:
            print(f"警告: {os.path.basename(img_path)} 中未检测到人脸")
            continue

        for i, (x, y, w, h) in enumerate(faces):
            face_img = frame[y:y + h, x:x + w]
            output_path = os.path.join(output_dir, f"{name}_{count}.jpg")
            cv2.imwrite(output_path, face_img)
            print(f"已保存 {os.path.basename(output_path)} (来自 {os.path.basename(img_path)} 的人脸 #{i + 1})")
            count += 1

    print(f"\n完成! 共保存{count}张{name}的人脸样本")
    print(f"样本保存在: {output_dir}")
