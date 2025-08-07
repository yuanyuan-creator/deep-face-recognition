# train_model.py
"""
功能：模型训练模块
作用：使用采集的人脸样本训练识别模型
关联：
  - 被main.py的选项2调用
  - 处理capture_faces.py采集的样本
  - 生成模型文件供recognize.py和realtime_recognition.py使用
  - 调用utils.py的目录创建和训练历史绘制函数
  - 输出模型到models目录，报告到output目录
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from utils import create_directory, plot_training_history
from sklearn.model_selection import train_test_split
from collections import Counter

# 模型配置
IMG_SIZE = 160  # 输入尺寸
BATCH_SIZE = 16  # 批次大小
EPOCHS = 100  # 训练轮次
PATIENCE = 10  # 早停耐心值
VALIDATION_SPLIT = 0.2  # 验证集比例
MIN_SAMPLES_PER_CLASS = 20  # 每个类别最小样本数


def create_model(input_shape, num_classes):
    """创建CNN人脸识别模型"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def load_dataset():
    """加载并平衡人脸数据集"""
    X, y = [], []
    label_map = {}
    base_dir = "data/known_faces"

    if not os.path.exists(base_dir):
        raise ValueError(f"错误: 目录 {base_dir} 不存在")

    if not os.listdir(base_dir):
        raise ValueError(f"错误: 目录 {base_dir} 为空")

    # 创建标签映射
    persons = [d for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))]

    if len(persons) < 2:
        raise ValueError("错误: 至少需要两个不同人的样本进行训练")

    for label, person in enumerate(persons):
        label_map[label] = person

    # 加载图像数据
    print("\n加载人脸样本...")
    class_counts = Counter()

    for label, person in enumerate(persons):
        person_dir = os.path.join(base_dir, person)
        person_samples = 0

        for img_file in os.listdir(person_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(person_dir, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(label)
                    person_samples += 1
                else:
                    print(f"警告: 无法读取图像 {img_path}")

        class_counts[label] = person_samples
        print(f"  {person}: {person_samples} 个样本")

        if person_samples < MIN_SAMPLES_PER_CLASS:
            print(f"  警告: {person} 样本不足，建议至少提供 {MIN_SAMPLES_PER_CLASS} 个样本")

    if not X:
        raise ValueError("错误: 未找到人脸样本")

    # 平衡数据集
    print("\n平衡数据集...")
    min_samples = min(class_counts.values())
    balanced_X, balanced_y = [], []

    for label in class_counts:
        # 获取该类别所有样本索引
        indices = [i for i, lbl in enumerate(y) if lbl == label]

        # 如果样本数量超过最小值，进行欠采样
        if len(indices) > min_samples:
            indices = np.random.choice(indices, min_samples, replace=False)

        # 添加到平衡数据集
        balanced_X.extend([X[i] for i in indices])
        balanced_y.extend([y[i] for i in indices])

    # 打乱数据集
    indices = np.arange(len(balanced_X))
    np.random.shuffle(indices)
    balanced_X = np.array([balanced_X[i] for i in indices])
    balanced_y = np.array([balanced_y[i] for i in indices])

    total_samples = len(balanced_X)
    print(f"成功加载 {total_samples} 个样本，{len(label_map)} 个人")
    print(f"平衡后每个类别样本数: {min_samples}")

    return balanced_X, balanced_y, label_map


def train_model():
    """训练并保存模型"""
    create_directory("models")

    print("\n=== 开始训练人脸识别模型 ===")
    try:
        X, y, label_map = load_dataset()
    except ValueError as e:
        print(str(e))
        print("请先运行 '采集人脸样本' 功能")
        return

    # 数据预处理
    X = X.astype('float32') / 255.0

    # 创建数据增强生成器
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT
    )

    # 创建模型
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    num_classes = len(label_map)
    model = create_model(input_shape, num_classes)

    print("\n模型架构:")
    model.summary()

    # 训练配置
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "models/face_recognition_best.h5",
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            verbose=1
        )
    ]

    # 开始训练
    print("\n训练模型...")
    try:
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=VALIDATION_SPLIT,
            stratify=y,
            random_state=42
        )

        # 使用数据增强训练模型
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=len(X_train) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
    except Exception as e:
        print(f"\n训练出错: {str(e)}")
        return

    # 保存最终模型
    model.save("models/face_recognition_model.h5")
    np.save("models/label_map.npy", label_map)

    # 评估模型
    val_acc = max(history.history['val_accuracy'])
    print(f"\n训练完成! 最佳验证准确率: {val_acc:.2%}")

    # 绘制训练曲线
    plot_training_history(history)

    import datetime  # 添加这行导入

    # 保存训练报告
    with open("output/training_report.txt", "w") as f:
        f.write("=== 人脸识别模型训练报告 ===\n")
        # 使用标准的 datetime 模块获取当前时间
        f.write(f"训练日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"样本数量: {len(X)} (平衡后)\n")
        f.write(f"类别数量: {len(label_map)}\n")
        f.write(f"最佳验证准确率: {val_acc:.2%}\n")
        f.write("\n类别分布:\n")
        for label, name in label_map.items():
            f.write(f"  {name}: {np.sum(y == label)} 个样本\n")

    print(f"\n训练报告已保存到 output/training_report.txt")
    print(f"模型已保存到 models/ 目录")
