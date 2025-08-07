# main.py
"""
功能：系统主入口模块
作用：提供命令行菜单界面，协调各功能模块的执行
关联：
  - 作为程序入口点调用其他所有模块
  - 调用capture_faces.py采集样本
  - 调用train_model.py训练模型
  - 调用recognize.py处理单张图片
  - 调用realtime_recognition.py进行实时识别
  - 依赖utils.py的依赖检查和目录创建
"""


import os
import sys
from utils import check_and_install_dependencies, create_directory
from importlib.metadata import version
from packaging.version import Version

# 检查依赖
check_and_install_dependencies()


def check_tf_installation():
    """检查TensorFlow是否可用"""
    try:
        # 使用importlib.metadata获取版本号
        tf_version = version('tensorflow')
        min_version = "2.0.0"

        if Version(tf_version) < Version(min_version):
            print(f"\n错误: 需要TensorFlow版本 >={min_version}，当前版本 {tf_version}")
            print("请执行: pip install --upgrade tensorflow")
            return False
        return True
    except ImportError:
        print("错误: TensorFlow未安装")
        print("请执行: pip install tensorflow")
        return False


def show_menu():
    """显示主菜单"""
    print("\n" + "=" * 50)
    print("人脸识别系统")
    print("=" * 50)
    print("1. 采集人脸样本")
    print("2. 训练识别模型")
    print("3. 识别单张图片")
    print("4. 实时摄像头识别")
    print("5. 退出系统")
    print("=" * 50)
    return input("请选择: ").strip()


if __name__ == "__main__":
    # 确保目录存在
    create_directory("data/raw_images")
    create_directory("data/known_faces")
    create_directory("models")
    create_directory("output")

    # 分别跟踪不同模块的加载状态
    train_loaded = False
    recognize_loaded = False
    realtime_loaded = False

    # 初始化函数引用
    train_func = None
    recognize_func = None
    realtime_func = None

    while True:
        choice = show_menu()

        if choice == '1':
            from capture_faces import capture_known_faces

            capture_known_faces()

        elif choice == '2':
            if check_tf_installation():
                if not train_loaded:
                    from train_model import train_model as train_func

                    train_loaded = True
                train_func()

        elif choice == '3':
            if check_tf_installation():
                test_image = input("请输入图片路径: ").strip()
                if not recognize_loaded:
                    from recognize import recognize_image as recognize_func

                    recognize_loaded = True
                recognize_func(test_image)

        elif choice == '4':
            if check_tf_installation():
                if not realtime_loaded:
                    from realtime_recognition import realtime_recognition

                    realtime_func = realtime_recognition
                    realtime_loaded = True
                realtime_func()

        elif choice == '5':
            print("再见!")
            sys.exit(0)

        else:
            print("无效选择")
