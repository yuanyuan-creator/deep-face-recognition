# deep-face-recognition
# 人脸识别系统
主程序入口main.py

基于OpenCV和TensorFlow的人脸识别系统，提供完整的人脸采集、模型训练和识别功能，支持图片识别和实时摄像头识别。

## 功能特性
- **人脸采集**：从原始图片中提取人脸样本
- **模型训练**：训练高精度人脸识别CNN模型
- **图片识别**：识别单张图片中的人脸
- **实时识别**：通过摄像头进行实时人脸识别
- **数据增强**：提高模型泛化能力
- **平衡训练**：优化不平衡数据集
- **训练可视化**：生成训练过程曲线和详细报告
- **依赖管理**：自动检测和安装缺失依赖

## 环境要求
- Python 3.6+
- 支持CUDA的GPU（推荐）
- 摄像头（用于实时识别）

## 操作指南

### 1. 克隆仓库
```bash
git clone https://github.com/your-repo/face-recognition-system.git
cd face-recognition-system

### 2. 安装依赖
```bash
pip install -r requirements.txt

### 3. 创建目录结构
```bash
python -c "from utils import create_directory; create_directory('data/raw_images'); create_directory('data/known_faces'); create_directory('models'); create_directory('output')"

### 4.准备原始图片
在data/raw_images目录下为每个人员创建独立文件夹：
data/raw_images/
    ├── 张三/
    │   ├── photo1.jpg
    │   └── photo2.png
    └── 李四/
        ├── image1.jpeg
        └── image2.jpg

### 5.采集人脸样本
运行主程序并选择选项1：输入需要采集的人员姓名，程序会自动从原始图片中提取人脸样本并保存到data/known_faces
```bash
python main.py

### 6.训练模型
当至少有两类人脸样本后，在主菜单中选择选项2开始训练：

模型将保存到models/目录
训练报告和曲线图保存到output/目录
训练过程需要一定时间，请耐心等待

### 7.进行人脸识别
单张图片识别​（选项3）：输入图片路径，结果保存为output/result.jpg
​实时摄像头识别​（选项4）：开启摄像头实时识别，按ESC键退出

### 8.查看训练报告
训练完成后在output/目录查看：

training_report.txt：训练详细报告
training_history.png：训练过程曲线


***=====项目目录结构=====***
deep-face-recognition
├── data/
│   ├── raw_images/    # 原始图片
│   └── known_faces/   # 处理后的人脸样本
├── models/            # 训练好的模型
├── output/            # 输出结果
├── utils.py           # 工具函数
├── capture_faces.py   # 人脸采集模块
├── train_model.py     # 模型训练模块
├── recognize.py       # 图片识别模块
├── realtime_recognition.py # 实时识别模块
├── main.py            # 主程序
└── requirements.txt   # 依赖列表




***=====注 意 事 项=====***
​样本要求​：
每个人员至少需要20张人脸样本才能有效训练
确保原始图片光线充足、人脸清晰
建议使用不同角度和表情的照片
​硬件要求​：
训练阶段推荐使用支持CUDA的GPU
实时识别需要摄像头支持
内存建议8GB以上
​首次运行​：
系统会自动检测并安装缺失依赖
确保网络连接正常以下载必要组件
安装完成后需要重启程序
常见问题
1.训练时报错"目录为空"
​原因​：尚未采集人脸样本或数据目录不存在
​解决方法​：

先使用"采集人脸样本"功能添加人脸数据
检查data/known_faces目录是否包含子文件夹
确保训练前至少有两个不同人的样本

2.无法加载人脸检测器
​原因​：缺少OpenCV的Haar级联文件
​解决方法​：将下载的文件放入项目根目录
```bash
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml


###程序软件环境
PyCharm Community Edition 2024.3.3
下载地址：https://www.jetbrains.com/zh-cn/pycharm/#，找到download即可
或：https://www.jetbrains.com/pycharm/download/?section=windows

###安装教程（参考知乎大神）：
https://zhuanlan.zhihu.com/p/24884367657




