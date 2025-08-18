import os
import shutil
import random
from sklearn.model_selection import train_test_split

# 配置参数
dataset_dir = r"D:\datasets\dataset_Single goal\good_data\can"  # 原始数据集根目录（包含类别子文件夹）
output_dir = r"D:\datasets\dataset_Single goal\good_good_data\can"  # 输出目录（存放划分后的训练集和测试集）
train_ratio = 0.8  # 训练集比例
test_ratio = 0.2  # 测试集比例
random_seed = 42  # 随机种子（确保可复现）

# 创建输出目录结构
os.makedirs(output_dir, exist_ok=True)
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 遍历每个类别文件夹
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)

    # 跳过非文件夹（如隐藏文件）
    if not os.path.isdir(class_path):
        continue

    # 获取该类别下的所有图片文件
    img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_paths = [os.path.join(class_path, f) for f in img_files]

    # 随机划分训练集和测试集
    train_paths, test_paths = train_test_split(
        img_paths,
        train_size=train_ratio,
        test_size=test_ratio,
        random_state=random_seed
    )

    # 创建类别子文件夹
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # 复制训练集图片
    for path in train_paths:
        shutil.copy(path, os.path.join(train_class_dir, os.path.basename(path)))

    # 复制测试集图片
    for path in test_paths:
        shutil.copy(path, os.path.join(test_class_dir, os.path.basename(path)))

    print(f"类别 '{class_name}': 训练集 {len(train_paths)} 张, 测试集 {len(test_paths)} 张")

print("数据集划分完成！")