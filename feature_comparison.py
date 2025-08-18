import cv2
import torch
from torchvision import transforms
import numpy as np
import json
import os
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
# from net3 import FeatureNet
from net import EfficientNetB5FeatureExtractor
from numpy.linalg import norm
import threading

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径配置（建议使用绝对路径）
MODEL_PATH = {
    'bag': os.path.join('D:/projects/PythonProject/smart supermarket', 'pt/best_bag_net.pt'),
    'bottle': os.path.join('D:/projects/PythonProject/smart supermarket', 'pt/best_bottle_net.pt'),
    'box': os.path.join('D:/projects/PythonProject/smart supermarket', 'pt/best_box_net.pt'),
    'can': os.path.join('D:/projects/PythonProject/smart supermarket', 'pt/best_can_net.pt')
}

FEATURE_BANK_PATH = {
    'bag': os.path.join('D:/projects/PythonProject/smart supermarket', 'bag_feature_bank.json'),
    'bottle': os.path.join('D:/projects/PythonProject/smart supermarket', 'bottle_feature_bank.json'),
    'box': os.path.join('D:/projects/PythonProject/smart supermarket', 'box_feature_bank.json'),
    'can': os.path.join('D:/projects/PythonProject/smart supermarket', 'can_feature_bank.json')
}

# 预处理变换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 模型缓存
model_cache: Dict[str, EfficientNetB5FeatureExtractor] = {}

def get_model(big_label: str) -> EfficientNetB5FeatureExtractor:
    """加载或从缓存获取模型"""
    if big_label not in MODEL_PATH:
        raise ValueError(f"无效的商品类别: {big_label}。可用类别: {list(MODEL_PATH.keys())}")

    if big_label in model_cache:
        return model_cache[big_label]

    try:
        model = EfficientNetB5FeatureExtractor().to(device)
        model.load_state_dict(torch.load(MODEL_PATH[big_label], map_location=device))
        model.eval()
        model_cache[big_label] = model
        return model
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {str(e)}")


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    """预处理图像"""
    if img.ndim == 2:  # 灰度图转RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA转RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return transform(img).unsqueeze(0).to(device)

def extract_and_save_feature(img: np.ndarray, label: str, model: EfficientNetB5FeatureExtractor, feature_bank: dict):
    """提取单张图像的特征并保存到特征库"""
    input_tensor = preprocess_image(img)
    with torch.no_grad():
        feature = model(input_tensor)
        feature = feature / torch.norm(feature, dim=1, keepdim=True)  # L2归一化
        feature_list = feature.cpu().tolist()

        if label in feature_bank:
            feature_bank[label].extend(feature_list)
        else:
            feature_bank[label] = feature_list

def get_features(imgs: List[np.ndarray], label: str, big_label: str) -> None:
    """提取图像特征并保存到特征库"""
    if not imgs:
        raise ValueError("图像列表不能为空")

    model = get_model(big_label)

    # 确保特征库目录存在
    os.makedirs(os.path.dirname(FEATURE_BANK_PATH[big_label]), exist_ok=True)

    # 加载或初始化特征库
    feature_bank = {}
    if os.path.exists(FEATURE_BANK_PATH[big_label]):
        try:
            with open(FEATURE_BANK_PATH[big_label], "r") as f:
                feature_bank = json.load(f)
        except json.JSONDecodeError:
            print(f"警告: {FEATURE_BANK_PATH[big_label]} 不是有效的JSON文件，将创建新文件")

    # 使用多线程提取特征
    with ThreadPoolExecutor() as executor:
        # 使用线程池提取每张图片的特征
        futures = [executor.submit(extract_and_save_feature, img, label, model, feature_bank) for img in imgs]

        # 等待所有线程执行完成
        for future in futures:
            future.result()

    # 保存特征库
    with open(FEATURE_BANK_PATH[big_label], "w") as f:
        json.dump(feature_bank, f, indent=4)

    print(f"成功注册 {len(imgs)} 张图像到类别 '{label}'")



def load_features(big_label: str) -> Tuple[List[str], List[np.ndarray]]:
    """加载特征库"""
    if big_label not in FEATURE_BANK_PATH:
        raise ValueError(f"无效的商品类别: {big_label}")

    if not os.path.exists(FEATURE_BANK_PATH[big_label]):
        return [], []

    try:
        with open(FEATURE_BANK_PATH[big_label], "r") as f:
            feature_bank = json.load(f)

        labels = []
        features_list = []
        for label, features in feature_bank.items():
            labels.append(label)
            features_list.append(np.array(features))

        return labels, features_list
    except Exception as e:
        raise RuntimeError(f"加载特征库失败: {str(e)}")


def calculate_similarity(feat: np.ndarray, db_feat: np.ndarray) -> float:
    """计算两张图片特征的余弦相似度"""
    return np.dot(feat, db_feat)


def get_imgs(img: np.ndarray, big_label: str, threshold: float = 0.8) -> str:
    """识别图像所属类别（使用余弦相似度和投票机制）"""
    if big_label not in MODEL_PATH:
        return "未知商品类别"

    try:
        # 预处理图像
        input_tensor = preprocess_image(img)

        # 加载模型
        model = get_model(big_label)

        # 提取特征
        with torch.no_grad():
            feat = model(input_tensor).cpu().detach().numpy()
            feat = feat.flatten() / np.linalg.norm(feat)  # 展平并归一化特征向量

        # 加载特征库
        db_labels, db_features = load_features(big_label)
        if not db_labels:
            return "未知商品（特征库为空）"

        # 使用多线程计算相似度
        similarities = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(calculate_similarity, feat, np.array(f).flatten()) for features in db_features
                       for f in features]

            for future in futures:
                similarities.append(future.result())

        # 获取相似度结果并排序
        similarities = [(label, sim) for label, sim in zip(db_labels, similarities)]
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 检查第7个特征的相似度是否低于阈值
        if len(similarities) >= 7 and similarities[6][1] < threshold:
            return "未知商品"

        # 取前7个最相似的标签（如果不足7个则取全部）
        top_labels = [label for label, sim in similarities[:7]]

        # 使用Counter找出最常见的标签
        label_counts = Counter(top_labels)
        most_common_label = label_counts.most_common(1)[0][0]

        return most_common_label

    except Exception as e:
        print(f"识别过程中出错: {str(e)}")
        return "识别失败"

if __name__ == '__main__':
    # 示例用法
        imgs=[]
        path=r'D:\datasets\dataset_Single goal\dataset_Single goal\box\3'
        for img_name in os.listdir(path):
            img_path=os.path.join(path,img_name)
            img=cv2.imread(img_path)
            imgs.append(img)
            get_features(imgs,'Q帝','box')