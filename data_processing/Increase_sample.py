
import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# 定义增强流水线（分两步：PIL 操作 + Tensor 操作）
def get_augmentations():
    # 第一步：PIL 兼容的操作（如 Resize、Rotation、Flip）
    pil_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    ])

    # 第二步：需要 Tensor 的操作（如 RandomErasing、ColorJitter）
    tensor_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        transforms.ToPILImage(),  # 转回 PIL 以便保存
    ])

    # 合并两步
    def combined_transforms(img):
        img = pil_transforms(img)  # 先执行 PIL 操作
        img = tensor_transforms(img)  # 再执行 Tensor 操作
        return img

    return combined_transforms


# 输入/输出文件夹
for i in range(6):
    input_dir = rf"D:\datasets\dataset_Single goal\dataset_Single goal\can\{i}"
    output_dir = rf"D:\datasets\dataset_Single goal\good_data\can\{i}"
    os.makedirs(output_dir, exist_ok=True)
    num_augmentations = 5  # 每张原图生成 3 个增强版本

    for img_name in tqdm(os.listdir(input_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert("RGB")  # 确保 RGB 格式

        for i in range(num_augmentations):
            # 应用增强
            augmented_img = get_augmentations()(img)

            # 保存新文件
            name, ext = os.path.splitext(img_name)
            new_name = f"{name}_aug{i + 1}{ext}"
            save_path = os.path.join(output_dir, new_name)
            augmented_img.save(save_path)
