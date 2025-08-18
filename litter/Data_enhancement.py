import cv2
import numpy as np
import os
import random


def rotate_image_with_white_fill(image, angle):
    """
    旋转图像并用白色填充空白区域
    :param image: 输入图像 (BGR格式)
    :param angle: 旋转角度（顺时针）
    :return: 旋转后的图像
    """
    h, w = image.shape[:2]
    # 计算旋转中心
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # 计算旋转后的图像边界
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵的平移部分
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 执行旋转，用白色填充空白区域
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # 白色填充
    )

    return rotated


def augment_dataset_in_place(folder_path, angles=[-15, -10, -5, 5, 10, 15], copies=1):
    """
    直接覆盖原文件夹中的图像，进行旋转增强
    :param folder_path: 图像文件夹路径
    :param angles: 可能的旋转角度列表（顺时针）
    :param copies: 每张图像生成的增强副本数量
    """
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return

    # 遍历文件夹中的所有图像文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"无法读取图像: {img_path}")
                continue

            # 生成多个增强副本
            for i in range(copies):
                # 随机选择一个旋转角度
                angle = random.choice(angles)

                # 旋转图像并用白色填充
                rotated_image = rotate_image_with_white_fill(image, angle)

                # 直接覆盖原文件（或保存为新文件）
                # 选项1：覆盖原文件（不推荐，因为原始数据会丢失）
                # cv2.imwrite(img_path, rotated_image)

                # 选项2：保存为新文件（推荐，避免丢失原始数据）
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_rotated_{angle}_{i}{ext}"
                output_path = os.path.join(folder_path, new_filename)
                cv2.imwrite(output_path, rotated_image)
                print(f"已保存增强图像: {output_path}")


# 使用示例
if __name__ == "__main__":
    folder_path = r"D:\datasets\dataset_Single goal\dataset_Single goal\bag\fragrant and crispy"  # 替换为你的图像文件夹路径
    augment_dataset_in_place(folder_path, copies=4)  # 每张图像生成2个增强副本