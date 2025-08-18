import os
from torch.nn.functional import one_hot
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
from tqdm import tqdm
import timm
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MNIST_Dataset(Dataset):
    def __init__(self, root, isTrain=True):
        self.datasets = []
        sub_dir = 'train' if isTrain else 'test'
        for target in os.listdir(os.path.join(root, sub_dir)):
            target_path = os.path.join(root, sub_dir, target)
            for img_name in os.listdir(target_path):
                img_path = os.path.join(target_path, img_name)
                self.datasets.append((img_path, target))  # 图片路径，类别

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        img_path, target = self.datasets[index]
        img = cv2.imread(img_path)  # 默认读取3通道（BGR）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = transform(img)
        label = int(target)  # 直接返回整数标签，one-hot在训练时处理
        return img, label


class ArcFace(nn.Module):
    def __init__(self, feature_num, cls_num, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(feature_num, cls_num))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        # 归一化特征和权重
        x_norm = F.normalize(x, dim=1)
        weight_norm = F.normalize(self.weight, dim=0)

        # 计算余弦相似度
        cos_theta = torch.matmul(x_norm, weight_norm)

        # 添加角度边际
        theta = torch.arccos(cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        theta_m = theta + self.m
        cos_theta_m = torch.cos(theta_m)

        # 难例挖掘：只对正类添加边际
        one_hot_label = F.one_hot(label, num_classes=self.weight.shape[1]).float()
        logits = self.s * (one_hot_label * cos_theta_m + (1.0 - one_hot_label) * cos_theta)

        return logits


def calculate_intra_inter_distance(features, labels, metric='distance'):
    """
    计算平均类内距离和类间距离（基于余弦相似度）
    """
    # 转换为numpy数组
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # 计算余弦相似度矩阵
    cosine_sim_matrix = cosine_similarity(features)  # [n_samples, n_samples]

    intra_dis = []  # 存储每个类的类内距离/相似度
    inter_dis = []  # 存储每个类的类间距离/相似度

    for class_label in np.unique(labels):
        # 获取当前类的样本索引
        class_indices = np.where(labels == class_label)[0]

        # 计算类内距离/相似度
        class_similarities = cosine_sim_matrix[class_indices][:, class_indices]
        triu_indices = np.triu_indices(len(class_indices), k=1)  # 排除对角线
        intra_values = class_similarities[triu_indices]

        # 计算类间距离/相似度
        other_class_indices = np.where(labels != class_label)[0]
        if len(other_class_indices) > 0:
            inter_similarities = cosine_sim_matrix[class_indices][:, other_class_indices]
            inter_values = inter_similarities.flatten()

            # 根据metric参数选择输出距离（1-相似度）或相似度
            if metric == 'distance':
                intra_dis.append(np.mean(1 - intra_values))
                inter_dis.append(np.mean(1 - inter_values))
            elif metric == 'similarity':
                intra_dis.append(np.mean(intra_values))
                inter_dis.append(np.mean(inter_values))

    # 计算全局平均值
    avg_intra_dis = np.mean(intra_dis) if intra_dis else 0.0
    avg_inter_dis = np.mean(inter_dis) if inter_dis else 0.0

    return avg_intra_dis, avg_inter_dis


class EfficientNetB5FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('efficientnet_b5', pretrained=True, features_only=True)
        # 添加一个适配器，将多尺度特征合并为单一特征（例如取最后一层）
        self.adapter = nn.AdaptiveAvgPool2d(1)  # 全局平均池化

    def forward(self, x):
        features_list = self.model(x)  # 返回多尺度特征列表
        features = features_list[-1]   # 取最后一层特征（最高语义层次）
        features = self.adapter(features)  # [B, C, 1, 1]
        features = features.view(features.size(0), -1)  # [B, C]
        return features


class Trainer:
    def __init__(self, model_path, data_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = EfficientNetB5FeatureExtractor().to(self.device)
        if model_path is not None and os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path))
            print('模型已加载')

        # 修改：ArcFace现在直接处理原始特征，不单独优化
        self.arc = ArcFace(feature_num=512, cls_num=7).to(self.device)
        self.opt = torch.optim.Adam(params=self.net.parameters(), lr=1e-3)
        # 使用较小的学习率用于ArcFace权重
        self.opt_arc = torch.optim.SGD(self.arc.parameters(), lr=1e-2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=20, gamma=0.5)
        self.scheduler_arc = torch.optim.lr_scheduler.StepLR(self.opt_arc, step_size=20, gamma=0.5)
        # 修改：使用CrossEntropyLoss配合ArcFace的logits
        self.loss_fn = nn.CrossEntropyLoss()

        train_data = MNIST_Dataset(root=data_path, isTrain=True)
        self.train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

        test_data = MNIST_Dataset(root=data_path, isTrain=False)
        self.test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

    def train(self, epoch):
        self.net.train()
        sum_loss = 0
        count = 0
        for images, labels in tqdm(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # 获取原始特征
            features = self.net(images)


            # 通过ArcFace计算logits
            logits = self.arc(features, labels)

            # 计算损失
            loss = self.loss_fn(logits, labels)

            # 反向传播
            self.opt.zero_grad()
            self.opt_arc.zero_grad()
            loss.backward()
            self.opt.step()
            self.opt_arc.step()

            sum_loss += loss.item()
            count += 1

        avg_loss = sum_loss / count
        print(f'epoch:{epoch}, avg_loss:{avg_loss}')

    def train_test(self, epoch):
        self.net.eval()
        sum_test_loss = 0
        test_count = 0
        sum_acc = 0


        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # 获取特征
                features = self.net(images)

                # 保存特征和标签用于距离计算
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())

                # 通过ArcFace计算logits
                logits = self.arc(features, labels)

                # 计算损失
                loss = self.loss_fn(logits, labels)

                # 计算准确度
                pred = torch.argmax(logits, dim=1)
                acc = torch.mean((pred == labels).float())

                sum_acc += acc.item()
                sum_test_loss += loss.item()
                test_count += 1

            # 合并所有特征和标签
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # 计算类内和类间距离
            intra, inter = calculate_intra_inter_distance(all_features, all_labels)

            avg_test_loss = sum_test_loss / test_count
            avg_acc = sum_acc / test_count
            print(f'epoch:{epoch}, avg_test_loss:{avg_test_loss}, avg_acc:{avg_acc}, '
                  f'avg_intra_dis:{intra}, avg_inter_dis:{inter}')


if __name__ == '__main__':
    for data_name in os.listdir(r'D:\datasets\dataset_Single goal\good_good_data'):
        data_path = os.path.join(r'D:\datasets\dataset_Single goal\good_good_data',data_name)
        trainer = Trainer(model_path=None, data_path=data_path)
        for epoch in range(500):
            trainer.train(epoch)
            trainer.train_test(epoch)
            # 保存模型
            torch.save(trainer.net.state_dict(), f'best_{data_name}_net.pt')