import torch
import torch.nn as nn


class MultiImageVGG(nn.Module):
    def __init__(self, num_images, image_channels=3, image_size=224):
        super(MultiImageVGG, self).__init__()
        self.num_images = num_images

        # 增强版卷积层 - 添加BN和Block5
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # 添加BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 添加BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # 添加BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 添加BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14

            # Block 5 (新增)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 添加BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7
        )

        # 全局平均池化层 - 替代展平操作
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 精简全连接层
        # 每个图像经过GAP后变为512维特征向量
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * num_images, 256),  # 大幅减少参数
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 增加dropout防止过拟合

            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(64, 6)  # 输出6个值
        )

    def forward(self, images):
        batch_size = images.size(0)
        features = []

        # 处理每个图像
        for i in range(self.num_images):
            img = images[:, i, :, :, :]
            conv_out = self.conv_layers(img)  # 形状: [batch, 512, 7, 7]

            # 应用全局平均池化
            pooled_out = self.global_avg_pool(conv_out)  # 形状: [batch, 512, 1, 1]
            pooled_out_flat = pooled_out.view(batch_size, -1)  # 形状: [batch, 512]
            features.append(pooled_out_flat)

        # 拼接所有图像的特征
        combined_features = torch.cat(features, dim=1)  # 形状: [batch, 512 * num_images]

        # 通过精简的全连接层
        output = self.fc_layers(combined_features)
        return output