
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, image_channels=3, image_size=224):
        super(VGG, self).__init__()

        # 增强版卷积层 - 添加BN和Block5
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 添加BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # 添加BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 添加BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 添加BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14
        )

        # 计算卷积层输出尺寸
        conv_output_size = image_size // (2 ** 4)  # 4次pooling，每次/2
        self.flattened_size = 512 * conv_output_size * conv_output_size

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 6)
        )

    def forward(self, x):
        # 直接通过卷积层
        conv_out = self.conv_layers(x)

        # 通过全连接层
        output = self.fc_layers(conv_out)
        return output