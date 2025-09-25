import torch.nn as nn
import torch

class ResidualMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=3):
        super().__init__()
        layers = []
        # 构建多层MLP
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_features if i == 0 else hidden_features, hidden_features),
                nn.BatchNorm1d(hidden_features), nn.ReLU(), nn.Dropout(0.3)
            ])
        layers.append(nn.Linear(hidden_features, out_features))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# 使用方式
#self.force_head = ResidualMLP(4096, 512, 3, num_layers=3)
#self.torque_head = ResidualMLP(4096, 512, 3, num_layers=4)  # 力矩头稍深