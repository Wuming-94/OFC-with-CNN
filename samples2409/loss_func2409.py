import torch
import torch.nn as nn

# f_pred: 网络预测向量
# f_true: 真实向量

def magnitude_loss(pred, true):
    # L2损失，用于大小
    mag_pred = torch.norm(pred, dim=1)
    mag_true = torch.norm(true, dim=1)
    loss = nn.MSELoss()(mag_pred, mag_true)
    return loss

def angle_loss(f_pred, f_true, eps=1e-8):
    # 余弦相似度损失，用于方向
    # 将向量单位化
    f_pred_unit  = f_pred / (torch.norm(f_pred, dim=1, keepdim=True) + eps)  # f_pred<1 unit vector
    f_true_unit = f_true / (torch.norm(f_true, dim=1, keepdim=True) + eps)

    cos_sim = torch.sum(f_pred_unit * f_true_unit, dim=1)  # batch 内点乘
    loss = torch.mean(1 - cos_sim) + torch.mean(torch.relu(-cos_sim))


    #theta = torch.acos(torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7))  # rad :cut cos in[-1,1],theta:[0,pi]

    #loss = torch.mean(-torch.log((cos_sim + 1) / 2 + eps))  # 范围 [0, +∞)



    return loss

class VectorLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha  # 大小损失权重
        self.beta = beta    # 角度损失权重

    def forward(self, f_pred, f_true):
        mag_loss = magnitude_loss(f_pred, f_true)
        ang_loss = angle_loss(f_pred, f_true)

        total_loss = self.alpha * mag_loss + self.beta * ang_loss
        return total_loss


