import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class MultiImageDataset(Dataset):
    def __init__(self, data_dir, label_file, num_images=3, transform=None):
        """
        data_dir: Proj Path
        label_file: Force Path
        num_images: 3
        transform: Image preprocessing methods
        """
        self.data_dir = data_dir
        self.num_images = num_images
        self.transform = transform

        # read label
        self.labels = {}
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split() #
                idx = int(parts[0])  # 图片编号
                values = [float(x) for x in parts[1:]]  # Fx, Fy, Fz, Tx, Ty, Tz
                self.labels[idx] = torch.tensor(values, dtype=torch.float32)

        # 收集所有样本编号（根据已有图片）
        self.samples = []
        for fname in os.listdir(data_dir):
            if fname.endswith(".png") and "xz" in fname:  # 只用一张视图筛编号
                idx = int(fname.split("_")[0].replace("Proj", ""))
                if idx in self.labels:
                    self.samples.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]

        # 读取多张图像 (假设是 xy, yz, xz 三个视图)
        view_suffixes = ["xy", "yz", "xz"]
        images = []
        for suffix in view_suffixes[:self.num_images]:
            img_path = os.path.join(self.data_dir, f"Proj{sample_id}_{suffix}.png")
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # 3 images [num_images, channels, H, W]
        images = torch.stack(images, dim=0)

        # 标签
        label = self.labels[sample_id]  # [6]

        return images, label
