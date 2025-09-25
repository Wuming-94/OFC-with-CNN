# compute_dataset_mean_std.py
import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def compute_mean_std(base_data_path, shapes, image_size=(224, 224)):
    """
    逐张图像计算数据集的通道均值和标准差
    """
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    num_pixels_total = 0

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()  # [0,1]
    ])

    for shape in shapes:
        data_dir = os.path.join(base_data_path, shape)
        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} does not exist")
            continue

        files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        print(f"Processing {shape}, {len(files)} images...")

        for fname in tqdm(files, desc=f"{shape}"):
            img_path = os.path.join(data_dir, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)  # [C,H,W]
                img.close()

                c, h, w = img_tensor.shape
                num_pixels = h * w

                channel_sum += img_tensor.sum(dim=(1, 2))
                channel_sum_sq += (img_tensor ** 2).sum(dim=(1, 2))
                num_pixels_total += num_pixels
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    mean = channel_sum / num_pixels_total
    std = (channel_sum_sq / num_pixels_total - mean ** 2).sqrt()

    print(f"\nDataset Mean: {mean}")
    print(f"Dataset Std: {std}")

    return mean, std


if __name__ == "__main__":
    base_data_path = "/home/wmzheng/PycharmProjects/pythonProject1/Data/"
    shapes = ["pyramid", "cube", "spherical"]

    mean, std = compute_mean_std(base_data_path, shapes)

    # 保存为 .pt 文件
    save_path = "dataset_mean_std.pt"
    torch.save({'mean': mean, 'std': std}, save_path)
    print(f"Mean and std saved to {save_path}")
