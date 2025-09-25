# compute_dataset_min_max.py
import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def compute_min_max(base_data_path, shapes, image_size=(224, 224)):
    """
    遍历数据集，计算通道最小值和最大值
    """
    # 初始化为极值
    min_vals = torch.full((3,), float('inf'))
    max_vals = torch.full((3,), float('-inf'))

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()  # [0,1] 范围
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

                # 通道最小值和最大值
                min_vals = torch.min(min_vals, img_tensor.view(3, -1).min(dim=1).values)
                max_vals = torch.max(max_vals, img_tensor.view(3, -1).max(dim=1).values)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"\nDataset Min: {min_vals}")
    print(f"Dataset Max: {max_vals}")

    return min_vals, max_vals


if __name__ == "__main__":
    base_data_path = "/home/wmzheng/PycharmProjects/pythonProject1/Data/"
    shapes = ["pyramid", "cube", "spherical"]

    min_vals, max_vals = compute_min_max(base_data_path, shapes)

    # 保存为 .pt 文件
    save_path = "dataset_min_max.pt"
    torch.save({'min': min_vals, 'max': max_vals}, save_path)
    print(f"Min and max saved to {save_path}")
