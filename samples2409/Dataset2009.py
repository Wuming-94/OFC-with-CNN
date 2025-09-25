import os
import torch
from torch.utils.data import Dataset, ConcatDataset,random_split
from PIL import Image
import torchvision.transforms as transforms

class DS(Dataset):  #inheritance from pytorch Dataset: length __len__() return __getitem__(idx)
    def __init__(self, data_dir, label_file, transform=None):
        """
        data_dir: Proj Path
        label_file: Force Path
        num_images: 3
        transform: Image preprocessing methods
        label_min, label_max, label_range: 用于归一化的参数，从训练集计算
        """
        self.data_dir = data_dir
        self.transform = transform

        # read label
        self.labels = {}
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                idx = int(parts[0])  # 图片编号
                values = [float(x) for x in parts[1:]]  # Fx, Fy, Fz, Tx, Ty, Tz
                self.labels[idx] = torch.tensor(values, dtype=torch.float32)

        # samples num
        self.sample_ids = []
        for fname in os.listdir(data_dir):
            if fname.endswith(".png") and "xz" in fname:
                idx = int(fname.split("_")[0].replace("Proj", ""))
                if idx in self.labels:
                    self.sample_ids.append(idx) #[1,10000]

        # preloading
        self.loaded_images = []  # 加载的图像
        self.loaded_labels = []  # 加载的标签

        print(f"Preloading {len(self.sample_ids)} samples from {data_dir}...")

        for sample_id in self.sample_ids:
            # read images
            image_set = []
            for suffix in ["xy", "yz", "xz"]:
                img_path = os.path.join(self.data_dir, f"Proj{sample_id}_{suffix}.png")
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)  # tensor: [1, H, W] (单通道) [0,1]
                #img = img * 2 - 1  #   [-1,1]
                #img = img * 51   # 数值缩放 [0,51]
                image_set.append(img)  # 3 tensor: [1, H, W]

            # 合并为三通道图像
            combined_image = torch.cat(image_set, dim=0)  # tensor: [3, H, W]

            # 存储到内存中
            self.loaded_images.append(combined_image)
            self.loaded_labels.append(self.labels[sample_id])

        self.loaded_images = torch.stack(self.loaded_images, dim=0)
        self.loaded_labels = torch.stack(self.loaded_labels, dim=0)
        self.normalized_labels = None
        print(f"Finished preloading {len(self.loaded_images)} images")

    def normalize_labels(self, label_min=None, label_max=None, label_range=None):
            if label_min is not None and label_range is not None:
                self.normalized_labels = 2 * (self.loaded_labels - label_min) / label_range - 1 #[-1,1]
                print(f"Labels normalized using provided min/max")
                print(f"Normalized labels - Min: {self.normalized_labels.min().item():.6f}, "
                      f"Max: {self.normalized_labels.max().item():.6f}, ")
            else:
                self.normalized_labels = self.loaded_labels.clone()
                print(f"No normalization applied")

    def __len__(self):
            return len(self.loaded_images)

    def __getitem__(self, idx):
        if self.normalized_labels is not None:
            return self.loaded_images[idx], self.normalized_labels[idx]
        else:
            return self.loaded_images[idx], self.loaded_labels[idx]

def load_split_ds(base_data_path, shapes, transform=None,
                      train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):

        datasets = []
        for shape in shapes:
            data_dir = os.path.join(base_data_path, shape)
            label_file = os.path.join(base_data_path, f"{shape}_op.txt")
            if os.path.exists(data_dir) and os.path.exists(label_file):
                dataset = DS(data_dir, label_file, transform=transform)
                datasets.append(dataset)
            else:
                print(f"Warning: Data not found for {shape}")

        if not datasets:
            raise ValueError("No datasets loaded")

        combined_dataset = ConcatDataset(datasets)
        total_size = len(combined_dataset)

        # divide dataset
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            combined_dataset, [train_size, val_size, test_size]
        )

        # calc minmax from train dataset
        all_train_labels = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))], dim=0)
        label_min = all_train_labels.min(dim=0).values
        label_max = all_train_labels.max(dim=0).values
        label_range = label_max - label_min
        print(f"label_trmin:{label_min}")
        print(f"label_trmax:{label_max}")


        # norm for all dataset
        for ds in datasets:
        #for i, ds in enumerate(datasets):
            #print(f"===> Normalizing dataset {i}")
            ds.normalize_labels(label_min, label_max, label_range)

        return train_dataset, val_dataset, test_dataset, label_min, label_max, label_range

# 使用示例
if __name__ == "__main__":
    # 定义图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    #
    shapes = ["spherical"]  #
    num=2
    plotname = f"{shapes[0]}2"
    print(f"{plotname}_{num}")

    # 加载并划分数据集
    train_dataset, val_dataset, test_dataset, label_min, label_max, label_range = load_split_ds(
        base_data_path="/home/wmzheng/PycharmProjects/pythonProject1/Data/",
        shapes=shapes,
        transform=transform,
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Label min: {label_min}")
    print(f"Label max: {label_max}")


    def subset_labels_minmax(subset, name="Subset"):
        all_labels = []
        for i in range(len(subset)):
            _, label = subset[i]  # 通过 __getitem__ 获取标签
            all_labels.append(label)
        all_labels = torch.stack(all_labels, dim=0)
        print(f"{name} labels min: {all_labels.min().item():.6f}, max: {all_labels.max().item():.6f}")


    # 调用
    subset_labels_minmax(train_dataset, "Train set")
    subset_labels_minmax(val_dataset, "Validation set")
    subset_labels_minmax(test_dataset, "Test set")

    # check
    print("\nChecking normalization for each dataset:")

    # train
    print("\nTraining set samples (normalized labels):")
    for i in range(min(3, len(train_dataset))):  # 检查前3个样本
        _, label = train_dataset[i]
        print(f"Sample {i}: {label}")


    # val
    print("\nValidation set samples (normalized labels):")
    for i in range(min(3, len(val_dataset))):  # 检查前3个样本
        _, label = val_dataset[i]
        print(f"Sample {i}: {label}")

    # test
    print("\nTest set samples (normalized labels):")
    for i in range(min(3, len(test_dataset))):  # 检查前3个样本
        _, label = test_dataset[i]
        print(f"Sample {i}: {label}")


    # verify [0, 1]
    print("\nVerifying all labels are in [-1, 1] range:")


    def check_range(dataset, name):
        min_val = float('inf')
        max_val = float('-inf')


        for i in range(len(dataset)):
            _, label = dataset[i]
            min_val = min(min_val, label.min().item())
            max_val = max(max_val, label.max().item())

        print(f"{name}: min={min_val:.6f}, max={max_val:.6f}, in_range={min_val >= -1 and max_val <= 1}")
        return min_val >= 0 and max_val <= 1


    train_in_range = check_range(train_dataset, "Training set")
    val_in_range = check_range(val_dataset, "Validation set")
    test_in_range = check_range(test_dataset, "Test set")

    if train_in_range and val_in_range and test_in_range:
        print("\n✓ All datasets are properly normalized to [0, 1] range!")
    else:
        print("\n✗ Warning: Some datasets may not be properly normalized!")