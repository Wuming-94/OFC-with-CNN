import os
import torch
from torchvision import transforms
from PIL import Image
from model import VGG  # 确保导入你的模型类


def predict_single_sample(model, image_paths, transform, device):
    """

    - image_paths: 三张图片的路径列表 [xy_path, yz_path, xz_path]

    """
    # 确保有三张图片
    assert len(image_paths) == 3, "需要提供三张图片的路径"

    # 读取并预处理图片
    images = []
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"NotFound: {path}")

        image = Image.open(path).convert('RGB')
        image = transform(image)
        images.append(image)

    # 将三张图片堆叠成一个张量 [3, 3, 224, 224]
    combined_images = torch.stack(images, dim=0)

    # 添加批次维度 [1, 3, 3, 224, 224]
    combined_images = combined_images.unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        output = model(combined_images)

    return output.squeeze(0)  # 移除批次维度

def load_labels(label_file):
    """
    读取标签文件 {shape}_op.txt
    返回一个 dict: { "Proj1": [f1, f2, f3, t1, t2, t3], ... }
    """
    labels = {}
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # 至少要有名字+6个数
                continue
            sample_name = parts[0]
            values = list(map(float, parts[1:7]))
            labels[sample_name] = values
    return labels


if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model_path = "sp_unNorm_model.pt"  # 你保存的模型路径
    #model_path = "sp_model.pt"  # 你保存的模型路径
    model = VGG(num_images=3, image_channels=3, image_size=224).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("model loaded!")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 和模型输入一致
        transforms.ToTensor(),

    ])

    # 示例: 替换为你的三张图片路径
    shape = "spherical"
    num = "1486"

    image_paths = [
        f"/home/wmzheng/PycharmProjects/pythonProject1/Data/{shape}/Proj{num}_xy.png",
        f"/home/wmzheng/PycharmProjects/pythonProject1/Data/{shape}/Proj{num}_xz.png",
        f"/home/wmzheng/PycharmProjects/pythonProject1/Data/{shape}/Proj{num}_yz.png"
    ]

    label_file = f"/home/wmzheng/PycharmProjects/pythonProject1/Data/{shape}_op.txt"
    labels = load_labels(label_file)

    #label_min = torch.load("label_min_sp.pt", weights_only=True).to(device)
    #label_max = torch.load("label_max_sp.pt", weights_only=True).to(device)
    label_min = torch.load("label_min_sp.pt", weights_only=True).to(device)
    label_max = torch.load("label_max_sp.pt", weights_only=True).to(device)

    label_range = label_max - label_min

    # 进行预测
    try:
        #output = predict_single_sample(model, image_paths, transform, device)
        #prediction = output * label_range + label_min
        prediction = predict_single_sample(model, image_paths, transform, device)

        force_pre = prediction[:3].cpu().numpy()
        torque_pre = prediction[3:].cpu().numpy()

        target = labels[num]
        force_true = target[:3]
        torque_true = target[3:]

        print(f"force_pre(x, y, z): {force_pre}")
        print(f"torque_pre (x, y, z): {torque_pre}")
        print(f"force_true(x, y, z): {force_true}")
        print(f"torque_true (x, y, z): {torque_true}")

    except Exception as e:
        print(f"Error: {e}")