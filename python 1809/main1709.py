
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from Dataset1709 import load_split_ds
from model import VGG
import matplotlib.pyplot as plt
from EarlyStopping import EarlyStopping
from err_plot import plot_vector_errors

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path):
    """
    训练模型并验证
    """
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    batch_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        batch_count = 0.0

        for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):  # 改为 batch_images
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

            # 前向传播
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1

            batch_losses.append(loss.item())

            if batch_idx % 10 == 9:
                avg_loss = running_loss / 10
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_loss:.4f}')
                running_loss = 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:  # 改为 val_images
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = model(val_images)
                loss = criterion(outputs, val_labels)
                val_loss += loss.item()
                val_count += 1

        val_loss = val_loss / val_count if val_count > 0 else 0

        # without Dropout train loss
        clean_train_loss = 0.0
        clean_train_count = 0
        with torch.no_grad():
            for clean_images, clean_labels in train_loader:  # 改为 clean_images
                clean_images, clean_labels = clean_images.to(device), clean_labels.to(device)
                outputs = model(clean_images)
                loss = criterion(outputs, clean_labels)
                clean_train_loss += loss.item()
                clean_train_count += 1

        clean_train_loss = clean_train_loss / clean_train_count if clean_train_count > 0 else 0

        train_losses.append(clean_train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss (with dropout): {epoch_loss / batch_count:.4f}, "
              f"Train Loss (no dropout): {clean_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with validation loss: {val_loss:.4f}')

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        del outputs, loss
        torch.cuda.empty_cache()

    return train_losses, val_losses, batch_losses
def test_model(model, test_loader, criterion, device):
    """
    测试模型性能
    """
    model.eval()
    test_loss = 0.0
    test_count = 0

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_count += 1

            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    if len(all_outputs) > 0:
        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
    else:
        all_outputs, all_labels = np.array([]), np.array([])

    test_loss = test_loss / test_count if test_count > 0 else 0
    print(f'Test Loss: {test_loss:.4f}')
    return test_loss, all_outputs, all_labels

def plot_training_curves(train_losses, val_losses, test_loss=None, batch_losses=None):
    """
    绘制训练和验证损失曲线
    """
    plt.figure(figsize=(10, 6))

    # 绘制训练损失

    if batch_losses is not None:
        plt.plot(range(1, len(batch_losses) + 1), batch_losses,label='Batch Loss', color='orange', linestyle='-', alpha=0.6)

    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue', linestyle='-')

    # 绘制验证损失
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red', linestyle='-')

    # 如果有测试损失，添加一条水平线表示测试损失
    if test_loss is not None:
        plt.axhline(y=test_loss, color='green', linestyle='--', label=f'Test Loss: {test_loss:.4f}')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig('tr_sp1809_2.png')
    plt.close()

    print("Training curves saved as 'tr_sp1809_2.png'")



if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 定义多个形状的数据路径
    base_data_path = "/home/wmzheng/PycharmProjects/pythonProject1/Data/"
    shapes = ["spherical"]  # "spherical", "cube", "pyramid"

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 和模型输入一致
        transforms.ToTensor(),

    ])

    train_dataset, val_dataset, test_dataset, label_min, label_max, label_range = load_split_ds(
        base_data_path=base_data_path,
        shapes=shapes,
        transform=transform,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )

    # Batch
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = VGG(image_channels=3, image_size=224).to(device)

    # 测试一下数据和模型
    print("\nTesting data and model...")
    for images, labels in train_loader:
        print("Images shape:", images.shape)  # [batch, num_images, 3, 224, 224]
        print("Labels shape:", labels.shape)  # [batch, 6]

        # 将数据移动到设备
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        output = model(images)
        print("Output shape:", output.shape)  # [batch, 6]
        break

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练参数
    num_epochs = 80
    model_save_path = "best_model_sp1809_2.pt"
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_save_path)
    torch.cuda.empty_cache()
    # Training
    print("\nStarting training...")

    train_losses, val_losses, batch_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs, device, model_save_path
    )
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(model_save_path))

    # test
    print("\nTesting model...")
    test_loss, pre_outputs, test_labels = test_model(model, test_loader, criterion, device)

    tpre = np.array(pre_outputs)
    tlabel = np.array(test_labels)

    label_min = np.array(label_min)
    label_max = np.array(label_max)
    label_range = label_max - label_min

    predtest = tpre * label_range + label_min
    truetest = tlabel * label_range + label_min

    f_pred = predtest[:, :3]
    f_true = truetest[:, :3]
    T_pred = predtest[:, 3:]
    T_true = truetest[:, 3:]

    plotname = "spherical2"

    plot_vector_errors(f_pred, f_true, vector_name='Force', plotname=plotname)
    plot_vector_errors(f_pred, T_true, vector_name='Torque', plotname=plotname)

    # 更新训练曲线，包含测试损失
    print("\nUpdating training curves with test loss...")
    plot_training_curves(train_losses, val_losses, test_loss)

    print("\nTraining and testing completed!")

    del images, labels
    torch.cuda.empty_cache()



