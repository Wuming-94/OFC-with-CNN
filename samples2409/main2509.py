
import torch
import torch.nn as nn
import os
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from Dataset2009 import load_split_ds
from model_VGG19 import VGG
from EarlyStopping import EarlyStopping
from plot_err_2509 import plot_vector_errors, plot_loss
from loss_func2209 import VectorLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

save_dir = "save/samples2409"
os.makedirs(save_dir, exist_ok=True)

def seperate_loss(images,labels):
    # seperate_loss
    '''outputs = model(images) #[batchsize, 6]

    force_pred, torque_pred = outputs[:, :3], outputs[:, 3:] #torch.Size([64, 3])
    force_true, torque_true = labels[:, :3], labels[:, 3:]  #tensor([[-0.0727,  0.6147, -0.3523]...
    #print(force_pred.shape)   #torch.Size([64, 3])
    #print(torque_pred.shape)
    loss_force = criterion1(force_pred, force_true)  # loss = criterion(outputs, batch_labels) shape:torch.Size([]) 0dim tensor(2.3372,
    loss_torque = criterion2(torque_pred, torque_true)
    loss = loss_force + loss_torque  #'''

    # 6 dim loss
    outputs = model(images)
    loss = criterion(outputs, labels)

    return loss, outputs.detach()
def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, model_save_path):
    """
    训练模型并验证
    """
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    #batch_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        batch_count = 0.0

        for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):  # 8000 samples 64 Batchsize =125 batch
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device) # batch_labels import from Dataset train_loader
            #forword
            loss, tr_ops = seperate_loss(batch_images, batch_labels)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  #avg_loss of batch tensor-->float
            epoch_loss += loss.item()
            batch_count += 1

            #batch_losses.append(loss.item())

            del loss, tr_ops, batch_images, batch_labels

            if batch_idx % 10 == 9:
                avg_loss = running_loss / 10
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {avg_loss:.4f}')
                running_loss = 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:  # load val_images
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                loss, val_ops = seperate_loss(val_images, val_labels)

                del val_images, val_labels, val_ops

                val_loss += loss.item()
                val_count += 1

        val_loss = val_loss / val_count if val_count > 0 else 0

        # Update the learning rate scheduler
        #scheduler.step(val_loss)
        #current_lr = optimizer.param_groups[0]['lr']
        #print(f"Current learning rate: {current_lr:.6f}")

        # without Dropout train loss
        clean_train_loss = 0.0
        clean_train_count = 0

        with torch.no_grad():
            for clean_images, clean_labels in train_loader:  # 改为 clean_images

                clean_images, clean_labels = clean_images.to(device), clean_labels.to(device)

                loss, dry_tr_ops = seperate_loss(clean_images, clean_labels)

                clean_train_loss += loss.item()
                clean_train_count += 1

                del loss, dry_tr_ops, clean_images, clean_labels

        clean_train_loss = clean_train_loss / clean_train_count if clean_train_count > 0 else 0
        #if clean_train_count < 0,equals 0
        # result = value_if_true if condition else value_if_false when condition true,back result,else,back value_if_false

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

        '''early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break'''

        #del outputs, loss
        torch.cuda.empty_cache()

    return train_losses, val_losses

def test_model(model, test_loader, device):

    model.eval()
    test_loss = 0.0
    test_count = 0

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images, test_labels = test_images.to(device), test_labels.to(device)

            loss, tst_ops = seperate_loss(test_images, test_labels) #calc loss

            test_loss += loss.item()
            test_count += 1

            all_outputs.append(tst_ops.cpu())
            all_labels.append(test_labels.cpu())

            del test_images, test_labels, tst_ops, loss

    if len(all_outputs) > 0:
        all_outputs = torch.cat(all_outputs, dim=0).numpy() #[]
        all_labels = torch.cat(all_labels, dim=0).numpy()
    else:
        all_outputs, all_labels = np.array([]), np.array([])

    test_loss = test_loss / test_count if test_count > 0 else 0
    print(f'Test Loss: {test_loss:.4f}')
    return test_loss, all_outputs, all_labels




if __name__ == "__main__":

    #seed = 42
    #torch.manual_seed(seed)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # define
    base_data_path = "/home/wmzheng/PycharmProjects/pythonProject1/Data/"
    shapes = ["spherical"]  # "spherical", "cube", "pyramid"
    num = 2
    date = 2509


    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  #
        transforms.ToTensor() #[0,1]
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = VGG(image_channels=3, image_size=224).to(device)

    # 测试一下数据和模型
    print("\nTesting data and model...")
    for images, labels in train_loader:
        print("Images shape:", images.shape)  # [batch, 3, 224, 224]
        print("Labels shape:", labels.shape)  # [batch, 6]

        # 将数据移动到设备
        images, labels = images.to(device), labels.to(device)

        # forward prop
        output = model(images)
        print("Output shape:", output.shape)  # [batch, 6]
        break

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    #criterion1 = VectorLoss(alpha=1.0, beta=1.5)
    #criterion2 = VectorLoss(alpha=1.0, beta=3.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    '''optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),  # 默认值
        eps=1e-08,  # 默认值
        weight_decay=0.01,  # 添加权重衰减
        amsgrad=False  # 是否使用AMSGrad变体
    )'''

    # 学习率调度器：当验证集 loss 10 个 epoch 内没有下降，就乘以 0.5
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 训练参数
    num_epochs = 80
    model_save_path = f"save/bm_{shapes[0]}{date}_{num}.pt"
    #early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.0001, path=model_save_path)
    torch.cuda.empty_cache()
    # Training
    print("\nStarting training...")

    train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer,
        num_epochs, device, model_save_path
    )
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(model_save_path))

    # test
    print("\nTesting model...")
    test_loss, pre_outputs, test_labels = test_model(model, test_loader, device)

    plotname = f"{shapes[0]}{num}_{date}"

    plot_vector_errors(pre_outputs, test_labels, label_min, label_max, plotname, save_dir=save_dir)
    #plot_vector_errors(T_pred, T_true, vector_name='Torque', plotname=plotname, save_dir=save_dir)


    # 更新训练曲线，包含测试损失
    print("\nUpdating training curves with test loss...")


    plot_loss(train_losses, val_losses, test_loss, plotname, save_path=save_dir)

    print("\nTraining and testing completed!")

    del images, labels
    torch.cuda.empty_cache()



