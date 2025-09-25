import numpy as np
import matplotlib.pyplot as plt
import os


def plot_vector_errors(pre_outputs, test_labels, label_min, label_max, plotname, save_dir=None):

    tpre = np.array(pre_outputs)
    tlabel = np.array(test_labels)

    label_min = np.array(label_min)
    label_max = np.array(label_max)
    label_range = label_max - label_min

    predtest = (tpre+1) * label_range/2 + label_min
    truetest = (tlabel+1) * label_range/2 + label_min
    # 2 * (self.loaded_labels - label_min) / label_range - 1
    f_pred = predtest[:, :3]
    f_true = truetest[:, :3]
    T_pred = predtest[:, 3:]
    T_true = truetest[:, 3:]

    #-----------------------------------------------------------------------------------------

    # force mag err
    mag_force_pred = np.linalg.norm(f_pred, axis=1)
    mag_force_true = np.linalg.norm(f_true, axis=1)
    mag_force_error = np.abs(mag_force_pred - mag_force_true)  # 可以取 np.abs(mag_error)

    # force angle err
    cos_theta_force = np.sum(f_pred * f_true, axis=1) / (mag_force_pred * mag_force_true )
    cos_theta_force = np.clip(cos_theta_force, -1, 1)  # 防止浮点误差
    force_angle_error = np.degrees(np.arccos(cos_theta_force))  # 方向误差 (0~180°)

    # torque mag err
    mag_torque_pred = np.linalg.norm(T_pred, axis=1)
    mag_torque_true = np.linalg.norm(T_true, axis=1)
    mag_torque_error = np.abs(mag_torque_pred - mag_torque_true)  # 可以取 np.abs(mag_error)

    # torque angle err
    cos_theta_torque = np.sum(T_pred * T_true, axis=1) / (mag_torque_pred * mag_torque_true )
    cos_theta_torque = np.clip(cos_theta_torque, -1, 1)  # 防止浮点误差
    torque_angle_error = np.degrees(np.arccos(cos_theta_torque))  # 方向误差 (0~180°)

    #-------------------------------------------------------------------------------------------------
    # subplot force err mag

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.hist(mag_force_error, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(mag_force_error), color='r', linestyle='--', label=f'Mean={np.mean(mag_force_error):.2f}')
    plt.axvline(np.mean(mag_force_error) + np.std(mag_force_error), color='g', linestyle=':', label=f'+1σ')
    plt.axvline(np.mean(mag_force_error) - np.std(mag_force_error), color='g', linestyle=':', label=f'-1σ')
    plt.xlabel('Magnitude Error')
    plt.ylabel('Frequency')
    plt.title(f'{plotname} Force Magnitude Error Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # force_angle_error

    plt.subplot(2, 2, 2)
    plt.hist(force_angle_error, bins=30, alpha=0.7, color='salmon', edgecolor='black')
    plt.axvline(np.mean(force_angle_error), color='r', linestyle='--', label=f'Mean={np.mean(force_angle_error):.2f}°')
    plt.axvline(np.mean(force_angle_error) + np.std(force_angle_error), color='g', linestyle=':', label=f'+1σ')
    plt.axvline(np.mean(force_angle_error) - np.std(force_angle_error), color='g', linestyle=':', label=f'-1σ')
    plt.xlabel('Direction Error (deg)')
    plt.ylabel('Frequency')
    plt.title(f'{plotname} Force Direction Error Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    #subplot torque mag err

    plt.subplot(2, 2, 3)
    plt.hist(mag_torque_pred, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(mag_torque_pred), color='r', linestyle='--', label=f'Mean={np.mean(mag_torque_pred):.2f}')
    plt.axvline(np.mean(mag_torque_pred) + np.std(mag_torque_pred), color='g', linestyle=':', label=f'+1σ')
    plt.axvline(np.mean(mag_torque_pred) - np.std(mag_torque_pred), color='g', linestyle=':', label=f'-1σ')
    plt.xlabel('Magnitude Error')
    plt.ylabel('Frequency')
    plt.title(f'{plotname} Torque Magnitude Error Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # subplot torque angle err

    plt.subplot(2, 2, 4)
    plt.hist(torque_angle_error, bins=30, alpha=0.7, color='salmon', edgecolor='black')
    plt.axvline(np.mean(torque_angle_error), color='r', linestyle='--', label=f'Mean={np.mean(torque_angle_error):.2f}°')
    plt.axvline(np.mean(torque_angle_error) + np.std(torque_angle_error), color='g', linestyle=':', label=f'+1σ')
    plt.axvline(np.mean(torque_angle_error) - np.std(torque_angle_error), color='g', linestyle=':', label=f'-1σ')
    plt.xlabel('Direction Error (deg)')
    plt.ylabel('Frequency')
    plt.title(f'{plotname} Torque Direction Error Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    filename = os.path.join(save_dir, f"{plotname}_outputs Errors.png")
    plt.savefig(filename, dpi=300)
    plt.tight_layout()
    plt.show()



    # --- 大小误差 vs 方向误差散点图 + 回归线 ---
    '''coef = np.polyfit(mag_error, angle_error_deg, 1)  # 一次线性拟合
    reg_line = coef[0] * mag_error + coef[1]

    plt.figure()
    plt.scatter(mag_error, angle_error_deg, alpha=0.5, c=angle_error_deg, cmap='viridis')
    plt.plot(mag_error, reg_line, color='r', linewidth=2)  # 回归线
    plt.xlabel('Magnitude Error')
    plt.ylabel('Direction Error (deg)')
    plt.title(f'{plotname} {vector_name} Magnitude vs Direction Error')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.colorbar(label='Direction Error (deg)')

    filename = os.path.join(save_dir, f"{plotname}_{vector_name}_Magnitude vs Direction Error.png")
    plt.savefig(filename, dpi=300)'''

    #plt.show()

    # --- 输出平均值和标准差 ---
    print(f"Force Magnitude Error: Mean={np.mean(mag_force_error):.3f}, Std={np.std(mag_force_error):.3f}")
    print(f"Force Direction Error: Mean={np.mean(force_angle_error):.3f}°, Std={np.std(force_angle_error):.3f}°")
    print(f"Torque Magnitude Error: Mean={np.mean(mag_torque_error):.3f}, Std={np.std(mag_torque_error):.3f}")
    print(f"Torque Direction Error: Mean={np.mean(torque_angle_error):.3f}°, Std={np.std(torque_angle_error):.3f}°")

def plot_loss(train_losses, val_losses, test_loss, plotname, save_path=None):
    """
    绘制训练和验证损失曲线
    """
    plt.figure(figsize=(10, 6))

    training_curve_path = os.path.join(save_path, f"tr_{plotname}.png")
    #Batch_loss_curves = os.path.join(save_path, f"bt_{plotname}.png")

    #
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
    plt.savefig(training_curve_path)
    plt.close()

    print(f"Training curves saved as 'tr_{plotname}.png'")

    '''if batch_losses is not None:
        plt.figure()
        plt.plot(range(1, len(batch_losses) + 1), batch_losses, label='Batch Loss', color='orange', linestyle='-',
                 alpha=0.6)
        plt.title('Batch Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # 保存图像
        plt.savefig(Batch_loss_curves)
        plt.close()'''


