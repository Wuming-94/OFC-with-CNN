import numpy as np
import matplotlib.pyplot as plt


def plot_vector_errors(f_pred, f_true, vector_name='Force',plotname=None):
    """
    可视化矢量预测误差（大小和方向）
    参数:
        f_pred: np.ndarray, shape (N,3) 预测向量
        f_true: np.ndarray, shape (N,3) 真实向量
        vector_name: str, 矢量名称，用于标题
    输出:
        三张图：大小误差直方图、方向误差直方图、大小-方向误差散点图
    """
    f_pred = np.array(f_pred)
    f_true = np.array(f_true)

    # --- 计算大小误差 ---
    mag_pred = np.linalg.norm(f_pred, axis=1)
    mag_true = np.linalg.norm(f_true, axis=1)
    mag_error = mag_pred - mag_true  # 可以取 np.abs(mag_error)

    # --- 计算方向误差 ---
    cos_theta = np.sum(f_pred * f_true, axis=1) / (mag_pred * mag_true)
    cos_theta = np.clip(cos_theta, -1, 1)  # 防止浮点误差
    angle_error_deg = np.degrees(np.arccos(cos_theta))  # 方向误差 (0~180°)

    # --- 大小误差直方图 ---
    plt.figure()
    plt.hist(mag_error, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(mag_error), color='r', linestyle='--', label=f'Mean={np.mean(mag_error):.2f}')
    plt.axvline(np.mean(mag_error) + np.std(mag_error), color='g', linestyle=':', label=f'+1σ')
    plt.axvline(np.mean(mag_error) - np.std(mag_error), color='g', linestyle=':', label=f'-1σ')
    plt.xlabel('Magnitude Error')
    plt.ylabel('Frequency')
    plt.title(f'{plotname} {vector_name} Magnitude Error Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{plotname} {vector_name} Magnitude Error Distribution.png", dpi=300)
    plt.show()

    # --- 方向误差直方图 ---
    plt.figure()
    plt.hist(angle_error_deg, bins=30, alpha=0.7, color='salmon', edgecolor='black')
    plt.axvline(np.mean(angle_error_deg), color='r', linestyle='--', label=f'Mean={np.mean(angle_error_deg):.2f}°')
    plt.axvline(np.mean(angle_error_deg) + np.std(angle_error_deg), color='g', linestyle=':', label=f'+1σ')
    plt.axvline(np.mean(angle_error_deg) - np.std(angle_error_deg), color='g', linestyle=':', label=f'-1σ')
    plt.xlabel('Direction Error (deg)')
    plt.ylabel('Frequency')
    plt.title(f'{plotname} {vector_name} Direction Error Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{plotname} {vector_name} Direction Error Distribution.png", dpi=300)
    plt.show()

    # --- 大小误差 vs 方向误差散点图 + 回归线 ---
    coef = np.polyfit(mag_error, angle_error_deg, 1)  # 一次线性拟合
    reg_line = coef[0] * mag_error + coef[1]

    plt.figure()
    plt.scatter(mag_error, angle_error_deg, alpha=0.5, c=angle_error_deg, cmap='viridis')
    plt.plot(mag_error, reg_line, color='r', linewidth=2)  # 回归线
    plt.xlabel('Magnitude Error')
    plt.ylabel('Direction Error (deg)')
    plt.title(f'{plotname} {vector_name} Magnitude vs Direction Error')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.colorbar(label='Direction Error (deg)')
    plt.savefig(f"{plotname} {vector_name} Magnitude vs Direction Error", dpi=300)
    plt.show()

    # --- 输出平均值和标准差 ---
    print(f"{vector_name} Magnitude Error: Mean={np.mean(mag_error):.3f}, Std={np.std(mag_error):.3f}")
    print(f"{vector_name} Direction Error: Mean={np.mean(angle_error_deg):.3f}°, Std={np.std(angle_error_deg):.3f}°")
