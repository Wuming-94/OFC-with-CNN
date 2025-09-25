# moniter.py
import matplotlib.pyplot as plt
from collections import defaultdict
import os


class TrainingMonitor:
    def __init__(self, save_dir="save/monitor_plots"):
        self.history = defaultdict(list)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def update(self, metrics, epoch):
        for key, value in metrics.items():
            self.history[key].append(value)

        # 每5个epoch保存一次监控图
        if epoch % 5 == 0:
            self.plot_progress(epoch)

    def plot_progress(self, epoch):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 总损失
        if self.history['total_loss']:
            axes[0, 0].plot(self.history['total_loss'])
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True)

        # 力方向误差
        if self.history['force_angle_error']:
            axes[0, 1].plot(self.history['force_angle_error'])
            axes[0, 1].set_title('Force Direction Error (°)')
            axes[0, 1].set_ylabel('Angle Error (°)')
            axes[0, 1].grid(True)

        # 扭矩方向误差
        if self.history['torque_angle_error']:
            axes[0, 2].plot(self.history['torque_angle_error'])
            axes[0, 2].set_title('Torque Direction Error (°)')
            axes[0, 2].set_ylabel('Angle Error (°)')
            axes[0, 2].grid(True)

        # 力损失
        if self.history['force_loss']:
            axes[1, 0].plot(self.history['force_loss'])
            axes[1, 0].set_title('Force Loss')
            axes[1, 0].grid(True)

        # 扭矩损失
        if self.history['torque_loss']:
            axes[1, 1].plot(self.history['torque_loss'])
            axes[1, 1].set_title('Torque Loss')
            axes[1, 1].grid(True)

        # 学习率
        if self.history.get('learning_rate'):
            axes[1, 2].plot(self.history['learning_rate'])
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'training_monitor_epoch_{epoch:03d}.png'))
        plt.close()

        print(f"Monitor plots saved for epoch {epoch}")

    def print_epoch_summary(self, epoch):
        if self.history['total_loss']:
            print(f"Epoch {epoch} Summary:")
            if self.history['force_angle_error']:
                print(f"  Force Error: {self.history['force_angle_error'][-1]:.2f}°")
            if self.history['torque_angle_error']:
                print(f"  Torque Error: {self.history['torque_angle_error'][-1]:.2f}°")
            print(f"  Total Loss: {self.history['total_loss'][-1]:.4f}")