import numpy as np
import matplotlib.pyplot as plt
import os

# 加载数据
op_data = np.loadtxt('Data/spherical/spherical_op.txt')  #
#
values = op_data[:, 1:]  # 2 column til the end

# 绘制每一列的直方图
column_names = ['Fx','Fy','Fz','Tx','Ty','Tz']

plt.figure(figsize=(12, 8))
for i in range(values.shape[1]): #0-5
    plt.subplot(2, 3, i+1)
    plt.hist(values[:, i], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column_names[i]}')
    plt.xlabel(column_names[i])
    plt.ylabel('Frequency')

plt.tight_layout()
path = os.path.join("save/samples2309", f"label_histogramm.png")
plt.savefig(path)
plt.show()
