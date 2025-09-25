import numpy as np
import matplotlib.pyplot as plt

# 1. 读取数据
# 假设你的文件名叫 "data.txt"
name = "spherical" #pyramid,spherical,cube
data = np.loadtxt(f"/home/wmzheng/PycharmProjects/pythonProject1/Data/{name}_para.txt")

# 2. 提取 v.X, v.Y, v.Z (注意索引：第一列是 i，所以 X=第2列)
X = data[:, 1]  # v.X
Y = data[:, 2]  # v.Y
Z = data[:, 3]  # v.Z

# 3. 绘制3D散点图
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X, Y, Z, c='b', marker='o', s=10)

# 设置坐标轴标签
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Scattering Plot of v.X, v.Y, v.Z")

plt.savefig(f"sca3d_{name}.png", dpi=300)   # 保存为 PNG，分辨率 300 dpi

#plt.savefig("sca3d.pdf")           # 保存为 PDF 矢量图


plt.show()


