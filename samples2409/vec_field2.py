import numpy as np
import matplotlib.pyplot as plt
import os


def load_vectors(filename, size=None):
    data = np.loadtxt(filename)
    vec_f = data[:, 1:4]
    vec_T = data[:, 4:7]

    if size is not None:
        idx = np.random.choice(data.shape[0], size=size, replace=False)
        vec_f = vec_f[idx]
        vec_T = vec_T[idx]

    return vec_f, vec_T


def load_coords(filename, size=None):
    coords = np.loadtxt(filename)
    coords = coords[:, 1:4]

    if size is not None:
        idx = np.random.choice(coords.shape[0], size=size, replace=False)
        coords = coords[idx]

    return coords


def plot_vector_field(coords, vec1, vec2, arrow_length=0.1):
    """
    绘制三维向量场，所有箭头等长，只显示方向

    Parameters:
    coords: 坐标点 (N, 3) - 箭头的起始位置
    vec1, vec2: 向量 (N, 3) - 箭头的方向
    arrow_length: 箭头长度（固定值）
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 方法：获取方向向量（固定长度）
    def get_direction_vectors(vectors, length):
        # 计算每个向量的模长
        norms = np.linalg.norm(vectors, axis=1)
        # 避免除以零
        norms[norms == 0] = 1
        # 归一化并乘以固定长度
        return (vectors / norms[:, np.newaxis]) * length

    vec1_dir = get_direction_vectors(vec1, arrow_length)
    vec2_dir = get_direction_vectors(vec2, arrow_length)

    print(f"坐标范围: X[{np.min(coords[:, 0]):.3f}, {np.max(coords[:, 0]):.3f}], "
          f"Y[{np.min(coords[:, 1]):.3f}, {np.max(coords[:, 1]):.3f}], "
          f"Z[{np.min(coords[:, 2]):.3f}, {np.max(coords[:, 2]):.3f}]")
    print(f"箭头长度: {arrow_length}")

    # 绘制第一个向量场（红色箭头）
    # 箭头的起始位置是 coords，方向是 vec1_dir
    ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
              vec1_dir[:, 0], vec1_dir[:, 1], vec1_dir[:, 2],
              color='r', length=1.0, normalize=False,
              arrow_length_ratio=0.3, linewidth=1.5, label="Force", alpha=0.7)

    # 绘制第二个向量场（蓝色箭头）
    '''ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
              vec2_dir[:, 0], vec2_dir[:, 1], vec2_dir[:, 2],
              color='b', length=1.0, normalize=False,
              arrow_length_ratio=0.3, linewidth=1.5, label="Torque", alpha=0.7)'''

    # 也绘制坐标点，便于观察箭头位置
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c='g', s=10, alpha=0.5, label="Points")

    # 设置坐标轴标签和图例
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # 设置坐标轴范围，确保箭头完全可见
    padding = arrow_length * 3
    ax.set_xlim([np.min(coords[:, 0]) - padding, np.max(coords[:, 0]) + padding])
    ax.set_ylim([np.min(coords[:, 1]) - padding, np.max(coords[:, 1]) + padding])
    ax.set_zlim([np.min(coords[:, 2]) - padding, np.max(coords[:, 2]) + padding])

    ax.set_title(f"Vector Field Direction (N={len(coords)} points)")
    plt.tight_layout()
    filename = os.path.join("save", f"{shape}_Force_100sample_vectorfield.png")
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    shape = "spherical"
    vector_file = f"Data/{shape}_op.txt"
    coord_file = f"Data/{shape}_para.txt"

    vecf, vecT = load_vectors(vector_file, size=100)  # 先试100个点
    coords = load_coords(coord_file, size=100)

    # 根据坐标范围调整箭头长度
    coord_range = np.ptp(coords, axis=0)  # 坐标范围
    arrow_length = np.mean(coord_range) * 0.1  # 箭头长度为平均范围的10%

    print(f"坐标范围: {coord_range}")
    print(f"自动计算箭头长度: {arrow_length}")

    plot_vector_field(coords, vecf, vecT, arrow_length=arrow_length)