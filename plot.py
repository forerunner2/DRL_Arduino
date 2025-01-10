import matplotlib.pyplot as plt
import os
import numpy as np


def draw(path, label, color):
    x_data = []
    y_data = []

    # 获取文件夹中所有的 txt 文件
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    for file in files:
        file_path = os.path.join(path, file)
        data = np.loadtxt(file_path)

        # 假设第一列是横轴，第二列是纵轴
        x_values = data[:, 0]
        y_values = data[:, 1]

        x_data = x_values
        y_data.append(y_values)

    y_data = np.array(y_data)
    mean_y = np.mean(y_data, axis=0)
    std_y = np.std(y_data, axis=0)

    # 绘制
    plt.plot(x_data, mean_y, label='reward', color='b')
    plt.fill_between(x_data, mean_y - std_y, mean_y + std_y, color='b', alpha=0.2)
    plt.xlabel('step')
    plt.ylabel('reward')
    plt.title('Reward Curve')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

