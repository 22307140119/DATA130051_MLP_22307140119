import matplotlib.pyplot as plt
import numpy as np
import layers as ly
import random

random.seed(2)

model_dir = "saved_model_full_5680"

def load(base_path):   # 从指定路径加载模型

    # 加载元数据
    meta = np.load(f"{base_path}/meta.npz", allow_pickle=True)
    weights = []
    
    fcno = 1    # 全连接层编号
    for layer_type in meta["layer_types"]:

        # 全连接层
        if layer_type == "FCLayer":
            # 获取 fcx.npz 中的全连接层参数，并且设置到前面创建的层中
            fc_data = np.load(f"{base_path}/fc{fcno}.npz")
            weights.append(fc_data["W"])
            fcno += 1
    
    return weights


if __name__ == '__main__':
    weights = load(model_dir)
    w1 = weights[0]
    w2 = weights[1]
    w3 = weights[2]
    # print(np.shape(w1))


    # 直接可视化第一层全连接层 R/G/B
    color = 2   # 指定 RGB 通道
    plt.figure(figsize=(15, 6))
    plt.imshow(
        w1[color*1024:color*1024+1024].T,
        cmap='bwr',      # 颜色映射
        interpolation='nearest',  # 无插值
        vmin=-0.125,
        vmax=0.125
    )
    plt.colorbar(label='Value')
    plt.show()

    # 可视化 R/G/B 的 32*32 小图
    idxlist = [random.randint(0, 512) for _ in range(10)]
    print(idxlist)
    fig, axes = plt.subplots(1, len(idxlist), figsize=(15, 2))
    for i in range(len(idxlist)):
        f = w1[color*1024:color*1024+1024, idxlist[i]].reshape(32, 32)
        axes[i].imshow(
            f,
            cmap='bwr',      # 颜色映射
            interpolation='nearest',  # 无插值
            vmin=-0.125,
            vmax=0.125
        )
        axes[i].axis('off')
    plt.show()

    # 三张 32*32 的小图变成彩色图像
    fig, axes = plt.subplots(1, len(idxlist), figsize=(15, 2))
    for i in range(len(idxlist)):
        f = w1[:, idxlist[i]].reshape(3, 32, 32)
        f = np.transpose(f, (1, 2, 0))
        f = ((f+0.125)*256/0.25).astype(np.uint8)
        axes[i].imshow(f)
        axes[i].axis('off')  # 关闭坐标轴
    plt.show()


    # 可视化第二个全连接层    
    plt.figure(figsize=(15, 6))
    plt.imshow(
        w2.T,
        cmap='bwr',      # 颜色映射
        interpolation='nearest',  # 无插值
    )
    plt.colorbar(label='Value')
    plt.show()
    
    # 可视化第三个全连接层
    plt.figure(figsize=(15, 3))
    plt.imshow(
        w3.T,
        cmap='bwr',      # 颜色映射
        interpolation='nearest',  # 无插值
    )
    plt.colorbar(label='Value')
    plt.show()


