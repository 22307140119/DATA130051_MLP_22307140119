import pickle
import numpy as np
import os

DIR = r"E:\cifar-10-batches-py"    # 数据集路径
# FILE = r"E:\cifar-10-batches-py\data_batch_1"   # 待打开的文件路径


# 打开单个文件并返回 数据X 和 标签Y
def unpickle(file):
    with open(file, 'rb') as fo:    # 网站上的
        dict = pickle.load(fo, encoding='bytes')

    X = dict[b"data"]               # 图像数据，尺寸 (10000, 3072)
    Y = np.array(dict[b"labels"])   # 标签，尺寸 (10000,)

    return X, Y


# 将 标签Y 转换为 one-hot 编码
def to_one_hot(labels, num_classes=10):
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_classes))   # 初始化全零的空数组
    one_hot[np.arange(num_labels), labels] = 1      # 直接用索引设置 one_hot
    return one_hot


# 数据预处理：根据训练集的均值和标准差标准化
def preprocess_data(train_X, test_X):
    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)

    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std
    return train_X, test_X


# 调用前面的三个函数，从大文件夹中加载数据
def load_cifar10(dir):

    # 训练集
    train_X, train_Y = [], []
    for i in range(1, 6):  # CIFAR-10 训练集分布在 5 个文件中
        file = os.path.join(dir, f"data_batch_{i}")
        X, Y = unpickle(file)
        train_X.append(X)
        train_Y.append(Y)
    train_X = np.vstack(train_X)  # (50000, 3072)
    train_Y = np.hstack(train_Y)  # (50000,)

    # 测试集
    test_file_path = os.path.join(dir, "test_batch")
    test_X, test_Y = unpickle(test_file_path)

    # 数据X 标准化
    train_X, test_X = preprocess_data(train_X, test_X)

    # 标签Y 转化为 one_hot 编码
    train_Y = to_one_hot(train_Y, 10)
    test_Y = to_one_hot(test_Y, 10)

    return train_X, train_Y, test_X, test_Y


# 调试用
if __name__ == "__main__":
    train_X, train_Y, test_X, test_Y = load_cifar10(DIR)
    print(f"size of train_X: {np.shape(train_X)}")
    print(f"size of train_Y: {np.shape(train_Y)}")
    print(f"size of test_X: {np.shape(test_X)}")
    print(f"size of test_Y: {np.shape(test_Y)}")

    X = train_X[0:1]    # 这样取出来的大小才是 (1, 3072)，否则大小会变成 (3072,)
    print(f"size of X: {np.shape(X)}")

