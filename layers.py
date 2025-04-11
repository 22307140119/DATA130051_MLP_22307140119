import numpy as np


# 全连接层
# W, b, X(input), output, dW, db, dX
class FCLayer():
    def __init__(self, din, dout):
        self.W = 0.01* np.random.randn(din, dout)   # 权重初始化为随机数
        self.b = np.zeros((1, dout))                # 偏置初始化为0


    def forward(self, X):
        self.X = X              # 存一下输入，反向传播里面要用
        self.output = np.dot(X, self.W) + self.b
        return self.output

    def backward(self, upgd):
        self.dW = np.dot(self.X.T, upgd)
        self.db = np.mean(upgd, axis=0, keepdims=True)  # upgd 维数是 batchsize*dout，取平均（损失函数中取平均）为 1*dout
        self.dX = np.dot(upgd, self.W.T)
        return self.dX


    # 计算正则化损失和正则化梯度，method为空则把这两项均置零
    def regularization(self, method, lambdal):
        if method == 'l2' or method == 'L2':
            self.penalty = np.sum(self.W ** 2) * lambdal / 2
            self.drdW = self.W * lambdal 
        elif method == 'l1' or method == 'L1':
            self.penalty = np.sum(np.abs(self.W)) * lambdal
            self.drdW = np.sign(self.W) * lambdal
        elif method == None:
            self.penalty = 0    # 正则化项
            self.drdW = 0       # 梯度
        else:
            raise ValueError(f"undefined regularization method.")
        
        return self.penalty
    

    def GDupdate(self, lr):
        self.W -= lr * (self.dW + self.drdW)
        self.b -= lr * self.db
    


# 激活函数父类
# Z(input), output
class ActivationFunc():
    def __init__(self):
        pass

    def forward(self, Z):
        raise NotImplementedError

    def backward(self, upgd):
        raise NotImplementedError
    


# ReLU 激活函数
class ReLU(ActivationFunc):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.Z = Z
        self.output = np.maximum(0, Z)
        return self.output
    
    def backward(self, upgd):
        dZ = upgd * (self.Z > 0)    # 逐元素乘以 ReLU 的导数
        return dZ       


# Sigmoid / logistic 激活函数
class Sigmoid(ActivationFunc):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.Z = Z
        self.output = 1 / (1 + np.exp(-Z))
        return self.output
    
    def backward(self, upgd):
        sig = self.output
        dZ = upgd * sig * (1 - sig)  # 逐元素乘以 Sigmoid 的导数
        return dZ


# Tanh 激活函数
class Tanh(ActivationFunc):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.Z = Z
        self.output = np.tanh(Z)
        return self.output
    
    def backward(self, upgd):
        dZ = upgd * (1 - self.output**2)  # 逐元素乘以 Tanh 的导数
        return dZ


# Leaky ReLU
class LeakyReLU(ActivationFunc):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha  # 负区间的斜率参数（默认 0.01）

    def forward(self, Z):
        self.Z = Z
        self.output = np.where(Z > 0, Z, self.alpha * Z)    # > 0 的元素取 Z，否则取 alpha*Z
        return self.output
    
    def backward(self, upgd):
        dZ = upgd * np.where(self.Z > 0, 1, self.alpha)
        return dZ


# 其它激活函数可以添加在这里



# Softmax 函数，为了防止指数爆炸，每个数据均减去当前 batch 中的最大值
def Softmax(Z):
    Z_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))    # 计算一堆 exp(zi)
    p = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
    return p


# CrossEntropy 损失函数，返回平均损失，为了防止 log(0) 加上小量 1e-15
def CrossEntropyLoss(p, Y):                 # p, Y 尺寸 batchsize*c
    loss = -np.sum(Y * np.log(p + 1e-15), axis=1)   # 此处 loss 尺寸 batchsize*1
    loss = np.mean(loss)                    # 此处 loss 为标量
    return loss



# softmax 转换成概率后，计算交叉熵
# 相比分开来可以简化反向传播的过程，在最后一个线性层后使用
# p(softmax概率值), loss, Y(gold)
class Softmax_CrossEntropy():
    def __init__(self):
        pass

    def forward(self, Z, Y):    # 输入 Z，计算平均损失和，Y 必须是 one-hot 编码的形式
        self.Y = Y              # 反向传播用
        self.p = Softmax(Z)     # 用 self.p 存 softmax 算出来的概率，后面反向传播用
        self.loss = CrossEntropyLoss(self.p, Y)

        pred = np.argmax(self.p, axis=1)
        gold = np.argmax(Y, axis=1)
        self.accuracy = np.mean(pred == gold)

        return {"loss": self.loss, 
                "accuracy": self.accuracy}

    def backward(self):
        batch_size = self.Y.shape[0]
        return (self.p - self.Y) / batch_size
    








# 调试用


if __name__ == "__main__":

    # 将 标签Y 转换为 one-hot 编码
    def to_one_hot(labels, num_classes=10):
        num_labels = len(labels)
        one_hot = np.zeros((num_labels, num_classes))   # 初始化空数组
        one_hot[np.arange(num_labels), labels] = 1      # 直接用索引设置
        return one_hot

    np.random.seed(1)
    # 输入和标签（假设为 one-hot 编码）
    X = np.array([[0.1, 0.2, 0.3],  # 输入特征
                 [0.4, 0.5, 0.6]])
    labels = np.array([0, 1])      # 标签（整数）
    Y = to_one_hot(labels, num_classes=3)  # 转换为 one-hot

    # 构建模型
    # 可以改成模型类，用列表实现各个层
    fc1 = FCLayer(din=3, dout=3)
    ac = ReLU()
    fc2 = FCLayer(din=3, dout=3)
    loss_layer = Softmax_CrossEntropy()

    for i in range(1):
        # 前向传播
        Z = fc1.forward(X)
        X = ac.forward(Z)
        Z = fc2.forward(X)
        loss = loss_layer.forward(Z, Y)
        print("Loss:", loss)

        # 反向传播
        grad = loss_layer.backward()
        grad_fc2 = fc2.backward(grad)
        grad_ac = ac.backward(grad_fc2)
        grad_fc = fc1.backward(grad_ac)
        # print("Gradient from FC layer:\n", grad_fc)

        # 参数更新
        lr = 5
        fc1.W -= lr*fc1.dW
        fc1.b -= lr*fc1.db
        fc2.W -= lr*fc2.dW
        fc2.b -= lr*fc2.db