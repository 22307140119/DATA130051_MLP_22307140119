import os
import numpy as np
import layers as ly

class SimpleModel():
    def __init__(self, layers, regularization_method=None, lambdal=0):
        self.layers = []
        for item in layers:
            self.layers.append(item)
        self.method = regularization_method # 正则化手段
        self.lambdal = lambdal              # 正则化强度


    # 最后一层必须是损失层
    # 损失层的 forward() 必须只接受两个参数：输入X 和 标答Y，且只有一个返回值
    # 其它层的 forward() 必须只接受一个参数，且只有一个返回值
    def forward(self, X, Y):
        # 前向传播
        for layer in self.layers[:-1]:
            X = layer.forward(X)
        layer = self.layers[-1]
        dict = layer.forward(X, Y) # 损失层，接收两个参数

        # 调用 FCLayer 中的函数求正则化损失
        # L1 和 L2 损失都是“可加”的
        for layer in self.layers[:-1]:
            if hasattr(layer, "regularization"):    # 具有 regularization 方法的类（FCLayer）
                penalty = layer.regularization(self.method, self.lambdal)
                dict['loss'] += penalty

        return dict
    

    # 最后一层必须是损失层
    # 损失层的 backward() 不接收任何参数，且只有一个返回值
    # 其它层的 backward() 必须只接受一个参数，且只有一个返回值
    def backward(self):
        layer = self.layers[-1]
        grad = layer.backward()
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad)
        return grad
    
    
    # 更新参数
    def GDupdate(self, lr):
        for layer in self.layers:
            if hasattr(layer, "GDupdate"):    # 具有 GDupdate 方法的类（FCLayer）
                layer.GDupdate(lr)

    
    # 保存模型，全连接层参数保存到 fc.npz，元数据保存到 
    def save(self, base_path):
        # 自动创建路径
        os.makedirs(base_path, exist_ok=True)

        # 保存模型元数据（各层类型，全连接层的输入输出维数，正则化方法，正则化强度）
        metadata = {
            "layer_types": [type(layer).__name__ for layer in self.layers],
            "fc_shape": [],
            "regularization": self.method,
            "lambdal": self.lambdal
        }
        
        # 保存全连接层参数
        fcno = 1    # 全连接层编号
        for layer in self.layers:
            if isinstance(layer, ly.FCLayer):

                # 用单独的文件 fcx.npz 保存全连接层x的参数
                np.savez(f"{base_path}\\fc{fcno}.npz", W=layer.W, b=layer.b)
                fcno += 1

                # 全连接层的输入输出维数存入 metadata 中
                metadata["fc_shape"].append({
                    "din": layer.W.shape[0],
                    "dout": layer.W.shape[1]
                })
        
        # 元数据保存到 meta.npz 中
        np.savez(f"{base_path}/meta.npz", **metadata)

    
    
    @classmethod
    def load(cls, base_path):   # 从指定路径加载模型

        # 加载元数据
        meta = np.load(f"{base_path}/meta.npz", allow_pickle=True)
        layers = []
        
        # 重建各层
        fcno = 1    # 全连接层编号
        for layer_type in meta["layer_types"]:

            # 全连接层
            if layer_type == "FCLayer":
                # 获取 meta 中存储的全连接层输入输出维数，创建全连接层
                shape = meta["fc_shape"][fcno-1]
                layer = ly.FCLayer(shape["din"], shape["dout"])
                # 获取 fcx.npz 中的全连接层参数，并且设置到前面创建的层中
                fc_data = np.load(f"{base_path}/fc{fcno}.npz")
                layer.W, layer.b = fc_data["W"], fc_data["b"]
                fcno += 1

            # 激活函数层
            elif layer_type == "ReLU":
                layer = ly.ReLU()
            elif layer_type == "Sigmoid":
                layer = ly.Sigmoid()
            elif layer_type == "Tanh":
                layer = ly.Tanh()
            elif layer_type == "LeakyReLU":
                layer = ly.LeakyReLU()
            
            # 交叉熵损失层
            elif layer_type == "Softmax_CrossEntropy":
                layer = ly.Softmax_CrossEntropy()
        
            layers.append(layer)
            
        return cls(layers=layers,
                   regularization_method=meta["regularization"],
                   lambdal=meta["lambdal"])



    # 获取模型超参数
    def get_params(self):
        # 隐藏层的维数
        hidden = []
        for layer in self.layers:
            if isinstance(layer, ly.FCLayer):
                hidden.append(layer.W.shape[1])
        hidden = hidden[:-1]    # 去掉输出层维数

        return {"hidden_dim": hidden,
                "regularization": self.method,
                "lambdal": self.lambdal
                }

