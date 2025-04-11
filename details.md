## 详细实现思路

### 数据集
- CIFAR10 数据集
  - CIFAR10 数据集包含 60000 张 32×32 像素的三通道彩色图像，分为 10 个类别。
  - 数据集分为 50000 张训练图像和 10000 张测试图像。
  - 训练图像存储在 data_batch_1 - data_batch_5 文件中，测试数据存储在 test_batch 文件中。
  - 图像数据以展平后 3072 维向量的形式给出，维数顺序 3*32*32，标签为 0-9 的整数。
- 读取数据和数据预处理
  - 通过 pickle.load() 加载训练集和测试集数据。
  - 将训练和测试图像标准化。
  - 把训练和测试标签转换为 one-hot 编码。
- 划分验证集和打乱数据
  - 在开始训练后 (train.py中) 实现。



#### load_CIFAR10.py：读取数据集，独立于其它文件
通过调用 load_cifar10(dir)，可以从 dir 目录中读取整个 CIFAR10 数据集。返回标准化后的训练集和数据集，Y 以 one-hot 编码形式存储。

- 加载整个 CIFAR10 数据集 load_cifar10(dir)：
  - 输入参数：CIFAR-10数据集根目录路径（包含 data_batch_1-5 和 test_batch）。
  - 输出：
    - 训练集：
      - train_X：维数 (50000, 3072)，标准化后的像素值。
      - train_Y：维数 (50000, 10)，one-hot 编码标签。
    - 测试集：
      - test_X：维数 (10000, 3072)，标准化后的像素值。
      - test_Y：维数 (10000, 10)，one-hot 编码标签。
  - 调用辅助函数 unpickle() 依次读取五个训练数据文件和一个测试数据文件。
  - 调用辅助函数 preprocess_data() 进行数据预处理（标准化）。
  - 调用辅助函数 to_one_hot() 将标签转换为 one-hot 编码。

- 读取单个文件 unpickle(file)：<br>
  打开由 file 指定的单个数据文件，返回 数据 X 和 标签 Y。
  - 使用 pickle.load() 加载文件，存储到字典 dict 中。
  - 获取字典中，键 b"data" 和 b"labels" 对应的数据，即为 X 和 Y。
  - X 维数 (10000, 3072)，每行对应一张图像的展平像素值（3*32*32）。
  - Y 维数 (10000,)，对应图像的类别标签（0-9）。

- 将标签转换为 one-hot 编码：to_one_hot(labels, num_classes=10)：<br>
  提供 原始标签 labels 和 类别总数 num_classes，将 labels 转换为 one-hot 编码。
  - 直接创建新的全零数组，通过索引直接把对应元素置 1。
  - 后续调用时，原始标签 Y 维数 (10000,)，转换为 one-hot 编码后维数 (10000, 10)。
  - 虽然转换为 one-hot 编码后尺寸增加，但相比 数据 X 的 (10000, 3072) 维数仍然是较小的，不会构成主要复杂度。

- 数据预处理 preprocess_data(train_X, test_X)：<br>
  输入 训练集数据 train_X 和 测试集数据 test_X，进行数据标准化，返回标准化后的数据。
  - 计算训练集 train_X 的均值 (np.mean()) 和标准差 (np.std())。
  - 根据训练集的数据，标准化训练集和测试集。测试集同样使用训练集的统计量，避免数据泄露。



### 模型
#### layers.py：最底层的结构实现
定义了全连接层、激活函数层、softmax + 交叉熵损失层。<br>
每层中含有：初始化方法 \_\_init__()，前向传播 forward()，反向转播 backward()。<br>
线性全连接层 FCLayer 中额外含有 regularization()（实现 L1 和 L2 正则化）和 GDupdate()（实现梯度下降更新参数）。

##### 全连接层 FCLayer类
- 初始化 \_\_init(din, dout)__：<br>
  提供 输入维数 din 和 输出维数 dout 作为参数。
  - 初始化权重 W，维数为 (din, dout)，采用 0.01*N(0,1) 初始化。
  - 初始化偏置 b，维数为 (1, dout)。这里初始化为全零。

- 前向传播 forward(X) ：<br>
  提供 输入 X （维数 (batchsize, din)），返回前向传播的结果。
  - 根据权重 W 和偏置 b，计算 $ XW + b $。
  - 需要存储输入的 X，用于反向传播。

- 反向传播 backward(upgd) ：<br>
  提供上游梯度 $ upgd $（维数 (batchsize, dout)），返回向下层传递的下游梯度。
  - 计算损失关于输入 X 的梯度：$ dL/dX = upgd · W^T $
  - 线性层需要更新参数，额外计算损失关于参数 W,b 的梯度：$ dL/dW = X^T · upgd $, $ dL/db = mean_i(upgd_i) $

- 正则化 regularization() ：<br>
  提供 正则化方法 method（字符串 "L1" 或 "L2"）和 正则化强度系数 lambdal，返回正则化项的值。
  - 正则化方法 method 支持 L1、L2 或无正则化。
  - 根据正则化方法，计算罚项 penalty 和 参数梯度 drdW （用于参数更新），分别存入对应变量。
  - 返回罚项值，用于和交叉熵损失累加得到最终的损失。
  - 正则化应该只需要考虑权重 W，不需要考虑偏置 b。

- minibatch GD 更新参数 GDupdate(lr)：<br>
  提供学习率 lr，根据学习率和梯度更新模型中的参数。
  - 权重 W 的梯度分为两部分：
    - 反向传播得到的梯度（交叉熵损失的梯度）dW。
    - 正则化项的梯度 drdW。
  - 更新参数时将原本的参数 W 或 b 减去 lr*对应梯度即可。


##### 激活函数父类
定义抽象接口，子类需实现 forward() 和 backward() 方法。<br>

- 实现了 ReLU, Sigmoid, tanh, Leaky ReLU 四种激活函数，其中 Leaky ReLU 中参数 α 默认为 0.01。
- 由于激活函数的本质都是把标量映射成标量， forward() 和 backward() 均为逐元素操作，不会改变输入/输出维数 (batchsize, c)。

##### 损失函数类
- 前向传播 forward(Z)：<br>
  提供 输入 Z 和 标签 Y，标签 Y 必须为 ong-hot 编码（在 load_CIFAR10.py 中已经保证了），返回前向传播后的值。
  - 对输入 Z 作用 Softmax()，获取预测的概率 p。
  - 根据预测的 概率 p 和 标签 Y，使用 CrossEntropyLoss() 计算交叉熵损失 loss。
  - 计算（训练集上的）预测准确率，得到 accuracy。
  - 返回字典，含有 "loss" 和 "accuracy" 两个字段，分别存放训练集上的 交叉熵损失 loss 和 准确率 accuracy。

- 反向传播 backward(upgd)：<br>
  提供上游梯度 $ upgd $，返回向下层传递的下游梯度。
  - softmax + cross entropy loss 的梯度可以化简为 (p-Y)，其中 p 为预测概率，Y 为标签的 one-hot 编码。
  - 由于交叉熵损失 cross entropy loss 采用了平均损失，梯度相应地除以 batchsize。

- 无需更新参数


##### 辅助函数
- Softmax()：<br>
  输入每个类别的预测值 Z，返回转化后的概率值 p。
  - 为保证数值稳定性，先将每个数据均减去当前 batch 中的最大值。

- CrossEntropyLoss()：<br>
  提供 预测概率 p 和 one-hot标签 Y，返回交叉熵损失。
  - 为了防止 log(0)，对 p 取对数前加上小量 1e-15。



#### models.py：层组织成模型、模型存取
定义了 SimpleModel 类，实现 前向传播、反向传播、参数更新、模型保存与加载。<br>
支持自定义模型架构（含正则化方式和正则化强度）。

- 初始化 \_\_init__(layers, regularization_method, lambdal)：<br>
  - 参数：
    - layers：列表，内容为模型结构（模型中的各个层）。
    - regularization_method：正则化方法，字符串 "L1" 或 "L2"。
    - lambdal：正则化强度。
  - 把各个“层”的实例存入“层列表” self.layers。
    - 最后一层的 forward() 必须接受 X, Y 两个参数，返回包含字段 "accuracy" 和 "loss" 的字典。
    - 最后一层的 backward() 必须不接受任何参数，只返回当前层向下传播的梯度。
    - 其余层的 forward() 必须只接受 X 一个参数，只传回一个返回值。
    - 其余层的 backward() 必须只接受 上游梯度 一个参数，只返回当前层向下传播的梯度。
  - 存储正则化方法 self.method 和正则化强度 self.lambdal 。

- 前向传播 forward(X, Y)：<br>
  提供 X, Y，返回包含 "loss" 和 "accuracy" 的字典，"loss" 中包含了正则化项。
  - 逐层前向传播：
    - 对除了最后一层外的其他层调用 forward(X)，
    - 最后一层为损失层，调用 forward(X, Y) 计算损失和准确率，得到具有 "loss" 和 "accuracy" 的字典。
  - 累加正则化损失：
    - 遍历所有层，对所有定义了 regularization 方法的层调用 regularization ，将返回值累加到字典中的 "loss" 字段上。
    - 这里要求“正则化项”在各层之间是可加的。
    - 由于这里是累加的逻辑，可以对每一层指定不同的正则化强度 lambdal 值，以列表形式传入 \_\_init__() ，稍微修改累加逻辑即可。

- 反向传播 backward()：<br>
  完成反向传播流程，得到各层梯度。
  - 逐层反向传播：
    - 对最后一层损失层调用 backward()。
    - 对除了最后一层外的其他层调用 backward(grad) ，传递梯度。

- 参数更新 GDupdate(lr)：<br>
  提供学习率 lr，更新所有层中的可更新参数。
  - 遍历所有层，对具有 GDupdate() 方法的层，调用 GDupdate() 更新参数。


##### 保存和加载模型

- 保存模型 save(base_path)：<br>
  将模型元数据和参数保存到给定的 base_path 目录下，路径不存在会自动创建。
  - 保存元数据：将层的类型 "layer_types"、全连接层的维度 "fc_shape"、正则化方法 "regularization" 和正则化强度 "lambdal" 保存到 meta.npz。
  - 保存参数：将每个全连接层的 W 和 b 参数保存为独立的 .npz 文件（文件命名如 fc1.npz）。
  - 实际实现时，
    - 通过 type(layer).\_\_name__ 获取层的类型。
    - 直接通过 model 类中的 self.method 和 self.lambdal 获取正则化相关超参。
    - 遍历各层 ;ayer，对所有全连接层，使用 layer.W 和 layer.b 获取参数，用 layer.W.shape[0] 和 layer.W.shape[1] 获取输入、输出维数。

  - 存储格式：
    - base_path <br>
      - meta.npz （元数据）<br>
        - "layer_types": [ ..., ... ]
        - "fc_shape": [ \{ "din": ..., "dout": ... }, ... ]
        - "regularization"
        - "lambdal" 
      - fc1.npz （隐藏层1）<br>
        - "W":
        - "b":
      - fc2.npz （隐藏层2）<br>
        - "W":
        - "b":
      - ... （后续隐藏层）<br>
    - 读取时的格式必须和存储格式相同。


- 加载模型 load(base_path)：<br>
  从给定目录加载模型数据并重构模型，返回重建的 SimpleModel 实例。
  - 加载元数据，重新创建各个层：
    - 全连接层 FCLayer：
      - 需要由 "fc_shape" 知其输入和输出维数，创建 FCLayer 的实例
      - 需要额外从 fcX.npz 中获取全连接层的参数，修改 layer.W 和 layer.b。
    - 激活函数层 和 交叉熵损失层 的初始化无需额外参数，直接创建实例即可。
  - 最后将各个层放到列表中，获取 "regularization" 和 "lambdal" 字段，创建返回当前的 SimpleModel 类的实例。


- 获取模型的部分超参数 get_params()：<br>
  用于记录日志。
  - 以字典形式，返回模型的所有隐藏层维数、正则化方法 和 正则化强度。
  - 沿用 save() 的思路，获取每一个隐藏层的维数，存入列表中。



### 训练
#### lr_scheduler.py：学习率策略
定义了两种学习率策略，初始化后可通过 get_lr() 获取学习率。

##### 基类 scheduler
定义学习率调度器的通用接口。

- 初始化 \_\_init__(init_lr)：
  提供初始学习率 init_lr。
  - 初始化 self.init_lr（初始学习率）和 self.lr（当前学习率）。
  - 初始化步数 self.step_count，每调用一次学习率调度器，应该增加一次步数，由学习率调度器自身维护。
- get_lr()：返回当前学习率。
- get_params()：返回调度器参数字典，用于日志记录。


##### StepExpLR：学习率预热 + 固定步长的指数衰减
- 初始化 \_\_init__():
  - 参数：
    - init_lr：初始学习率。
    - step_interval：每 step_interval 步衰减一次学习率。
    - warm_up_steps：预热阶段的步数，预热阶段学习率从 0 开始线性增长。
    - gamma：衰减系数，gamma 越小衰减越快。
  - 调用父类的初始化函数。参数存入类的私有变量中。

- 获取学习率 get_lr()：<br>
  无需提供参数，返回学习率。
  - 每次调用时，self.step_count 自增。
  - 预热阶段：学习率从 0 开始线性增长，达到 self.warm_up_steps 步时增长到 self.init_lr。
  - 指数衰减阶段：每走 self.step_interval 步，学习率变成原本的 gamma 倍。

- 获取超参 get_params()：<br>
  无需提供参数，返回学习率调度器的超参字典。


##### MultiStepExpLR：可变步长的多阶段指数衰减

- 初始化 \_\_init__()：
  - 参数：
    - init_lr：列表，各阶段的初始学习率。
    - step_interval：列表，各阶段的衰减步长。
    - step_milestones：列表，各个阶段的步数上界。
    - gamma：列表，各阶段的衰减系数。
  - 由于 init_lr 是列表，故当前的学习率 self.lr 需要被设置为 init_lr[0]。
  - 额外定义表示当前“阶段数”的变量 self.stage 和当前阶段内的步数 self.in_step_count。
    - 在阶段 i，步数刚好小于 step_milestones[i]，应该采用 init_lr 等列表中下标为 i 的数值计算学习率。

- 获取学习率 get_lr()：<br>
  无需提供参数，返回学习率。
  - 每次调用时，self.step_count, self.in_step_count 自增。
  - 如果到达了阶段分界点：self.stage 自增，重置 self.in_step_count 和 self.lr。
  - 每个阶段内部的 self.in_step_count 积累到 self.step_interval[self.stage] 时，调整学习率。

- 获取超参 get_params()：<br>
  无需提供参数，返回学习率调度器的超参字典。



#### trainer.py：训练

##### Trainer类
Trainer(model, training_set, validation_proportion, lr_scheduler, batch_size, max_epochs, early_stop_tol, log_step, plot)：<br>
  - 参数：
    - model 模型实例
    - training_set 训练集
    - validation_proportion 验证集比例
    - lr_scheduler 学习率调度器的实例
    - batch_size 批大小
    - max_epochs 最大epoch数量
    - early_stop_tol 早停容忍度(损失和准确率指标连续tol轮未提升则终止)
    - log_step 指定模型每看过多少数据输出相关信息(对应控制台输出和最后 detailed training loss 图像)
    - plot 是否需要画图(用不同超参连续运行多轮训练时设为 False)


- 类初始化 \_\_init()__：
  - 存储传入的参数到类的私有变量中。
  - 调用 split_training_set() 划分训练集和测试集，存入 self.train_X/Y, self.valid_X/Y。
  - 初始化存储 历史损失loss 和 历史准确率acc 的列表。
  - 初始化另一些记录训练状态的变量。


- 组织训练流程 train()：
  - 输出：
    - 训练过程中，输出到终端：
      - 训练进度
      - 每 log_step 条训练集数据的平均损失
      - 每个 epoch 的训练集/验证集损失和准确率
      - 每个 epoch 的耗时
    - 训练过程中，输出到 output.txt：
      - 每个 epoch 的训练集/验证集损失和准确率。（窗口意外关闭后可以知道当前保存的模型是哪一个 epoch）
    - 训练完成后，输出：
      - 训练集/验证集损失图像
      - 训练集/验证集准确率图像
      - 训练集上较细粒度的损失图像（粒度由 log_step 指定）
    - 训练完成后，输出到 log.csv：
      - model, lr_scheduler, trainer 的超参
      - 每个 epoch 的训练集和验证集损失与准确率
      - 当前时间和训练耗时
    - 每训练一个 epoch，如果损失或准确率表现有提高，则分别把模型存入 trained_model_min_loss 和 trained_model_max_accuracy 目录。

  - 流程：
    - 调用 train() 前：
      - 已经在 \_\_init__() 中随机分割了训练集和验证集。
    - 一个 epoch：
      - 调用 reshuffle_train() 随机重排分割后的训练集
      - 对于 epoch > 0，调用 train_epoch() 训练。
      - 对于 epoch = 0，调用 run_evaluation() 计算初始损失和准确率，不进行训练。
    - 完成一个 epoch 训练：
      - 得到训练集/验证集损失和准确率，存储到历史记录列表中
      - 调用 save_stats() 将相关信息输出到终端和 output.txt。
      - 调用 save_and_stop()，根据 验证集的损失和准确率 判断是否需要 保存模型 或 早停。
    - 完成训练后：
      - 输出模型在 训练集/验证集 的 损失loss 和 准确率acc 上的最好表现。
      - 调用 save_log() 输出日志。
      - 调用 visualize() 可视化图像（如果 plot 设置为 True 才调用）。
      - 如果训练集长度为 0，则前面不会保存在验证集上表现最佳的模型，要在这里保存模型（到 trained_model 路径下）。


- 分割训练集和验证集 split_training_set()：<br>
  根据给定比例 self.validation_proportion ，从 training_set 中分割出训练集和验证集。<br>
  训练集和验证集存入  self.train_X, self.train_Y, self.valid_X, self.valid_Y，一共四个列表。<br>
  要求 training_set 为字典，具有 'images' 和 'labels' 两个键。
  - 先检查训练集中 X 和 Y 的数量是否相等。
  - 用 np.random.permutation 随机重排索引，打乱数据顺序。
  - 按照比例求验证集和训练集长度，存入 self.valid_len, self.train_len。
  - 若 验证集长度 不为 0，分割训练集和验证集。否则把数据全部作为训练集。


- 重排训练集 reshuffle_train()：<br>
  对 self.train_X 和 self.train_Y 打乱顺序重新排列。
  - 仍然使用 np.random.permutation 实现。


- 单轮训练 train_epoch()：<br>
  完成一个 epoch 的训练，训练过程中输出进度信息。
  - 根据给定的批大小 bs 从训练数据中截取一个 batch 的数据 X, Y，截取到末尾时用 min(training_len, idx+bs) 保证下标不会越界。
  - 调用 train_one_batch(X, Y) 在截取出的数据上训练，累加 train_one_batch(X, Y) 返回的一个 batch 上的损失和准确率（为了计算 epoch 的损失和准确率）。
  - 每完成 log_step 条数据会输出训练进度、学习率、训练集平均损失。
  - 注意这个函数会把 log_step 条数据的损失存入 self.detailed_train_loss_hist，但不会把一个 epoch 的 loss 和 acc 存入 self.train_loss_hist 和 self.train_acc_hist

- 单批次训练 train_one_batch(X_batch, Y_batch)：<br>
  提供 一个 batch 的训练数据 X_batch, Y_batch。<br>
  更新参数，返回（参数更新前的）模型在当前 batch 的训练数据上的 loss 和 accuracy。
  - 前向传播：调用 model.forward(X_batch, Y_batch)，获取包含当前 batch 的 loss 和 accuracy 的字典。
  - 反向传播：调用 model.backward()，得到各层参数的梯度。
  - 参数更新：先调用 lr_scheduler.get_lr() 获取学习率，再调用 model.GDupdate(lr) 更新参数。


- 保存模型和早停 save_and_stop()：<br>
  根据 loss 和 acc 的历史列表，决定是否需要保存当前 epoch 的模型，以及根据 early_stop_tol 决定是否需要早停。<br>
  会针对验证集上“损失最小化”和“准确率最大化”保存两个模型。<br>
  不需要早停返回 1，需要则返回 0。
  - 由于该函数在第 0 个 epoch（只计算初始损失，不训练）也会被调用，故一开始检查 存储历史记录的列表 的长度是否为 1，若是则直接返回 1，不保存模型。
  - 通过训练集的 loss 和 acc 的历史列表，判断是否需要保存当前 epoch 的模型。需要则调用 model.save() 保存。
  - 仍然通过训练集的 loss 和 acc 的历史列表，判断是否需要早停。
  - 早停的判断逻辑：连续 early_stop_tol 轮迭代，验证集 loss 没有下降且验证集 acc 没有提升。


- 输出每个 epoch 的统计信息 save_stats()：<br>
  需要提供“前缀” prefix（一般是 "train" 或 "validation"）和要输出的 loss 和 acc 值。
  将内容格式化后输出到终端和 .txt 文件。


- 训练日志 save_log()：<br>
  完成训练后，将超参和每一个 epoch 的 训练集/验证集 loss 和 acc 值存入 log.csv 中。不需要额外提供参数。
  - 使用 model.get_params() 和 lr_scheduler.get_params() 获取模型和学习率调度器的超参。
  - 从历史记录列表中获取 训练集/验证集 loss 和 acc 值。如果 self.valid_len = 0 则只获取训练集的数据。


- 可视化 visualize()：<br>
  根据 self.valid_len 调用不同可视化函数。
  - self.valid_len > 0，调用 visualize_with_validation()。
  - 否则，调用 visualize_without_validation()。
  
- 有验证集的可视化 visualize_with_validation()：<br>
  从存储历史 loss 和 acc 的列表中获取数据，绘图。<br>
  - 左图表示 训练集/验证集的 loss 变化，
  - 中图表示 训练集/验证集的 acc 变化，
  - 右图表示 训练集上更细粒度的 loss 变化（粒度由参数 log_step 指定）。

- 无验证集的可视化 visualize_without_validation()：<br>
  用于正式训练只有训练集没有验证集时。<br>
  - 左图表示 训练集的 loss 变化，
  - 中图表示 训练集的 acc 变化，
  - 右图表示 训练集上更细粒度的 loss 变化（粒度由参数 log_step 指定）。


##### 非 Trainer 类的辅助函数
- 模型评估 RunEvaluation(X, Y, model)：<br>
  提供 模型 model 和 训练集/验证集/测试集数据 X, Y，返回损失和准确率。
  - 直接调用模型的 forward() 方法实现。



### 测试
#### load_and_test.py
  在 DATA_DIR 中填写 CIFAR10 所在目录，在 MODEL_DIR 中填写模型保存目录。<br>
  运行 load_and_test.py 即可看到输出结果。
  - 调用 Load_CIFAR10.py 中定义的 load_cifar10(DATA_DIR) 从给定目录中加载测试集 test_X, test_Y 。
  - 调用 models.py 中定义的 load(MODEL_DIR) 从给定目录中获取模型实例 trained_mlp。
  - 调用 trainer.py 中定义的 run_evaluation(test_X, test_Y, trained_mlp) 获取模型在训练集上的损失和准确率。



### 参数查找
#### main_search.py
  实现了对 隐藏层维数 dim_hidden，初始学习率 init_lr，学习率衰减系数 gamma 和 正则化强度 weight_decay 的查找。
  - 使用字典 hyperparams 定义超参数搜索空间。
    - 键为超参数的名称。
    - 值为列表，存放待搜索的超参数。
  - 通过 product(*hyperparams.values()) 遍历所有可能的超参组合。
  - 使用给定的超参定义并启动训练器。
  - 每一次训练会输出 相关超参信息 和 训练集/验证集上的损失和准确率 到 param_search_log.csv 中。
  - 遍历过程中会将验证集上 准确率 最好的模型保存到 searched_model_max_accuracy 文件夹下。