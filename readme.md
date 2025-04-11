# 从零开始构建三层神经网络分类器，实现图像分类

### 代码架构简述
详见 details.md

- load_CIFAR10.py：读取数据集，独立于其它文件。
  可以调用 load_cifar10(dir)，从 dir 目录中读取整个 CIFAR10 数据集。
  返回标准化后的训练集和数据集，Y 以 one-hot 编码形式存储。

- layers.py：“层”的实现，最底层的文件。
  定义了 全连接层、激活函数、损失函数，含 前向/反向传播。
  辅助函数 Softmax(), CrossEntropyLoss()。
  包含 正则化 的底层实现。
  包含 梯度下降 的底层实现。

- models.py："层"组织成模型、模型存取，基于 layers.py
  定义了 SimpleModel 类，实现 前向传播、反向传播、参数更新、模型保存与加载。

- lr_scheduler.py：学习率策略，独立于其它文件。
  实现了 StepExpLR 和 MultiStepExpLR 两个学习率调度器。
  StepExpLR 支持 学习率预热 + 固定步长的指数衰减。
  MultiStepExpLR 支持 可变步长的多阶段指数衰减。

- trainer.py：训练，基于“模型”和“学习率调度器”。
  实现完整训练流程。
  含 训练集和验证集划分。
  辅助函数 run_evaluation()，评估模型在给定数据集上的性能。

- main.py：完整训练流程，依赖于上述所有文件。

- main_search.py：选择超参。

- load_and_test.py：加载模型并测试，基于 models.py 和 trainer.py。


### 输出说明
- 输出的训练集准确率是直接使用前向传播结果计算的，不会额外在训练集上评估。开始训练时的训练集准确率可能明显偏低。

- log.csv
    trainer.train() 日志的输出文件，在每一次训练完成后写入，不会被清空。

- output.txt
    trainer.train() 日志的输出文件，边训练边写入，在每一次训练开始时被清空。

- trained_model_min_loss
    运行 trainer.train() 后，验证集上准确率最高的模型会保存到该文件夹下。

- trained_model_max_accuracy
    运行 trainer.train() 后，验证集上损失最小的模型会保存到该文件夹下。

- param_search_log.csv
    main_search.py 日志的输出文件，不会被清空。

- searched_model_max_accuracy
    运行 main_search.py 后，验证集上准确率最高的模型会保存到该文件夹下。


### 启动训练
流程：加载数据、定义模型、定义学习率调度器、定义训练器、开始训练。

- 加载数据
  - 设置数据集路径
  - 调用 load_CIFAR10.load_cifar10() 获取数据

    ```python
    import load_CIFAR10 as ld

    DIR = r"cifar-10-batches-py"
    train_X, train_Y, test_X, test_Y = ld.load_cifar10(DIR)
    ```

- 定义模型：使用 SimpleModel 类。
  - layers: 列表，内容为模型结构（模型中的各个层）。
  - 可用的层：
    - layer.FCLayer(din, dout) 
    - layer.ReLU()
    - layer.Tanh()
    - layer.LeakyReLU(alpha)
    - layer.Sigmoid()
    - layer.Softmax_CrossEntropy()
  - regularization_method: 正则化方法。支持"L1", "L2", None。默认为 None。
  - lambdal: 正则化强度 λ。默认为 0。

    ```python
    import layers as ly
    import models as mdl

    mlp = mdl.SimpleModel(
        layers=[
            ly.FCLayer(din=3072, dout=512),
            ly.Sigmoid(),
            ly.FCLayer(din=512, dout=128),
            ly.Sigmoid(),
            ly.FCLayer(din=128, dout=10),
            ly.Softmax_CrossEntropy()
        ],
        regularization_method='L2',
        lambdal=1e-4
    )
    ```

- 定义学习率调度器 <br>
  定义了两种学习率调度器。
  - StepExpLR: 学习率预热 + 固定步长的指数衰减
    - init_lr: 初始学习率。
    - step_interval: 每 step_interval 步衰减一次学习率。
    - warm_up_steps: 预热阶段的步数，预热阶段学习率从 0 开始线性增长。默认为 0。
    - gamma: 学习率指数衰减的底数，gamma 越小衰减越快。默认为 0.1。
    - 注意：若设置了 warm_up_steps，会在 warm_up_steps + step_interval 步时首次减少学习率。

  - MultiStepExpLR: 可变步长的多阶段指数衰减
    - init_lr: 列表，各阶段的初始学习率。
    - step_interval: 列表，各阶段的衰减步长。
    - step_milestones: 列表，各个阶段的步数上界。
    - gamma: 列表，各阶段的衰减系数。
    - 注意：阶段 i 对应总步数恰好小于 step_milestones[i] 的阶段，初始学习率为 init_lr[i]，衰减步长为 step_interval[i]，衰减系数为 gamma[i]。

    ```python
    import lr_scheduler as lrs

    lr_scheduler = lrs.StepExpLR(
        init_lr=1e-2, 
        step_interval=1000, 
        warm_up_steps=100, 
        gamma=0.1
        )

    lr_scheduler = lrs.MultiStepExpLR(
        init_lr=[1e-2, 1e-3], 
        step_interval=[100, 200], 
        step_milestones=[500], 
        gamma=[0.6, 0.8]
        )
    ```

- 定义训练器并开始训练
  - 训练器 trainer 类
    - model: 前面定义的模型实例。
    - training_set: 前面读入的训练集。
    - validation_proportion: 验证集比例。默认为 0.2。
    - lr_scheduler: 前面定义的学习率调度器的实例。
    - batch_size: 批大小。默认为 64。
    - max_epochs: 最大 epoch 数量。默认为 4。
    - early_stop_tol: 早停容忍度。损失和准确率指标连续tol轮未提升则终止。默认为 4。
    - log_step: 指定模型每看过多少数据输出相关信息，对应控制台输出和最后 detailed training loss 图像。默认为 5000。
    - plot: 是否需要画图。需要用循环结构连续运行多轮训练时建议设为 False。
  - trianer.train() 开始训练
    - 返回验证集上的最好损失和准确率。
    - 自动保存最佳损失对应的模型到 trained_model_min_loss 文件夹下。
    - 自动保存最佳损失对应的模型到 trained_model_max_accuracy 文件夹下。
    - 单次训练中，每一个 epoch 的 训练集/验证集 损失和准确率 输出到 output.txt 文件，下一次开始训练时 output.txt 被清空。
    - 完成训练后，将日志输出到 log.csv 文件，log.csv 不会被清空。
    - 以上所有文件均会（在不存在时）自动创建。

    ```python
    import trainer

    my_trainer = trainer.Trainer(
        training_set={'images': train_X, 'labels':train_Y}, 
        validation_proportion=0.2,
        model=mlp, 
        lr_scheduler=lr_scheduler, 
        batch_size=64, 
        max_epochs=16,
        early_stop_tol=4,
        log_step=len(train_X)/10,
        plot=True
    )
    my_trainer.train()
    ```


### 测试：load_and_test.py
  在 DATA_DIR 中填写 CIFAR10 所在目录，在 MODEL_DIR 中填写模型保存目录。<br>
  运行 load_and_test.py 即可看到输出结果。


### 参数查找：main_search.py
  实现了对 隐藏层维数 dim_hidden，初始学习率 init_lr，学习率衰减系数 gamma 和 正则化强度 weight_decay 的查找。
  - 使用字典 hyperparams 定义超参数搜索空间。
    - 键为超参数的名称。
    - 值为列表，存放待搜索的超参数。
  - 每一次训练会输出 相关超参信息 和 训练集/验证集上的损失和准确率 到 param_search_log.csv 中。
  - 遍历过程中会将验证集上 准确率 最好的模型保存到 searched_model_max_accuracy 文件夹下。


### 可能的扩展
- 增加新的层/激活函数/损失函数
    直接在 layers.py 的对应位置添加。
    需要注意 models.py 中 SimpleModel 类对层的一些约束。

- 增加新的正则化方式
    在 layers.py 中，修改 FCLayer 类的 regularization()。
        要求“正则化项”在各层之间是可加的。
        无需修改 GDUpdate()，直接在 regularization() 中将梯度存入 self.drdW 即可。
    无需修改 models.py。

- 对每一层使用不同的 正则化方法/正则化强度参数
    可以考虑以列表形式向 model 类中传入不同层的 正则化方法/正则化强度参数。
    修改 models.py 的 forward() 中和 regularization 相关的逻辑即可。
    无需修改 layers.py。

- 增加其它优化算法实现
    代码中直接将 SGD 拆分到最底层的“层”中实现，和 layers.py, models.py, trainer.py 相关，并没有为其它优化算法预留接口。
    增加其它优化算法可能需要修改 layers.py, models.py, trainer.py 的相关部分。
    或者新增一个“优化器”文件，修改 trainer.py 中的 train_one_batch() 等。

- 增加其它学习率调度器
    在 lr_scheduler 中添加即可。

- 增加数据增强
    可以添加在 load_CIFAR10.py 的 preprocess_data() 中。
    也可以直接加在 main.py 中。
    也可以在 trainer.py 中添加函数，处理划分完成后的训练集。