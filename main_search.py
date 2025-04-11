import layers as ly
import load_CIFAR10 as ld
import models as mdl
import trainer
import lr_scheduler as lrs
import numpy as np
from itertools import product
import os
import shutil

DIR = r"E:\cifar-10-batches-py"
output_csv = 'param_search_log'

# 需要搜索的超参数
hyperparams = {
    # "dim_hidden": [[512, 256]],
    "dim_hidden": [[12, 6]],
    "init_lr": [1e-1, 1e-2],
    "gamma": [1, 0.5, 0.2],
    "weight_decay": [1e-4, 1e-2],
}

bs = 64
step_interval = 10000
warm_up_steps = 3
num_epochs = 1

d = 3072
c = 10

if __name__ == "__main__":
    train_X, train_Y, test_X, test_Y = ld.load_cifar10(DIR)
    length = 5000 # 训练集 + 验证集条数
    train_X, train_Y, test_X, test_Y = train_X[0:length], train_Y[0:length], test_X[0:length], test_Y[0:length]

    np.random.seed(0)
    max_valid_acc = 0

    # 如果结果文件不存在，创建文件
    if not os.path.exists(f"{output_csv}.csv"):
        m = 'w'
    else:
        m = 'a'
    with open(f"{output_csv}.csv", m) as f:
        f.write("\nhidden_size, init_lr, gamma, weight_decay, min_val_loss, max_val_acc\n")

    # 遍历所有参数组合
    for params in product(*hyperparams.values()):
        dim_hidden = params[0]
        init_lr = params[1]
        gamma = params[2]
        weight_decay = params[3]
        print(f"\ndim_hidden: \t{dim_hidden}")
        print(f"init_lr: \t{init_lr}")
        print(f"gamma: \t{gamma}\n")
        print(f"weight_decay: \t{weight_decay}")

        # 初始化模型
        h1, h2 = dim_hidden
        mlp = mdl.SimpleModel(
            layers=[
                ly.FCLayer(din=d, dout=h1),
                ly.ReLU(),
                ly.FCLayer(din=h1, dout=h2),
                ly.ReLU(),
                ly.FCLayer(din=h2, dout=c),
                ly.Softmax_CrossEntropy()
            ],
            regularization_method='L2',
            lambdal=weight_decay
        )

        # 学习率调度器
        lr_scheduler = lrs.StepExpLR(
            init_lr=init_lr,
            step_interval=step_interval,
            warm_up_steps=warm_up_steps,
            gamma=gamma
        )

        # 初始化Trainer
        my_trainer = trainer.Trainer(
            training_set={'images': train_X, 'labels': train_Y},
            validation_proportion=0.2,
            model=mlp,
            lr_scheduler=lr_scheduler,
            batch_size=bs,
            max_epochs=num_epochs,
            early_stop_tol=4,
            log_step=5000,
            plot=False
        )

        # 训练并获取结果
        valid_loss, valid_acc = my_trainer.train()

        # 记录结果
        with open(f"{output_csv}.csv", "a") as f:
            f.write(f"{dim_hidden}, {init_lr}, {gamma}, {weight_decay}, {valid_loss: .4f}, {valid_acc: .4f}\n")

        # 保存最佳模型
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc

            target_dir = "searched_model_max_accuracy"
            source_dir = "trained_model_max_accuracy"

            # 删除旧的模型
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            
            # 复制 trainer() 中存储的新模型到目标路径
            shutil.copytree(source_dir, target_dir)
            print(f"model saved at: {target_dir}")

    print(f"\nmax validation accuracy of parameter searching: {max_valid_acc}")
    print(f"parameter searching history saved at '{output_csv}.csv'.\n")

