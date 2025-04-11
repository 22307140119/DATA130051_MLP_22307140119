import layers as ly
import load_CIFAR10 as ld
import models as mdl
import trainer    # new!
import lr_scheduler as lrs
import numpy as np

DIR = r"E:\cifar-10-batches-py"    # 数据集路径


dim_hidden = [512, 256]

bs = 64
init_lr = 6e-2          # 初始学习率
step_interval = 10000    # 每走多少步下降一次学习率
warm_up_steps = 0
gamma = 0.01             # 学习率下降系数
weight_decay = 1e-4     # 正则化强度
num_epochs = 64

d = 3072    # 输入维数
h1 = dim_hidden[0]
h2 = dim_hidden[1]
c = 10      # 类别数量



if __name__ == "__main__":
    train_X, train_Y, test_X, test_Y = ld.load_cifar10(DIR)
    # length = 5000 # 训练集 + 验证集条数
    # train_X, train_Y, test_X, test_Y = train_X[0:length], train_Y[0:length], test_X[0:length], test_Y[0:length]

    np.random.seed(1)

    
    # 构建模型
    mlp = mdl.SimpleModel(
        layers=[
            ly.FCLayer(din=d, dout=h1),
            # ly.ReLU(),
            # ly.Tanh(),
            # ly.LeakyReLU(0.01),
            ly.Sigmoid(),
            ly.FCLayer(din=h1, dout=h2),
            # ly.ReLU(),
            # ly.Tanh(),
            # ly.LeakyReLU(0.01),
            ly.Sigmoid(),
            ly.FCLayer(din=h2, dout=c),
            ly.Softmax_CrossEntropy()
        ],
        regularization_method='L2',
        lambdal=weight_decay
    )

    # lr_scheduler
    lr_scheduler = lrs.StepExpLR(init_lr=init_lr, step_interval=step_interval, warm_up_steps=warm_up_steps, gamma=gamma)
    # lr_scheduler = lrs.MultiStepExpLR(init_lr=[1e-2, 1e-3], step_interval=[100, 200], step_milestones=[500], gamma=[0.6, 0.8])

    # 训练
    my_trainer = trainer.Trainer(
        training_set={'images': train_X, 'labels':train_Y}, 
        validation_proportion=0.2,
        model=mlp, 
        lr_scheduler=lr_scheduler, 
        batch_size=bs, 
        max_epochs=num_epochs,
        early_stop_tol=4,
        log_step=len(train_X)/10,
        plot=True
    )
    my_trainer.train()


    # trained_mlp = SimpleModel.load('[path to your model]')
    # testing_loss, testing_accuracy = train.RunEvaluation(test_X, test_Y, trained_mlp)
    # print(f"testing_loss: {testing_loss}, testing_accuracy: {testing_accuracy}")
