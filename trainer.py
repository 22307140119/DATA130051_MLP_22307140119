import models as mdl
import numpy as np
import matplotlib.pyplot as plt

# for timing
import time
from datetime import datetime

# for writing into csv
import pandas as pd
import os

class Trainer:
    def __init__(self, model, training_set, validation_proportion=0.2, 
                 lr_scheduler=None, batch_size=64, max_epochs=4, 
                 early_stop_tol=4, log_step=5000, plot=True):
        # 初始化参数
        self.model = model
        self.validation_proportion = validation_proportion
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stop_tol = early_stop_tol
        self.log_step = log_step
        self.plot = plot
        
        # 准备数据
        self.split_training_set(training_set)
        
        # 训练状态
        self.train_loss_hist = []
        self.train_acc_hist = []
        self.valid_loss_hist = []
        self.valid_acc_hist = []
        self.detailed_train_loss_hist = []
        self.current_epoch = 0
        self.lr = None
        self.start_time = None
        self.end_time = None


    # 根据给定比例 validation_proportion 分割训练集和验证集
    def split_training_set(self, training_set):
        train_X = training_set['images']
        train_Y = training_set['labels']
        length = len(train_Y)
        assert len(train_X) == length   # X Y 长度一样

        # 索引随机重排
        idx = np.random.permutation(np.arange(length))
        
        train_X = train_X[idx]    # 重排
        train_Y = train_Y[idx]

        self.valid_len = int(length * self.validation_proportion)
        self.train_len = length - self.valid_len
        if self.valid_len > 0:
            self.valid_X = train_X[:self.valid_len] # 验证集
            self.valid_Y = train_Y[:self.valid_len]
            self.train_X = train_X[self.valid_len:] # 训练集
            self.train_Y = train_Y[self.valid_len:]
        else:
            self.train_X = train_X
            self.train_Y = train_Y


    # 重排训练集
    def reshuffle_train(self):
        idx = np.random.permutation(np.arange(self.train_len))
        self.train_X = self.train_X[idx]    # 重排
        self.train_Y = self.train_Y[idx]



    # 一个 batch 的 前向、反向、参数更新，返回批损失和批准确率
    def train_one_batch(self, X_batch, Y_batch):
        # 前向传播
        forward_dict = self.model.forward(X_batch, Y_batch)
        batch_loss = forward_dict['loss']
        batch_accuracy = forward_dict['accuracy']

        # 反向传播
        grad = self.model.backward()

        # 参数更新
        self.lr = self.lr_scheduler.get_lr()
        self.model.GDupdate(self.lr)

        return batch_loss, batch_accuracy


    def train_epoch(self):
        idx = 0
        epoch_sum_loss = 0
        epoch_sum_correct = 0
        training_len = len(self.train_X)

        detailed_sum_loss = 0
        detailed_lower = 0
        detailed_upper = 0

        while idx < training_len:
            # 一个 batch, 处理加上 bs 后超界的问题
            # 每跑 log_step 条输出信息
            if int(idx/self.log_step) - int((idx-self.batch_size)/self.log_step) == 1 or idx >= training_len-self.batch_size:
                print(f"\t{idx}/{training_len}", end='\t', flush=True)    # 输出当前训练条数
                print(f"lr: {self.lr: .6f}", end='\t')
                detailed_upper = idx
                detailed_avg_loss = detailed_sum_loss/(detailed_upper-detailed_lower)
                self.detailed_train_loss_hist.append(detailed_avg_loss) # 计算 log_step 条数据的平均损失
                detailed_sum_loss = 0
                detailed_lower = idx
                print(f"train subset avg loss: {detailed_avg_loss: .4f}")

            lower = idx
            upper = min(training_len, idx+self.batch_size)  # 不超界
            X = self.train_X[lower:upper]
            Y = self.train_Y[lower:upper]

            # batch_loss 是平均损失， batch_accuracy 是正确率
            batch_loss, batch_accuracy = self.train_one_batch(X, Y)

            idx += self.batch_size
            epoch_sum_loss += batch_loss * (upper-lower)
            epoch_sum_correct += batch_accuracy * (upper-lower)
            detailed_sum_loss += batch_loss * (upper-lower)      # 累加

        # 一个 epoch 训练结束
        # 训练集损失和准确率，直接取平均算
        training_loss = epoch_sum_loss/training_len
        training_accuracy = epoch_sum_correct/training_len
        return training_loss, training_accuracy



    # 根据 loss 和 acc 的历史列表，决定是否保存模型和早停
    # 返回是否需要继续训练（返回0代表需要早停）
    def save_and_stop(self):
        length = len(self.valid_loss_hist)
        assert length == len(self.valid_acc_hist)  # 检查长度相等
        if length == 1: # 第 0 个 epoch，还没有正式开始训练，直接返回
            return 1

        # 保存模型
        # 列表长度非常有限，这里不会占据很多复杂度
        if self.valid_loss_hist[-1] == min(self.valid_loss_hist):
            self.model.save('trained_model_min_loss')
            print('\tModel saved for minimum loss.')
        if self.valid_acc_hist[-1] == max(self.valid_acc_hist):
            self.model.save('trained_model_max_accuracy')
            print('\tModel saved for maximum accuracy.')

        # 早停
        tol = self.early_stop_tol
        if length <= tol:
            return 1
        else:
            for i in range(0, tol):
                # 最近的 tol+1 次记录中，如果出现 loss下降 或 acc提升，继续训练
                if self.valid_loss_hist[-tol+i-1] > self.valid_loss_hist[-tol+i] or self.valid_acc_hist[-tol+i-1] < self.valid_acc_hist[-tol+i]:
                    return 1
            return 0



    # loss, accuracy 输出到终端和 output.txt 中
    def save_stats(self, prefix, loss, accuracy):
        print(f" {prefix} avg loss: \t{loss: .4f}", end = '')
        print(f"\t\t {prefix} accuracy: \t{accuracy: .4f}")
        # 输出到 .txt 文档中
        if self.current_epoch == 0:
            m = "w" # 第一个 epoch，清空文档
        else:
            m = "a"
        if prefix == 'train':   # 输出训练集准确率之前，额外输出 epoch 数
            with open("output.txt", m, encoding="utf-8") as f:
                f.write(f"Epoch {self.current_epoch}\n")
        with open("output.txt", 'a', encoding="utf-8") as f:
            f.write(f"\t {prefix} avg loss: \t{loss: .4f}")
            f.write(f"\t\t {prefix} accuracy: \t{accuracy: .4f}\n")
        

    # 输出信息到 .csv
    def save_log(self):
        csv_path = 'log.csv'
        
        # 参数
        params_dict  = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'time': int(self.end_time-self.start_time),
            'model_params': self.model.get_params(),
            'lr_params': self.lr_scheduler.get_params(),
            'trainer_params': {
                'bs': self.batch_size,
                'epoch': self.max_epochs,
                'len_train': self.train_len
            }
        }

        if self.valid_len > 0:
        # 训练集 / 验证集 的 损失 / 准确率
            history_data = {
                'Train Loss': self.train_loss_hist,
                'Train Accuracy': self.train_acc_hist,
                'Validation Loss': self.valid_loss_hist,
                'Validation Accuracy': self.valid_acc_hist
            }
        else:
            history_data = {
                'Train Loss': self.train_loss_hist,
                'Train Accuracy': self.train_acc_hist
            }

        # 写入
        # 文件存在，空行后在后面写入（mode='a'）
        if os.path.exists(csv_path):
            with open(csv_path, 'a') as f:
                f.write('\n')  # 空行分隔不同训练记录
            pd.DataFrame([params_dict]).to_csv(csv_path, mode='a', index=False)
        # 文件不存在，自动创建
        else:
            pd.DataFrame([params_dict]).to_csv(csv_path, index=False)
        history_df = pd.DataFrame(history_data).T
        history_df.to_csv(csv_path, mode='a', header=False)
        
        print("\nHyper parameters and history saved at 'log.csv'")
    
    

    # 可视化训练集和验证集，和训练集上更详细的损失情况
    def visualize_with_validation(self):
        # 绘制训练和验证损失（每个 epoch 的点）
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.plot(self.train_loss_hist, label="Training Loss")
        plt.plot(self.valid_loss_hist, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss vs. Epoch")

        # 绘制验证准确率
        plt.subplot(1, 3, 2)
        plt.plot(self.train_acc_hist, label="Training Accuracy")
        plt.plot(self.valid_acc_hist, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy vs. Epoch")

        # 训练集上更详细的损失情况
        plt.subplot(1, 3, 3)
        plt.plot(self.detailed_train_loss_hist, label="Detailed Training Loss")
        plt.legend()
        plt.show()

    # 只可视化训练集
    def visualize_without_validation(self):
        # 绘制训练和验证损失（每个 epoch 的点）
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.plot(self.train_loss_hist, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss vs. Epoch")

        # 绘制验证准确率
        plt.subplot(1, 3, 2)
        plt.plot(self.train_acc_hist, label="Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy vs. Epoch")

        # 训练集上更详细的损失情况
        plt.subplot(1, 3, 3)
        plt.plot(self.detailed_train_loss_hist, label="Detailed Training Loss")
        plt.legend()
        plt.show()


    # 根据是否有验证集选择可视化方式
    def visualize(self):
        if self.valid_len > 0:
            self.visualize_with_validation()
        else:
            self.visualize_without_validation()



    # 最主要的
    def train(self):
        self.start_time = time.time()
        
        for self.current_epoch in range(self.max_epochs+1):
            epoch_start = time.time()
            # 重排数据
            self.reshuffle_train()
            
            if self.current_epoch == 0:  # 一开始计算初始损失
                training_loss, training_acc = run_evaluation(self.train_X, self.train_Y, self.model)
                self.detailed_train_loss_hist.append(training_loss)
                print('Initial loss:')
            
            else:
                # 一个 epoch 的正式训练流程
                print(f"\n\nEpoch {self.current_epoch}:")
                # self.lr = self.lr_scheduler.get_lr(self.current_epoch)
                training_loss, training_acc = self.train_epoch()
            
                
            # 存历史记录到列表
            self.train_loss_hist.append(training_loss)
            self.train_acc_hist.append(training_acc)
            # 输出训练集 loss 和 accuracy 到终端和文件
            print()
            self.save_stats('train', training_loss, training_acc)

            if self.valid_len > 0:
                # 验证集损失和准确率
                validation_loss, validation_acc = run_evaluation(self.valid_X, self.valid_Y, self.model)
                
                # 存历史记录到列表
                self.valid_loss_hist.append(validation_loss)
                self.valid_acc_hist.append(validation_acc)
                # 输出验证集 loss 和 accuracy 到终端和文件
                self.save_stats('validation', validation_loss, validation_acc)
                print()

                # 根据验证集的 loss 和 accuracy 历史，决定是否保存模型或者早停
                if self.save_and_stop() == 0:
                    print("Overfitting, training treminated.")
                    break

            epoch_end = time.time()
            print(f"\tTime spent for this epoch: {epoch_end-epoch_start: .4f}s")

        
        # 结束训练后
        # 时间
        self.end_time = time.time()
        print(f'\nTotal time for training: {int(self.end_time - self.start_time)}s')

        # 输出最优数据
        print(f'min train loss: {min(self.train_loss_hist): .4f}', end='\t\t')
        print(f'max train accuracy: {max(self.train_acc_hist): .4f}')
        if self.valid_len > 0:
            print(f'min validation loss: {min(self.valid_loss_hist): .4f}', end='\t')
            print(f'max validation accuracy: {max(self.valid_acc_hist): .4f}')

        self.save_log()
        if self.plot:
            self.visualize()

        if self.valid_len == 0:
            self.model.save("trained_model")
            print('model saved.')
        else:
            return min(self.valid_loss_hist), max(self.valid_acc_hist)
            


# 评估验证集/测试集
def run_evaluation(X, Y, model):
    forward_dict = model.forward(X,Y)

    loss = forward_dict['loss']
    accuracy = forward_dict['accuracy']
    return loss, accuracy