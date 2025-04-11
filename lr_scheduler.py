class scheduler():
    def __init__(self, init_lr):
        self.init_lr = init_lr  # 初始设定的学习率
        self.lr = init_lr       # 当前要返回的学习率
        self.step_count = 0     # 步数，由学习率调度器维护

    def get_lr(self):       # 获取学习率
        pass
    
    def get_params(self):   # 获取 scheduler() 参数
        pass


# 随着训练的 step 数量增加，逐渐降低学习率
class StepExpLR(scheduler):
    def __init__(self, init_lr, step_interval, warm_up_steps=0, gamma=0.1):
        super().__init__(init_lr)
        self.step_interval = step_interval
        self.warm_up_steps = warm_up_steps
        self.gamma = gamma
        print('StepExpLR initialized')


    def get_lr(self):
        self.step_count += 1

        # warm up 阶段
        if self.step_count <= self.warm_up_steps:
            self.lr = self.step_count * self.init_lr / self.warm_up_steps
        
        # 指数衰减阶段
        else:
            if (self.step_count-self.warm_up_steps) % self.step_interval == 0:
                # print(f'lr adjusted.', end='\t')
                self.lr *= self.gamma
        return self.lr


    def get_params(self):
        return {'lr_scheduler': 'StepExpLR',
                'init_lr': self.init_lr, 
                'step_interval': self.step_interval,
                'warm_up_steps': self.warm_up_steps, 
                'gamma': self.gamma}


# 随着训练的 step 数量增加，分段逐渐降低学习率
# 参数都是列表， step_milestones 标记 init_lr 和 gamma 变化的时间点
class MultiStepExpLR(scheduler):
    def __init__(self, init_lr, step_interval, step_milestones, gamma=0.1):
        super().__init__(init_lr)
        self.lr = init_lr[0]    # 这里初始化方法不一样
        # 应该不需要检查
        # assert len(step_milestones) < len(init_lr)  # step_milestones 给出了 n 个点，实际上有 n+1 个阶段
        # assert len(step_milestones) < len(step_interval)
        # assert len(step_milestones) < len(gamma)

        self.step_interval = step_interval
        self.step_milestones = step_milestones
        self.gamma = gamma
        self.in_step_count = 0  # 单个区间内的步数
        self.stage = 0          # 当前“阶段”
        print('MultiStepExpLR initialized')
    

    def get_lr(self):
        self.step_count += 1
        self.in_step_count += 1

        # 分阶段指数衰减
        # 到达阶段分界点（先判断是否越界，再看计数是不是已经达到了 milestone）
        if self.stage < len(self.step_milestones) and self.step_count == self.step_milestones[self.stage]:
            self.stage += 1
            self.in_step_count = 1              # 阶段内部的步数
            self.lr = self.init_lr[self.stage]  # 该阶段的初始学习率
            print(f'\treached {self.step_count} steps, reset lr.')
        
        # 每个阶段内部，累积走到一定步数时，更新学习率
        # 步数由 self.step_interval[] 确定，更新系数由 self.gamma[] 确定
        if self.in_step_count % self.step_interval[self.stage] == 0:
            # print(f'lr adjusted.', end='\t')
            self.lr *= self.gamma[self.stage]
        
        return self.lr


    def get_params(self):
        return {'lr_scheduler': 'MultiStepExpLR',
                'init_lr': self.init_lr, 
                'step_milestones': self.step_milestones,
                'gamma': self.gamma}


# 这个不用了。
# 随着训练的 epoch 数量增加，逐渐降低学习率
# 支持设置 warm_up_epochs ，在 warm_up 期间学习率线性增长
class EpochExpLR(scheduler):
    def __init__(self, init_lr, warm_up_epochs, gamma):
        super().__init__(init_lr)
        self.warm_up_epochs = warm_up_epochs
        self.gamma = gamma
    
    def get_lr(self, epoch_count):
        # warm up 阶段
        if epoch_count <= self.warm_up_epochs:
            return epoch_count * self.lr / self.warm_up_epochs
        # 指数衰减阶段
        if epoch_count > self.warm_up_epochs:
            print(f'lr = {self.lr}')
            self.lr *= self.gamma
            return self.lr

    def get_params(self):
        return {'lr_scheduler': 'EpochExpLR',
                'init_lr': self.init_lr, 
                'warm_up_epochs': self.warm_up_epochs, 
                'gamma': self.gamma}