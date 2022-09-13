import math
import pathlib
from copy import deepcopy

import torch
from wandb.sdk.wandb_run import Run


def wandb_id(model_name):
    """
    生成独一无二的wandb_id
    :param model_name:
    :return:
    """
    return str(abs(hash(model_name)))[:10]


class ModelLogging:

    def __init__(self, model, optim, lr_adj, log_file, patience, mode='max', wandb: Run = None):
        """
        :param wandb:
        :param model:
        :param optim:
        :param lr_adj:
        :param patience:
        :param log_file:
        """
        self.patience = patience
        self.cnt = 0
        self.model = model
        self.optim = optim
        self.lr_adj = lr_adj
        self.best_model = model
        self.best_optim = optim
        self.best_lr_scheduler = lr_adj
        self.wandb = wandb
        self.mode = mode
        if self.wandb:
            self.wandb.define_metric('test', summary=self.mode)

        self.log_file = pathlib.Path(log_file)
        self.train_list = []
        self.val_list = []
        self.test_list = []
        if mode == 'min':
            self.best_train = self.best_val = self.best_test = math.inf
        if mode == 'max':
            self.best_train = self.best_val = self.best_test = -math.inf

    def __call__(self, train_loss, val_loss, lr):
        """
        返回True进行测试，False早停，None pass
        :param lr:
        :param train_loss:
        :param val_loss:
        :return:
        """
        self.train_list.append(train_loss)
        self.val_list.append(val_loss)

        if self.mode == 'max':
            if val_loss > self.best_val:
                self.best_val = train_loss
                self.cnt = 0
                self.check_points()
            else:
                self.cnt += 1
            if train_loss > self.best_train:
                self.best_train = train_loss
        elif self.mode == 'min':
            if val_loss < self.best_val:
                self.best_val = train_loss
                self.cnt = 0
                self.check_points()
            else:
                self.cnt += 1
            if train_loss < self.best_train:
                self.best_train = train_loss

        # wandb中每个step的性能
        log = {
            'train': train_loss,
            'val': val_loss,
            'best_train': self.best_train,
            'best_val': self.best_val,
            'lr': lr
        }
        if self.wandb:
            # 如果使用wandb
            self.wandb.log(log)

        if self.cnt > self.patience:
            return True

    def check_points(self, test_loss=None):
        """
        保存最好的模型，优化器和lr_scheduler
        :return:
        """
        self.best_test = test_loss if test_loss else self.best_test
        self.best_model = deepcopy(self.model.state_dict())
        self.best_optim = deepcopy(self.optim.state_dict())
        self.best_lr_scheduler = deepcopy(self.lr_adj.state_dict())
        check_points = {
            'best_train': self.best_train,
            'best_val': self.best_val,
            'best_test': self.best_test,
            'best_model': self.best_model,
            'best_optim': self.best_optim,
            'best_lr_scheduler': self.best_lr_scheduler,
            'train_list': self.train_list,
            'val_list': self.val_list,
        }

        # wandb中最终的性能
        if self.wandb and test_loss:
            self.wandb.summary['test'] = test_loss

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(check_points, self.log_file)

    def load_model(self):
        """
        加载模型
        :return:
        """
        self.model.load_state_dict(self.best_model)
