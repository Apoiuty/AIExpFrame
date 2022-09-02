import math


class EarlyStopping:
    """
    早停实现
    """

    def __init__(self, patience, mode='max'):
        self.patience = patience
        self.cnt = 0
        self.mode = mode
        if mode == 'max':
            self.best_value = -math.inf
        elif mode == 'min':
            self.best_value = math.inf
        else:
            raise ValueError("Mode must be 'max' or 'min'")

    def __call__(self, value):
        """
        每轮中使用验证集指标进行调用
        :param value:
        :return:
        """
        if self.mode == 'max':
            if value > self.best_value:
                self.cnt = 0
                self.best_value = value
            else:
                self.cnt += 1
        elif self.mode == 'min':
            if value < self.best_value:
                self.cnt = 0
                self.best_value = value
            else:
                self.cnt += 1

        if self.cnt > self.patience:
            # 停止
            return True
        else:
            # 不停止
            return False


class ModelLogging:

    def __init__(self, log_file):
        """"""
        pass
