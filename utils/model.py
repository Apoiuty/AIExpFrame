# 在该文件中定义要实验的模型
from torch import nn


class MyModel(nn.Module):
    """
    各种模型
    """
    pass


def model_factory(model_class):
    return MyModel
