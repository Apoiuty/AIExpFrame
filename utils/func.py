import math
from .dataset import CVSplitDataset
from .utils import EarlyStopping
import os
import gc
import pathlib
import random
import re
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model import *


def add_weight_decay(model, weight_decay):
    """
    对参数施加L2正则化，其中不对偏置进行正则化
    :param model: 模型
    :param weight_decay 权重衰减
    :return:
    """
    weight = []
    bias = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight.append(param)
        else:
            bias.append(param)

    params = [{'params': weight, 'weight_decay': weight_decay}, {'params': bias}]
    return params


def set_seed(seed):
    """
    设置模型训练时的随机数种子；用于实现模型的可复现性
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    torch.use_deterministic_algorithms(True)
    # 设置CNN卷积使用固定算法
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_seed):
    """
    设置dataloader的随机数：
    使用方法：
    g = torch.Generator()
    g.manual_seed(0)
    DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    :param worker_seed:
    :return:
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(model, train_loader, loss_func, optimizer, device, tqdm_enable=False):
    """
    模型训练
    :param model:
    :param train_loader:
    :param device:
    :param optimizer:
    :param loss_func
    :param tqdm_enable:是否使用tqdm
    :return:
    """
    loss_all = 0
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for data in tqdm(train_loader, disable=not tqdm_enable):
        data = data.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            output = model(data)
            # todo:定义自己的loss项
            loss = loss_func(output, data.y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        del loss
        del data
        gc.collect()
        torch.cuda.empty_cache()
    return loss_all / len(train_loader.dataset)


def test(model, loader, loss_func, device, tqdm_enable=True):
    """
    模型测试
    :param model:
    :param loader:
    :param device:
    :loss_func
    :return:
    """
    model.eval()
    loss_all = 0
    with torch.inference_mode():
        for data in tqdm(loader, disable=not tqdm_enable):
            data = data.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(data)
                loss = loss_func(output, data.y)
                loss_all += loss.item()
            del loss
            del data
            gc.collect()
            torch.cuda.empty_cache()
    return loss_all / len(loader.dataset)


def find_idle_cuda():
    """
    返回计算利用率较低的torch.device
    :return:
    """
    patten = r'(?s)(\d)\s*NVIDIA.*?(\d*)%\s*Default'
    usage_dict = defaultdict(int)

    for i in range(5):
        time.sleep(0.5)
        result = os.popen('nvidia-smi').read()
        name_usage = re.findall(patten, result)

        if not name_usage:
            return torch.device('cpu')
        for name, usage in name_usage:
            usage_dict[name] += int(usage)

    min_usage = math.inf
    min_name = ''
    for name, usage in usage_dict.items():
        if usage < min_usage:
            min_usage = usage
            min_name = name

    return torch.device(f'cuda:{min_name}')


def wait_cuda(require_mem=23, overall_mem=24):
    """
    等待GPU运行完成
    :param require_mem:所要求的内存
    :overall_mem:每个显存设备的显存大小
    :return:
    """
    require_mem *= 1000
    overall_mem *= 1000
    while True:
        patten = r'(?s)(\d)\s*NVIDIA.*?(\d*)MiB\s*/'
        usage_dict = defaultdict(int)

        print(f'Try to get GPU at {datetime.now()}')
        for i in range(5):
            time.sleep(.5)
            result = os.popen('nvidia-smi').read()
            name_usage = re.findall(patten, result)

            if not name_usage:
                return torch.device('cpu')
            for name, usage in name_usage:
                usage_dict[name] += int(usage) // 5

        min_usage = math.inf
        min_name = ''
        for name, usage in usage_dict.items():
            if usage < min_usage:
                min_usage = usage
                min_name = name

        if overall_mem - usage_dict[min_name] >= require_mem:
            return torch.device(f'cuda:{min_name}')
        else:
            time.sleep(random.randint(10, 60))


def train_model(model, epochs, dataset_name, seed, exp_dir=pathlib.Path('.'), init_lr=1e-3, batch_size=32,
                weight_decay=1e-4, early_stop_patience=100, device=None):
    set_seed(seed)
    early_stop_patience = early_stop_patience
    exp_name = f'{model.__name__}_{dataset_name}'
    print(f'Start {exp_name}')
    device = find_idle_cuda() if device is None else device

    # 数据集变换和划分
    dataset_path = f'data/'
    # todo:自己的数据
    dataset = TUDataset(dataset_path, name=dataset_name, use_edge_attr=True, use_node_attr=True)
    dataset = dataset.shuffle()

    for train_data, val_data, test_data in CVSplitDataset(dataset, cv=1, val=True):
        # 数据集加载
        train_loader = DataLoader(train_data, batch_size=batch_size)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        # loss
        loss = F.mse_loss
        optimizer = torch.optim.SGD(model.parameters, lr=init_lr, momentum=.9, weight_decay=weight_decay)
        early_stopping = EarlyStopping(early_stop_patience, mode='max')
        for epoch in tqdm(range(epochs)):
            train_metric = train(model, train_loader, loss, optimizer, device)
            val_metric = test(model, val_loader, loss, device)

            if early_stopping(val_metric):
                break
