import gc
import math
import os
import pathlib
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from wandb.wandb_run import Run

from .dataset import CVSplitDataset
from .model import model_factory
from .utils import ModelLogging


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def add_weight_decay(model, weight_decay):
    """
    对参数施加L2正则化，不对偏置进行正则化
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


def train(model, train_loader: Iterable, loss_func, optimizer, device, tqdm_enable=False, amp_enable=True):
    """
    模型训练
    :param amp_enable:
    :param model:
    :param train_loader:
    :param device:
    :param optimizer:
    :param loss_func
    :param tqdm_enable:是否使用tqdm
    :return:
    """
    loss_all = 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enable)
    for data in tqdm(train_loader, disable=not tqdm_enable):
        data = data.to(device)
        optimizer.zero_grad()
        # 混合精度训练
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enable):
            output = model(data)
            # todo:定义自己的loss项
            loss = loss_func(output, data.y)
            loss_all += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        del loss
        del data
        gc.collect()
        torch.cuda.empty_cache()
    return loss_all / len(train_loader.dataset)


def test(model, loader: Iterable, loss_func, device, tqdm_enable=True, amp_enable=True):
    """
    模型测试
    :param amp_enable:
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
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enable):
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


def train_model(model_config, epochs, dataset_name, seed, exp_dir=pathlib.Path('.'), init_lr=1e-3, batch_size=32,
                weight_decay=1e-4, early_stop_patience=100, device=None, rank=None, world_size=None,
                wandb_run: Run = None, amp_enable=True):
    set_seed(seed)
    early_stop_patience = early_stop_patience
    exp_dir = pathlib.Path(exp_dir)
    exp_name = exp_dir.stem
    print(f'Start {exp_name}')
    if rank is None and world_size is None:
        device = find_idle_cuda() if device is None else device
    else:
        device = rank
        setup(rank, world_size)

    # 数据集变换和划分
    dataset_path = f'data/'

    # todo:自己的数据
    dataset = TUDataset(dataset_path, name=dataset_name, use_edge_attr=True, use_node_attr=True)
    dataset = dataset.shuffle()
    model_config['in_channels'] = dataset.num_features
    model_config['out_channels'] = dataset.num_classes

    for train_data, val_data, test_data in CVSplitDataset(dataset, cv=1, val=True):
        # 数据集加载
        train_loader = DataLoader(train_data, batch_size=batch_size)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        # todo:loss
        loss = F.cross_entropy
        model = model_factory(model_config).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=.9, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, .1)

        logging = ModelLogging(model, optimizer, lr_scheduler, exp_dir / 'checkpoints.pt', early_stop_patience,
                               wandb=wandb_run)
        if wandb_run:
            config = {
                'lr': init_lr,
                'epochs': epochs,
                'batch_size': batch_size,
                'weight_decay': weight_decay,
                'early_stop_patience': early_stop_patience,
                'optim': optimizer.__class__.__name__,
                'lr_scheduler': lr_scheduler.__class__.__name__,
                'model': model.__class__.__name__,
            }
            config.update(model_config)
            wandb_run.config.update(config)

        # todo:模型生成方法
        if rank is not None:
            model = DDP(model, device_ids=[rank])
        for epoch in tqdm(range(epochs)):
            train_metric = train(model, train_loader, loss, optimizer, device, amp_enable=amp_enable)
            val_metric = test(model, val_loader, loss, device, amp_enable=amp_enable)

            # 学习率
            lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
            if logging(train_metric, val_metric, lr) or epoch + 1 == epochs:
                # load最好的模型
                logging.load_model()
                # 记录最好的模型
                test_metric = test(model, test_loader, loss, device, amp_enable=amp_enable)
                logging.check_points(test_loss=test_metric)

            print(f'Epoch{epoch}: Train{train_metric}, Val{val_metric}, LR{lr}')

        if rank and world_size:
            # 并行训练清理
            cleanup()

    if wandb_run:
        wandb_run.finish()
