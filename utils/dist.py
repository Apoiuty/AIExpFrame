import os
import pathlib
from functools import wraps

import torch.distributed as dist
import torch.multiprocessing as mp

from .func import train_model


# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follows:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def parallel_train_func(func, model_name, epochs, dataset_name, seed, exp_dir=pathlib.Path('.'), init_lr=1e-3,
                        batch_size=32,
                        weight_decay=1e-4, early_stop_patience=100):
    """
    并行训练的函数，返回接口为rank，world_size的函数
    :param func:
    :param model_name:
    :param epochs:
    :param dataset_name:
    :param seed:
    :param exp_dir:
    :param init_lr:
    :param batch_size:
    :param weight_decay:
    :param early_stop_patience:
    :return:
    """

    @wraps(func)
    def wrapper(rank, world_size):
        train_model(model_name, epochs, dataset_name, seed, exp_dir, init_lr, batch_size, weight_decay,
                    early_stop_patience, rank, world_size)

    return wrapper


def run_n_exp(train_func, world_size):
    """
    并行训练函数
    :param train_func:
    :param world_size:
    :return:
    """
    mp.spawn(train_func,
             args=(world_size,),
             nprocs=world_size,
             join=True)
