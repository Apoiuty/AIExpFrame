# 最后用所有的模型一起实验
import pathlib
from itertools import product
from multiprocessing import Pool, set_start_method
from utils.model import MyModel
from utils.func import train_model

if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    datasets = ['DD', 'IMDB-BINARY', 'IMDB-MULTI', 'NCI1', 'NCI109', 'PROTEINS']
    models = [MyModel]
    # 进程数依据CPU占用率来设定
    with Pool(processes=len(datasets)) as pool:
        seed = 42
        epoch = 500
        params = []

        for model, dataset in product(models, datasets):
            # todo:这里定义去训练模型
            pool.apply_async(train_model, (model, epoch, dataset, seed,), {'exp_dir': pathlib.Path('topk0.8')})

        pool.close()
        pool.join()
