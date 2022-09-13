from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from torch.utils.data import Dataset


class CVSplitDataset:
    """
    交叉验证分割数据集
    """

    def __init__(self, dataset, cv=10, train_raio=.8, exp_num=1, val=False):
        """
        :param dataset:
        :param cv:
        :param train_raio:
        :param exp_num: 实验次数
        """
        self.train_ratio = train_raio
        self.cv = cv
        # 验证集
        self.val = val
        self.dataset = dataset
        self.n = exp_num

    def __iter__(self):
        if self.cv == 1:
            for i in range(self.n):
                train_set, test_set = train_test_split(self.dataset, train_size=self.train_ratio)
                if not self.val:
                    yield train_set, test_set
                else:
                    val = int(len(test_set) // 2)
                    yield train_set, test_set[:val], test_set[val:]
        else:
            if self.n == 1:
                cv = KFold(n_splits=self.cv, shuffle=True)
                for train_index, test_index in cv.split(self.dataset):
                    if not self.val:
                        yield self.dataset[train_index], self.dataset[test_index]
                    else:
                        val = int(len(test_index) // 2)
                        yield self.dataset[train_index], self.dataset[test_index[:val]], self.dataset[test_index[val:]]
            else:
                cv = RepeatedKFold(n_splits=self.cv, n_repeats=self.n)
                for train_index, test_index in cv.split(self.dataset):
                    if not self.val:
                        yield self.dataset[train_index], self.dataset[test_index]
                    else:
                        val = int(len(test_index) // 2)
                        yield self.dataset[train_index], self.dataset[test_index[:val]], self.dataset[test_index[val:]]


class MyDataset(Dataset):
    """
    实现自己的数据集
    """
    pass
