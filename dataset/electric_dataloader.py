import config
from dataset.electric_dataset import ElectricDataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class ElectricDataloader:
    def __init__(self, preprocessor):
        const = config.Constant
        hp = config.Hyperparameter

        train_dataset = ElectricDataset(preprocessor)

        split = int(const.TRAIN_SIZE * len(train_dataset))
        idx_list = list(range(len(train_dataset)))
        train_idx, val_idx = idx_list[:split], idx_list[split:]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        self.__train_loader = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, sampler=train_sampler)
        self.__val_loader = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, sampler=val_sampler)

    @property
    def train_loader(self):
        return self.__train_loader

    @property
    def val_loader(self):
        return self.__val_loader
