from dataset.electric_dataloader import ElectricDataloader
from dataset.preprocessor import Preprocessor


class WrappedDataloader:
    def __init__(self, dataloader, func):
        self.dataloader = dataloader
        self.func = func

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        iter_dataloader = iter(self.dataloader)
        for batch in iter_dataloader:
            yield self.func(*batch)


def preprocess(x, y):
    return x.transpose(0, 1), y.transpose(0, 1)


preprocessor = Preprocessor()
dataloader = ElectricDataloader(preprocessor)
train_loader = dataloader.train_loader
val_loader = dataloader.val_loader

train_loader = WrappedDataloader(train_loader, preprocess)
val_loader = WrappedDataloader(val_loader, preprocess)
