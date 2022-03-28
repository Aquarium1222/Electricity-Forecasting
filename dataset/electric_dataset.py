import pandas as pd
import numpy as np
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler

import config


class ElectricDataset(data.Dataset):
    def __init__(self, preprocessor):
        const = config.Constant
        hp = config.Hyperparameter
        df1 = pd.read_csv(const.RESERVE_MARGIN[0])
        df2 = pd.read_csv(const.RESERVE_MARGIN[1])
        df = pd.concat([df1, df2], axis=0).drop_duplicates().drop(columns=['日期', '備轉容量率(%)'])
        y = preprocessor.preprocessing(np.expand_dims(df['備轉容量(萬瓩)'].to_numpy(), axis=1) * 10, 'y')
        x = preprocessor.preprocessing(np.expand_dims(df['備轉容量(萬瓩)'].to_numpy(), axis=1) * 10, 'x')
        # x = preprocessor.preprocessing(df.to_numpy(), 'x')
        self.__data = []
        self.__result = []
        for i in range(hp.INPUT_SEQ_LEN, len(x) - (hp.OUTPUT_SEQ_LEN - 1)):
            self.__data.append(x[i-hp.INPUT_SEQ_LEN:i])
            self.__result.append(y[i:i+hp.OUTPUT_SEQ_LEN])
        self.__data = np.array(self.__data)
        self.__result = np.array(self.__result)

    def __getitem__(self, item):
        return self.__data[item], self.__result[item]

    def __len__(self):
        return len(self.__data)
