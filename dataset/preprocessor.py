from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    def __init__(self):
        self.__x_scaler = MinMaxScaler()
        self.__y_scaler = MinMaxScaler()

    def preprocessing(self, data, label):
        if label == 'x':
            scaler = self.__x_scaler
        else:
            scaler = self.__y_scaler
        tmp = data.reshape(-1, 1)
        scaler.fit(tmp)
        return scaler.transform(tmp).reshape(data.shape)

    def inverse(self, data, label):
        if label == 'x':
            scaler = self.__x_scaler
        else:
            scaler = self.__y_scaler
        return scaler.inverse_transform(data)
