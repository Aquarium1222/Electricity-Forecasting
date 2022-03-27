import torch.nn as nn

from config import *


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.output_dim = Hyperparameter.OUTPUT_DIM
        self.embedding_dim = Hyperparameter.EMBEDDING_DIM
        self.hidden_dim = Hyperparameter.HIDDEN_DIM
        self.num_layers = Hyperparameter.NUM_LAYERS

        self.embedding = nn.Sequential(
            nn.Linear(self.output_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, dropout=0.5)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.init_weights()

    def forward(self, x, h, c):
        embedded = self.embedding(x)
        outputs, (h, c) = self.lstm(embedded, (h, c))
        pred = self.linear(outputs)
        return pred, h, c

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
