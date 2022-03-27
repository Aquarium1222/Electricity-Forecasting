import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error

from config import *
from dataset.wrapped_dataloader import *
from model.seq2seq.encoder import Encoder
from model.seq2seq.decoder import Decoder
from model.seq2seq.seq2seq import Seq2Seq
from loss.rmse_loss import RMSELoss


def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for i, (x, y) in enumerate(dataloader):
        x = x.to(torch.float32).to(device)
        y = y.to(torch.float32).to(device)

        optimizer.zero_grad()

        y_pred = model(x, y)

        loss = criterion(y_pred, y)
        loss.backward()
        total_loss += loss.item()

        optimizer.step()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)

            y_pred = model(x, y, teacher_forcing_ratio=0)

            loss = criterion(y_pred, y)
            total_loss += loss.item()

            if i == len(dataloader) - 1:
                real_y = preprocessor.inverse(y[:, 0, :].detach().numpy(), 'y')
                real_y_pred = preprocessor.inverse(y_pred[:, 0, :].detach().numpy(), 'y')
    return real_y, real_y_pred, total_loss / len(dataloader)


# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    device = torch.device(device)

    encoder = Encoder()
    decoder = Decoder()
    model = Seq2Seq(encoder, decoder).to(device)
    print(model)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
    # optimizer = optim.Adam(model.parameters(), Hyperparameter.LEARNING_RATE, betas=(0.5, 0.999))
    criterion = RMSELoss()
    for e in range(Hyperparameter.EPOCH):
        train_loss = train(model, train_loader, optimizer, criterion)
        print('train loss:', train_loss)
        y, y_pred, eval_loss = evaluate(model, val_loader, criterion)
        print('eva loss:', eval_loss)
        print('actual loss:', eval_loss * 5474)
        print(y)
        print(y_pred)
