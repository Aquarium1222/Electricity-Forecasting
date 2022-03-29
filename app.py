import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

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

        y_pred = model(x, y).to(device)

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

            y_pred = model(x, y, teacher_forcing_ratio=0).to(device)

            loss = criterion(y_pred, y)
            total_loss += loss.item()

            if i == len(dataloader) - 1:
                real_y = preprocessor.inverse(y[:, 0, :].detach().cpu().numpy(), 'y')
                real_y_pred = preprocessor.inverse(y_pred[:, 0, :].detach().cpu().numpy(), 'y')
    return real_y, real_y_pred, total_loss / len(dataloader)


# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--training',
    #                     default='training_data.csv',
    #                     help='input training data file name')
    #
    # parser.add_argument('--output',
    #                     default='submission.csv',
    #                     help='output file name')
    # args = parser.parse_args()
    #
    # # The following part is an example.
    # # You can modify it at will.
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    device = torch.device(device)

    encoder = Encoder()
    decoder = Decoder()
    model = Seq2Seq(encoder, decoder).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), Hyperparameter.LEARNING_RATE, betas=(0.5, 0.999))
    criterion = RMSELoss()
    min_loss = 1
    trigger_count = 0
    for e in range(Hyperparameter.EPOCH):
        train_loss = train(model, train_loader, optimizer, criterion)
        y, y_pred, val_loss = evaluate(model, val_loader, criterion)

        if val_loss > min_loss:
            trigger_count += 1
        else:
            min_loss = val_loss
            trigger_count = 0

        print('\nEpoch:', e)
        print('train loss:', train_loss)
        print('eva loss:', val_loss)
        print('actual loss:', val_loss * 5474)
        print('trigger:', trigger_count)
        # print(y)
        # print(y_pred)

        if trigger_count >= Hyperparameter.PATIENCE:
            break
    torch.save(model, 'model.pt')

    from selenium import webdriver
    from selenium.webdriver.common.by import By
    import os
    import pandas as pd

    url = 'https://data.gov.tw/dataset/25850'
    chrome_options = webdriver.ChromeOptions()
    prefs = {'download.default_directory': os.getcwd() + '\\data'}
    chrome_options.add_experimental_option('prefs', prefs)

    browser = webdriver.Chrome(chrome_options=chrome_options)
    browser.get(url)
    ele = browser.find_element(By.CLASS_NAME, 'download-item')
    ele.find_element(By.CLASS_NAME, 'el-button--primary').click()
    browser.close()

    df1 = pd.read_csv(Constant.RESERVE_MARGIN)
    df2 = pd.read_csv(Constant.RESERVE_MARGIN_TEST)
    df = pd.concat([df1, df2], axis=0).drop_duplicates().reset_index(drop=True)
    x = preprocessor.preprocessing(np.expand_dims(df['備轉容量(萬瓩)'].to_numpy(), axis=1) * 10, 'x')
    test_index = df.index[df['日期'] == Constant.START_DATE].tolist()[0]
    data = []
    result = []
    hp = Hyperparameter
    for i in range(test_index, len(x)):
        data.append(x[i - hp.INPUT_SEQ_LEN + 1:i + 1])
    data = torch.from_numpy(np.array(data))
    output = pd.DataFrame({'date', 'operating_reserve(MW)'})
    output = []
    pred_date = datetime.strptime(Constant.START_DATE, '%Y/%m/%d') + timedelta(days=1)
    for each in data:
        each = each.unsqueeze(2).to(torch.float32).to(device)
        print('each', each)
        r = model(each, torch.zeros((Hyperparameter.OUTPUT_SEQ_LEN, 1, 1)), teacher_forcing_ratio=-1)
        print(r.shape)
        for each_r in preprocessor.inverse(r[:, 0, :].cpu().detach().numpy(), 'y'):
            output.append([
                pred_date.strftime('%Y%m%d'),
                each_r[0]
            ])
            pred_date = pred_date + timedelta(days=1)

    output = pd.DataFrame(output, columns=['date', 'operating_reserve(MW)'])
    output.to_csv(Constant.OUTPUT_FILE, index=False)


