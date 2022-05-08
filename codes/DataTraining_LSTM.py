import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from GPUManager import GPUManager


def set_device():
    if torch.cuda.is_available():
        gm = GPUManager()
        device = torch.device("cuda:{}".format(gm.auto_choice()))
    else:
        device = torch.device("cpu")
    return device


device = set_device()


class LstmModel(torch.nn.Module): # 继承自 torch.nn.Module
    def __init__(self, input_dims, input_seq_length, output_dims, output_seq_length, h_dims, lstm_num_layers) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.input_seq_length = input_seq_length
        self.output_dims = output_dims
        self.output_seq_length = output_seq_length
        self.h_dims = h_dims
        self.lstm_num_layers = lstm_num_layers

        self.hidden_cell = (
            torch.zeros(self.lstm_num_layers, self.h_dims).to(device),
            torch.zeros(self.lstm_num_layers, self.h_dims).to(device)
        )

        # 神经网络包括5个层
        self.lstm1 = torch.nn.LSTM(
            input_size = self.input_dims,
            hidden_size = self.h_dims,
            num_layers = self.lstm_num_layers
        )
        self.linear1 = torch.nn.Linear(
            in_features = self.input_seq_length,
            out_features = self.output_seq_length
        )
        self.linear2 = torch.nn.Linear(
            in_features = self.h_dims,
            out_features = self.h_dims
        )
        self.relu = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(
            in_features = self.h_dims,
            out_features = self.output_dims
        )

    def forward(self, x):
        y1, self.hidden_cell = self.lstm1(x, self.hidden_cell)
        # y2 = torch.t(self.linear1(torch.t(y1.view(self.input_dims, self.h_dims))))
        y2 = y1[-self.output_seq_length:, :]
        y3 = self.linear2(y2)
        y4 = self.relu(y3)
        y5 = self.linear3(y4)
        return y5

    def init_hidden_cell(self):
        self.hidden_cell = (
            torch.zeros(self.lstm_num_layers, self.h_dims).to(device),
            torch.zeros(self.lstm_num_layers, self.h_dims).to(device)
        )


# 生成训练集和测试集
def dataset_generator(input_seq_length, output_seq_length, pred_length, source) -> tuple([torch.tensor, torch.tensor, torch.tensor, torch.tensor]):
    source = np.array(source)
    source_len = len(source)
    for i in range(source_len - input_seq_length - output_seq_length - pred_length):
        x = torch.from_numpy(source[i:i+input_seq_length]).to(torch.float32)
        y = torch.from_numpy(source[i+input_seq_length:i+input_seq_length+output_seq_length]).to(torch.float32)
        x_label = torch.from_numpy(source[i+pred_length:i+input_seq_length+pred_length]).to(torch.float32)
        y_label = torch.from_numpy(source[i+input_seq_length+pred_length:i+input_seq_length+output_seq_length+pred_length]).to(torch.float32)
        yield x, y, x_label, y_label


# 计划实现一个移动阅读框训练，是否能减少过拟合
def train(input_dims, input_seq_length, output_dims, output_seq_length, pred_length, h_dims, lstm_num_layers, source, subepochs_init, learning_rate) -> None:  
    model = LstmModel(
        input_dims,
        input_seq_length,
        output_dims,
        output_seq_length,
        h_dims,
        lstm_num_layers
    )
    model.to(device)
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    total_len = len(source) - input_seq_length - output_seq_length - pred_length
    subepochs = subepochs_init
    finished = 0
    for cases_data in dataset_generator(input_seq_length, output_seq_length, pred_length, source):
        # 目前希望更靠后的数据训练更多的 epoches
        for epoch in range(1, subepochs+1):
            x, y, x_label, y_label = cases_data
            model.init_hidden_cell()
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.init_hidden_cell()
            y_pred = model(x_label.to(device))
            test_loss = criterion(y_pred, y_label.to(device))
            optimizer.zero_grad()
            
            # 输出提示
            if epoch == subepochs:
                print('                       subepoch: {}/{}    train_loss: {}    test_loss: {}'.format(epoch, subepochs, loss, test_loss), end='\r')
                print('                     ]\r ' + '='*int(finished*20/total_len) + '>\r[', end = '\r')

            train_losses.append(float(loss.to('cpu').data))
            test_losses.append(float(test_loss.to('cpu').data))

        subepochs = int((subepochs_init+1)*((finished/total_len)*0.5+1))
        finished += 1

    print()
    return test_losses[-1]


# 运行
def run():
    return


if __name__ == '__main__':
    set_device()
    df = pd.read_csv('./data/data/New_cases_normalized.csv')

    final_losseses = []
    for subepochs in [10, 100, 1000, 10000, 100000, 1000000]:
        final_losses = []
        for learning_rate in [0.1, 0.01, 0.001]:
            print('train {} {}'.format(subepochs, learning_rate))
            final_losses.append(train(
                input_dims=228,
                input_seq_length=56,
                output_dims=228,
                output_seq_length=28,
                pred_length=7,
                h_dims=912,
                lstm_num_layers=2,
                source=df,
                subepochs_init=100,
                learning_rate=learning_rate
            ))
        final_losseses.append(final_losses)
    print()
    for fl in final_losseses:
        print(fl)

