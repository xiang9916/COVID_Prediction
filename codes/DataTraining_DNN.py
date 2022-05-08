import numpy as np
import pandas as pd
import torch

df = pd.read_csv('../data/data/New_cases.csv')

def main(country = 228, train_days = 790, label_days = 7,  pred_days = 7, h_dims = 1000, lr = 0.01):
    x = torch.from_numpy(np.array(df.iloc[-train_days -label_days -pred_days:-label_days -pred_days, -country:]))
    x = x.to(torch.float32)
    y = torch.from_numpy(np.array(df.iloc[-label_days -pred_days:-pred_days, -country:]))
    y = y.to(torch.float32)
    x_test = torch.from_numpy(np.array(df.iloc[-train_days -label_days:-label_days, -country:]))
    x_test = x_test.to(torch.float32)
    y_label = torch.from_numpy(np.array(df.iloc[-label_days:, -country:]))
    y_label = y_label.to(torch.float32)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            # 实例化一个线性对象，参数是输入特征得维度，输出特征的维度，是否要bias
            self.linear1 = torch.nn.Linear(in_features=country, out_features=h_dims)
            self.relu1   = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(in_features=h_dims, out_features=h_dims)
            self.relu2 = torch.nn.ReLU()
            self.linear3 = torch.nn.Linear(in_features=h_dims, out_features=country)

        def forward(self, x):
            y1 = self.linear1(x)
            y2 = self.relu1(y1)
            y3 = self.linear2(y2)
            y4 = self.relu2(y3)
            y5 = self.linear3(y4)
            return y5

    model = Model()
    critertion = torch.nn.MSELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # -----------训练模型------------
    for epoch in range(1000):
        y_pred = model(x)
        loss = critertion(y_pred, y)
        # print(epoch, loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_test = model(x_test)
    # print('y_test=', y_test.data)
    loss = critertion(y_test, y_label)
    print('loss = {}'.format(loss.data))

if __name__ == '__main__':
    for i in range(7, 701, 7):
        print(i)
        main(
            country=228,
            train_days=i,
            label_days=28, # 28
            pred_days=7,
            h_dims=1000,
            lr=0.01
        )    