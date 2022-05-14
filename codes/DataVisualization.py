import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

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
            torch.zeros(self.lstm_num_layers, self.h_dims).to('cpu'),
            torch.zeros(self.lstm_num_layers, self.h_dims).to('cpu')
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
        y1, self.hidden_cell = self.lstm1(x.view(self.input_seq_length, self.input_dims), self.hidden_cell)
        # y2 = torch.t(self.linear1(torch.t(y1.view(self.input_dims, self.h_dims))))
        y2 = y1[-self.output_seq_length:, :]
        y3 = self.linear2(y2)
        y4 = self.relu(y3)
        y5 = self.linear3(y4)
        return y5

    def init_hidden_cell(self):
        self.hidden_cell = (
            torch.zeros(self.lstm_num_layers, self.h_dims).to('cpu'),
            torch.zeros(self.lstm_num_layers, self.h_dims).to('cpu')
        )


def clustering():
    df = pd.read_csv('./data/data/New_cases_normalized.csv')
    X = np.array(df).T
    X_tsne = TSNE(n_components=2)
    X_tsne.fit_transform(X)

    for i in range(len(X_tsne.embedding_)):
        point = X_tsne.embedding_[i]
        region = df.columns[i]
        if region == 'Argentina':
            ar = plt.scatter(point[0], point[1], color='skyblue', zorder = 2)
        elif region == 'China':
            cn = plt.scatter(point[0], point[1], color='red', zorder = 2)
        elif region == 'Italy':
            it = plt.scatter(point[0], point[1], color='green', zorder = 2)
        elif region == 'Japan':
            jp = plt.scatter(point[0], point[1], color='maroon', zorder = 2)
        elif region == 'South Africa':
            sa = plt.scatter(point[0], point[1], color='orange', zorder = 2)
        elif region == 'United States of America':
            us = plt.scatter(point[0], point[1], color='blue', zorder = 2)
        else:
            plt.scatter(point[0], point[1], color='grey', zorder = 1)
    plt.legend(handles=[ar, cn, it, jp, sa, us],labels=['Argentina',  'China', 'Italy', 'Japan', 'South Africa', 'United States of America'],loc='best')
    plt.title("TSNE Plot")
    plt.savefig('./data/figures/countries_TSNE.jpg')
    plt.show()


    def elbow_plot(x):
        res = np.zeros(18)  # Sum用于储存肘部法则判断指标
        for i in range(2,20):
            kmeans=KMeans(n_clusters=i).fit(x)
            m = kmeans.labels_
            c = kmeans.cluster_centers_
            for j in range(len(x)):
                c1 = c[m[j]]
                x1 = x[j]
                res[i-2] = res[i-2]+sum((x1-c1)**2)
        plt.plot(np.arange(2, 20), res)
        plt.xticks(np.arange(2, 20))
        plt.title("Elbow Plot")
        plt.xlabel('the n in K-means model')
        plt.ylabel("SSE of K-means model")
        plt.savefig('./data/figures/countries_elbow_plot.jpg')
        plt.show()
        res_norm = (res - min(res)) / (max(res) - min(res))
        return res_norm


    elbow_res = elbow_plot(X_tsne.embedding_)
    res = []
    for i in range(2,20):
        i_norm = (i-2)/(19-2)
        res.append((elbow_res[i-2])**2 + i_norm**2)

    n_clusters = list(range(2,20))[res.index(min(res))]
    Y_tsne = KMeans(n_clusters=n_clusters).fit_predict(X_tsne.embedding_)
    for i in range(len(X_tsne.embedding_)):
        point = X_tsne.embedding_[i]
        label = Y_tsne[i]
        region = df.columns[i]
        if region == 'Argentina':
            ar = plt.scatter(point[0], point[1], color=plt.cm.Set1(label), marker='*', zorder = 2)
        elif region == 'China':
            cn = plt.scatter(point[0], point[1], color=plt.cm.Set1(label), marker='*', zorder = 2)
        elif region == 'Italy':
            it = plt.scatter(point[0], point[1], color=plt.cm.Set1(label), marker='*', zorder = 2)
        elif region == 'Japan':
            jp = plt.scatter(point[0], point[1], color=plt.cm.Set1(label), marker='*', zorder = 2)
        elif region == 'South Africa':
            sa = plt.scatter(point[0], point[1], color=plt.cm.Set1(label), marker='*', zorder = 2)
        elif region == 'United States of America':
            us = plt.scatter(point[0], point[1], color=plt.cm.Set1(label), marker='*', zorder = 2)
        else:
            plt.scatter(point[0], point[1], color=plt.cm.Set1(label), marker='o', zorder = 1)
    plt.legend(handles=[ar, cn, it, jp, sa, us],labels=['Argentina',  'China', 'Italy', 'Japan', 'South Africa', 'United States of America'],loc='best')
    plt.title("Clustering Plot")
    plt.savefig('./data/figures/countries_clustering.jpg')
    plt.show()


def model_compare():
    pytorch_models = os.listdir('./data/model/')
    models = {}
    for i in pytorch_models:
        models[i] = torch.load('./data/model/{}'.format(i), map_location='cpu')
    
    for i in models:
        model = models[i]
        print(i)
        print(model_run(model))


def model_run(model):
    df = pd.read_csv('./data/data/New_cases_normalized.csv')
    if model.input_dims == 1:
        input_seq = torch.from_numpy(np.array(df['United States of America'][-model.input_seq_length-7:-7])).to(torch.float32)
        model.init_hidden_cell()
        output_seq = model(input_seq)

        output = output_seq.detach().numpy().T
        mean = pd.read_pickle('./data/data/df_new.mean().pkl')['United States of America']
        std = pd.read_pickle('./data/data/df_new.std().pkl')['United States of America']
        res = output * std + mean
        return res
    else:
        pass
    return


def view_new_cases():
    df = pd.read_csv('./data/data/New_cases.csv')
    for region in df.columns:
        cases = df[region]
        plt.plot(list(range(len(cases))), cases)
        plt.title("{} 的 新增病例".format(region))
        plt.xlabel('Day n since 2020-01-03')
        plt.ylabel("New cases")
        plt.savefig('./data/figures/new_cases/{}.jpg'.format(region))
        # plt.show()
        plt.cla()


def view_cumulative_cases():
    df = pd.read_csv('./data/data/Cumulative_cases.csv')
    for region in df.columns:
        cases = df[region]
        ax = plt.gca()
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        plt.plot(list(range(len(cases))), cases)
        plt.title("{} 的 累计病例".format(region))
        plt.xlabel('Day n since 2020-01-03')
        plt.ylabel("Cumulative cases")
        plt.savefig('./data/figures/cumulative_cases/{}.jpg'.format(region))
        # plt.show()
        plt.cla()
