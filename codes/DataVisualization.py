import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def clustering():
    df = pd.read_csv('./data/data/New_cases_normalized.csv')
    X = np.array(df).T
    X_tsne = TSNE(n_components=2)
    X_tsne.fit_transform(X)

    for point in X_tsne.embedding_:
        plt.scatter(point[0], point[1], c='blue')
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
        # plt.xticks(np.arange(2, 20))
        plt.show()
        res_norm = (res - min(res)) / (max(res) - min(res))
        return res_norm


    elbow_res = elbow_plot(X_tsne.embedding_)
    res = []
    for i in range(2,20):
        i_norm = (i-2)/(19-2)
        res.append(elbow_res[i-2] + i_norm)

    n_clusters = list(range(2,20))[res.index(min(res))]
    Y_tsne = KMeans(n_clusters=n_clusters).fit_predict(X_tsne.embedding_)
    for i in range(len(X_tsne.embedding_)):
        point = X_tsne.embedding_[i]
        label = Y_tsne[i]
        plt.scatter(
            point[0], point[1],
            color=plt.cm.Set1(label)
        )
    plt.savefig('./data/figures/countries_clustering.jpg')
    plt.show()


def view_new_cases():
    df = pd.read_csv('./data/data/New_cases.csv')
    for region in df.columns:
        cases = df[region]
        plt.plot(list(range(len(cases))), cases)
        plt.savefig('./data/figures/new_cases/{}.jpg'.format(region))
        # plt.show()


def view_cumulative_cases():
    df = pd.read_csv('./data/data/Cumulative_cases.csv')
    for region in df.columns:
        cases = df[region]
        plt.plot(list(range(len(cases))), cases)
        plt.savefig('./data/figures/cumulative_cases/{}.jpg'.format(region))
        # plt.show()
