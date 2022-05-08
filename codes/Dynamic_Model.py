import numpy as np
from typing import Tuple


class SeirsModel:
    def __init__(self, s, e, i, c, r, beta, delta, gamma1, gamma2, xi) -> None:
        self.Ss = [s]
        self.Es = [e]
        self.Is = [i]
        self.Cs = [c]
        self.Rs = [r]
        self.N = s + e + i + c

        self.beta = beta
        self.delta = delta
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.xi = xi
        return

    def forward(self, steps=1000) -> None:
        S = self.Ss[-1]
        E = self.Es[-1]
        I = self.Is[-1]
        C = self.Cs[-1]
        R = self.Rs[-1]

        for _ in range(steps):
            dS = ((self.xi * C) - (self.beta * S * I) / self.N) / steps
            dE = (((self.beta * S * I) / self.N) - (self.delta * E)) / steps
            dI = ((self.delta * E) - (self.gamma1 * I)) / steps
            dC = ((self.gamma1 * I) - (self.xi * C)) / steps
            dR = (self.gamma2 * I) / steps

            S += dS
            E += dE
            I += dI
            C += dC
            R += dR

        self.Ss.append(S)
        self.Es.append(E)
        self.Is.append(I)
        self.Cs.append(C)
        self.Rs.append(R)
        return

    def output(self, epochs, len, steps=10000) -> [[float], [float], [float], [float], [float]]:
        for i in range(epochs):  # 最后的序列长度应为 epochs + 1
            self.forward(steps=steps)

        return [self.Ss[-len:], self.Es[-len:], self.Is[-len:], self.Cs[-len:], self.Rs[-len:]]


def criterion(pred, label) -> Tuple[float, float]:
    mse = (np.square(pred - label)).mean()
    mae = (np.abs(pred - label)).mean()
    return mse, mae/np.array(pred.mean(),label.mean()).mean()


if __name__ == '__main__':
    df = '../data/data/New_cases.csv'

    model = SeirsModel(
        s=329500000, # 总人口减后 4 者
        e=3165523, # 近 7 日新增的 3/7
        i=16161395, # 近 20 日新增
        c=64457026, # 前 20 日-前 100 日新增
        r=1003467, # 累计死亡
        beta=0.1,
        delta=0.3,
        gamma1=0.0475,
        gamma2=0.0025,
        xi=0.01
    )
    res = model.output(
        epochs=6,
        len=7,
        steps=1000
    )

    for r in res:
        print(r)

