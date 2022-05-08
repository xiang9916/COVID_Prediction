import json5
import numpy as np
import pandas as pd


class SeirsModel:
    def __init__(self, s, e, i, c, r, beta, delta, gamma1, gamma2, xi) -> None:
        self.Ss = [s]
        self.Es = [e]
        self.Is = [i]
        self.Cs = [c]
        self.Rs = [r]
        self.N = s + e + i + c + r

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

    def output(self, epochs, len, steps=10000):
        for i in range(epochs):  # 最后的序列长度应为 epochs + 1
            self.forward(steps=steps)
        return [self.Ss[-len:], self.Es[-len:], self.Is[-len:], self.Cs[-len:], self.Rs[-len:]]


def criterion(pred, label):
    mse = (np.square(pred - label)).mean()
    mae = (np.abs(pred - label)).mean()
    return mse, mae/np.array(pred.mean(),label.mean()).mean()


def main(start, length):
    best_loss = 999999999999
    df = pd.read_csv('./data/data/New_cases.csv')

    us = df['United States of America']
    us_len = len(us)
    training_data = np.array(us[:start])
    label = np.array(us[start:start+length])

    beta = 0.11
    delta = 1
    course = 19
    death_rate = 0.048
    gamma1 = (1/course) * (1-death_rate)
    gamma2 = (1/course) * death_rate
    xi = 0.01

    model = SeirsModel(
        s=329500000 - np.sum(training_data[-int(1/xi):]), # 总人口减后 4 者
        e=(1/delta) * np.mean(training_data[-round(1/delta):]), # 近 7 日感染的 3/7
        i=np.sum(training_data[-int(1/gamma1+gamma2):]) - (1/delta) * np.mean(training_data[-int(1/delta):]), # 近 20 日新增
        c=np.sum(training_data[-int(1/xi):-int(1/gamma1+gamma2)]) * (gamma1/(gamma1+gamma2)), # 前 20 日-前 100 日新增
        r=np.sum(training_data[-int(1/xi):-int(1/gamma1+gamma2)]) * (gamma2/(gamma1+gamma2)), # 累计死亡
        beta=beta,
        delta=delta,
        gamma1=gamma1,
        gamma2=gamma2,
        xi=xi
    )
    res = model.output(
        epochs=length,
        len=length+1,
        steps=1000
    )
    
    pred =np.array([])
    for l in range(-length,0):
        S = res[0][l]
        E = res[1][l]
        I = res[2][l]
        C = res[3][l]
        R = res[4][l]
        N = S+E+I+C+R
        dI_sum = 0
        for _ in range(steps:=1000):
            dS = ((xi * C) - (beta * S * I) / N) / steps
            dE = (((beta * S * I) / N) - (delta * E)) / steps
            dI = ((delta * E) - (gamma1 * I) - (gamma2 * I)) / steps
            dC = ((gamma1 * I) - (xi * C)) / steps
            dR = (gamma2 * I) / steps
            S += dS
            E += dE
            I += dI
            dI_sum += dI
            C += dC
            R += dR
        pred = np.append(pred, [dI_sum])
        # print(label)
    # print(criterion(pred, label))
    return pred


if __name__ == '__main__':
    main(-28,28)