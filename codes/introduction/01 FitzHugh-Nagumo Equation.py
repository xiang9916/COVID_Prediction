import matplotlib.pyplot as plt

# init
V = -1
R = 1
a = 0.2
b = 0.2
c = 3

V_list = []
R_list = []
for i in range(10000):
    dV = c * (V - V**3 / 3 + R) / 100
    dR = - 1 / c * (V - a + b * R) / 100 

    V += dV
    R += dR
    V_list.append(V)
    R_list.append(R)

plt.plot(list(range(10000)), V_list)
plt.plot(list(range(10000)), R_list)
plt.show()