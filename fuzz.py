import numpy as np
import matplotlib.pylab as plt


def fuszzify(x, var: str):
    y = None
    if var == "mv":
        y = np.array([mf(x, np.array([0, 0, 75, 100]) * 1000), mf(x, np.array([50, 100, 200, 250]) * 1000),
                      mf(x, np.array([200, 300, 650, 850]) * 1000), mf(x, np.array([650, 850, 1000, 1000]) * 1000)])
    elif var == "l":
        y = np.array([mf(x, np.array([0, 0, 1.5, 4])), mf(x, np.array([2.5, 5, 6, 8.5])),
                      mf(x, np.array([6, 8.5, 10, 10]))])
    elif var == "hv":
        y = np.array([mf(x, np.array([0, 0, 3])), mf(x, np.array([0, 3, 6])), mf(x, np.array([2, 5, 8])),
                      mf(x, np.array([4, 7, 10])), mf(x, np.array([7, 10, 10]))])
    return y


def mf(x, par: np.array):
    y = None
    par = np.sort(par)
    a = par[0]
    b = par[1]
    c = par[2]

    if (len(par) == 3):
        y = np.maximum(np.fmin((x - a) / (b - a), (c - x) / (c - b)), np.zeros(x.size))

    else:
        d = par[3]

        y = np.amax(np.amin(np.minimum((x - a) / (b - a), ((d - x) / (d - c))), initial=1), initial=0)
    return y


inp = np.array([700000, 6.5])
mv = fuszzify(inp[0], "mv")
l = fuszzify(inp[1], "l")
x = np.arange(0, 10, 0.01)
hv = fuszzify(x, "hv")
x = np.array(x)

plt.plot(x, hv[0, :])
plt.plot(x, hv[1, :])
plt.plot(x, hv[2, :])
plt.plot(x, hv[3, :])
plt.plot(x, hv[4, :])
plt.show()

rb = np.array([[1, 1, 2, 1], [1, 1, 2, 1], [1, 1, 1, 1], [1, 2, 2, 1], [1, 3, 3, 1], [1, 4, 4, 1], [2, 1, 2, 1],
               [2, 2, 3, 1], [2, 3, 4, 1], [2, 4, 5, 1], [3, 1, 3, 1], [3, 2, 4, 1], [3, 3, 5, 1], [3, 4, 5, 1]])


