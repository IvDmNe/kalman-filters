import numpy as np

from models import LDSSM
from matplotlib import pyplot as plt


if __name__ == '__main__':
    freq = 10
    Magnitude = 0.5
    dt = 0.01
    A = np.array([[0, 1], 
                [-freq**2, 0]])


    model = LDSSM(A, np.zeros((2, 1)), np.array([[1, 0]]), np.array([[0.0], [Magnitude * freq]]), dt)

    a = np.arange(0, 200)
    ys = []

    for i in a:
        y = model.step(0, dt)
        ys.append(y)

    ys = np.array(ys).squeeze()
    plt.plot(a, ys)
    plt.grid()
    plt.show()