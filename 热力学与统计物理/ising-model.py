"""
2D ising model with metropolis algorithm.

Author: Chen Huang
Data: Nov 12, 2020
"""


import numpy as np
import matplotlib.pyplot as plt


def solve_ising(T, N, J=1, B=0):
    """
    solve 2D ising model with metropolis algorithm
    :param T: temperature
    :param N: lattice size NxN
    :param J: interaction strength
    :param B: external magnetic field
    :return: magnetization m, spin configuration S
    """
    # initialization
    S = np.ones((N, N), dtype=int)
    # S=2*np.random.randint(0,2,size=[N,N],dtype=int)-1

    # MC loop
    for i in range(1000 * N ** 2):
        pos_x = np.random.randint(0, N)
        pos_y = np.random.randint(0, N)
        dE = get_de(N, S, pos_x, pos_y, J, B)
        if dE < 0:
            S[pos_x, pos_y] = -S[pos_x, pos_y]
        elif np.random.rand() < np.exp(-dE / T):
            S[pos_x, pos_y] = -S[pos_x, pos_y]
    m = np.sum(S) / (N ** 2)
    return m, S


def get_de(N, S, pos_x, pos_y, J, B):
    dE = 2 * B * S[pos_x, pos_y]
    for i, j in get_neighbor(N, pos_x, pos_y):
        dE = dE + 2 * J * S[pos_x, pos_y] * S[i, j]
    return dE


def get_neighbor(N, pos_x, pos_y):
    return [((pos_x + 1) % N, pos_y), (pos_x - 1, pos_y), (pos_x, (pos_y + 1) % N), (pos_x, pos_y - 1)]


T = np.linspace(0.001, 5, 12)
m_avg = []
N = 50

fig, axs = plt.subplots(3, 4, figsize=(15, 12), subplot_kw={'xticks': [], 'yticks': []})
for i in range(0, 3):
    for j in range(0, 4):
        m, S = solve_ising(T[i * 4 + j], N)
        axs[i, j].imshow(S, cmap='gray')
        axs[i, j].set_title('T=' + str(round(T[i * 4 + j], 2)) + '     m=' + str(round(m, 4)))
        print(f'T={round(T[i * 4 + j], 2)}, m={str(round(m, 5))}')
        m_avg.append(abs(m))

# plt.savefig('pic.png')
plt.show()

# plt.plot(T, m_avg, 'ro')
# plt.title('B=0  N=50*50')
# plt.xlabel('T')
# plt.ylabel('|m|')
# plt.savefig('pic2.png')
# plt.show()
