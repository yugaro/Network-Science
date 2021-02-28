import math
import random
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter
from palettable.cubehelix import Cubehelix
# from palettable import cubehelix
# from matplotlib.colors import ListedColormap
np.random.seed(0)
random.seed(2)
rc('text', usetex=True)
rc('font', **{'family': "sans-serif"})
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

# number of nodes (max: 50)
n = 50

palette = Cubehelix.make(n=(n + 5), start=0.5, rotation=-1.5,
                         max_sat=3,).colors
for i in range(n + 5):
    for j in range(3):
        palette[i][j] /= 255
palette = palette[:n]
random.shuffle(palette)
# palette.reverse()

# preparation
INF = 1e9
epsilon = 1e-15
In = np.identity(n)
On = np.zeros((n, n))

class FixedOrderFormatter(ScalarFormatter):
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=True):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset,
                                 useMathText=useMathText)

    def _set_orderOfMagnitude(self, range):
        self.orderOfMagnitude = self._order_of_mag


def event_trigger_func(x, xk, sigma, eta):
    # define triggering rule
    if math.fabs(x - xk) > sigma * x + eta:
        return 1
    else:
        return 0


def plot_data(K, L, sigma, eta, d_table, W, choice):
    # define time and gap
    Time = 800000
    h = 0.0001

    # define propotion of infected pepole
    x = np.zeros([Time, n])
    x0 = np.random.rand(n)
    x0[0] /= 3
    x[0] = x0
    xk = x0

    # define event and objective list
    event = np.zeros([Time, n])
    d_table_list = np.array([d_table for i in range(Time)])
    u_transition = np.zeros([Time - 1, n])
    v_transition = np.zeros([Time - 1, n])

    # for i in range(n):
    #    if W[1][i] == 1 and choice == 3:
    #        D[i][i] *= 1.3

    Bn = B / (10 * D_base_max)
    Dn = D / (10 * D_base_max)
    Kn = K / (10 * D_base_max)
    Ln = L / (10 * D_base_max)

    # collect transition data of propotion of infected pepole and triggerring event
    for k in range(Time - 1):

        # # choice 1 has no control input
        if choice == 1:
            x[k + 1] = x[k] + h * \
                (-Dn.dot(x[k]) + (In - np.diag(x[k])).dot(Bn.T).dot(x[k]))

        # # In the case of using feedback controller
        else:
            for i in range(n):
                # # choice 2 is the case of continuous controller
                if choice == 2 and event_trigger_func(x[k][i], xk[i], 0, 0) == 1:
                    xk[i] = x[k][i]
                    event[k + 1][i] = 1

                # # choice 3 is the case of event-triggered controller
                elif choice == 3 and event_trigger_func(x[k][i], xk[i], sigma[i], eta[i]) == 1:
                    xk[i] = x[k][i]
                    event[k + 1][i] = 1
            x[k + 1] = x[k] + h * (-(Dn * 1.1 + Kn.dot(np.diag(xk))).dot(x[k]) + (
                In - np.diag(x[k])).dot(Bn.T - Ln.dot(np.diag(xk)).T).dot(x[k]))

    # plot data
    # # subplot 1 is the transition data of x
    # fig, ax = plt.subplots(figsize=(16, 9.7))
    fig = plt.figure(figsize=(16, 9.7))
    ax1 = fig.add_axes((0, 0, 1, 1))

    # select color
    cm = plt.cm.get_cmap('cubehelix_r', n)

    # my_cmap = plt.cm.RdBu(np.arange(plt.cm.RdBu.N))
    # my_cmap[:, 0:3] *= 0.5
    # my_cmap = ListedColormap(my_cmap)
    for i in (range(n)):
        if W[0][i] == 1:
            ax1.plot(x.T[i], lw=2, color="darkred")
        elif W[1][i] == 1:
            ax1.plot(x.T[i], lw=2, color="midnightblue")
        elif W[2][i] == 1:
            ax1.plot(x.T[i], lw=2, color='g')
        # ax.plot(x.T[i], lw=3, color=cm(i), linestyle="dashed",)

    # # # plot setting
    ax1.set_xlabel(r'$t$', fontsize=60)
    ax1.set_ylabel(
        r'$x_i(t)$', fontsize=60)
    ax1.set_xticks([0, 200000, 400000, 600000, 800000])
    ax1.xaxis.set_major_formatter(FixedOrderFormatter(4, useMathText=True))
    ax1.xaxis.offsetText.set_fontsize(0)
    ax1.ticklabel_format(style="sci", axis="x", scilimits=(4, 4))
    ax1.tick_params(axis='x', labelsize=60)
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_yticklabels([r'$0$', r'$0.5$', r'$1$'])
    ax1.tick_params(axis='y', labelsize=60)
    ax1.grid(which='major', alpha=0.8, linestyle='dashed')

    if choice == 3:
        ax2 = fig.add_axes((0.3, 0.3, 0.65, 0.65))
        for i in (range(n)):
            ax2.plot(np.arange(400000, Time),
                     x.T[i][400000:], lw=2, color=palette[i])

        ax2.set_xticks([400000, 600000, 800000])
        ax2.xaxis.set_major_formatter(FixedOrderFormatter(4, useMathText=True))
        ax2.xaxis.offsetText.set_fontsize(0)
        ax2.ticklabel_format(style="sci", axis="x", scilimits=(4, 4))
        ax2.tick_params(axis='x', labelsize=60)
        # ax2.set_yscale('log')
        ax2.set_yticks([0, 0.05, 0.1])
        ax2.set_yticklabels([r'$0$', r'$0.05$', r'$0.1$'])
        ax2.tick_params(axis='y', labelsize=60)
        # ax2.grid(which="both", linestyle='dotted', color='gray')
        # ax2.grid(which='minor', alpha=0.4, linestyle='dotted')
        ax2.grid(which='major', alpha=0.6, linestyle='dotted')

    if choice == 1:
        fig.savefig("./images/x_all_zeroinput_normal.pdf",
                    bbox_inches="tight", dpi=300)
    elif choice == 2:
        fig.savefig("./images/x_all_continuous_normal.pdf",
                    bbox_inches="tight", dpi=300)
    elif choice == 3:
        fig.savefig("./images/x_all_event_normal80_comb4.pdf",
                    bbox_inches="tight", dpi=300)


if __name__ == '__main__':
    D_base_max = 1.8
    D = np.load('./data/matrix/D.npy')
    B = np.load('./data/matrix/B.npy')
    K = np.load('./data/matrix/K.npy')
    L = np.load('./data/matrix/L.npy')
    sigma = np.load('./data/matrix/sigma.npy')
    eta = np.load('./data/matrix/eta.npy')
    W = np.load('data/matrix/W.npy')
    d_table = np.load('data/matrix/d_table.npy')
    W = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0,
                   1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, ],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                   0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]])

    # plot data
    plot_data(K, L, sigma, eta, d_table, W, choice=3)
