import math
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter
np.random.seed(0)
rc('text', usetex=True)
rc('font', **{'family': "sans-serif"})
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

# number of nodes (max: 50)
n = 50

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


def plot_data_community(K, L, sigma, eta, d_table, group_part):
    # define time and gap
    Time = 800000
    h = 0.00015

    # define propotion of infected pepole
    x_noinput = np.zeros([Time, n])
    x_continuous = np.zeros([Time, n])
    x_control = np.zeros([Time, n])
    x0 = np.random.rand(n)
    x0[0] /= 3
    x_noinput[0] = x0
    x_continuous[0] = x0
    x_control[0] = x0
    xk = x0

    # define event and objective list
    event = np.zeros([Time, n])
    d_table_list = np.array([d_table for i in range(Time)])
    # collect transition data of propotion of infected pepole and triggerring event
    for k in range(Time - 1):
        x_noinput[k + 1] = x_noinput[k] + h * \
            (-D.dot(x_noinput[k]) +
             (In - np.diag(x_noinput[k])).dot(B.T).dot(x_noinput[k]))
        x_continuous[k + 1] = x_continuous[k] + h * (-(D * 1.15 + K.dot(np.diag(x_continuous[k]))).dot(x_continuous[k]) + (
            In - np.diag(x_continuous[k])).dot(B.T - L.dot(np.diag(x_continuous[k])).T).dot(x_continuous[k]))
        for i in range(n):
            # # choice 3 is the case of event-triggered controller
            if event_trigger_func(x_control[k][i], xk[i], sigma[i], eta[i]) == 1:
                xk[i] = x_control[k][i]
                event[k + 1][i] = 1
        x_control[k + 1] = x_control[k] + h * (-(D * 1.165 + K.dot(np.diag(xk))).dot(x_control[k]) + (
            In - np.diag(x_control[k])).dot(B.T - L.dot(np.diag(xk)).T).dot(x_control[k]))
    # 1: 1
    # 2: 1.15 - 1.165, 0.00015
    # 3: 0.93
    # plot data
    # # subplot 1 is the transition data of x
    fig = plt.figure(figsize=(16, 9.7))
    ax = fig.add_axes((0, 0, 1, 1))
    # initialize variable
    x_com_ave_noinput = 0
    x_com_ave_continuous = 0
    x_com_ave_control = 0
    community_member_num = 0

    # compute the average of trajectories in each group
    for i in range(n):
        if W[group_part - 1][i] == 1:
            x_com_ave_noinput += x_noinput.T[i]
            x_com_ave_continuous += x_continuous.T[i]
            x_com_ave_control += x_control.T[i]
            community_member_num += 1

    ax.plot(x_com_ave_noinput / community_member_num, linestyle="dotted",
            lw=7, color='lime', label=r'Zero Control Input $(\mathcal{V}_{%d})$' % (group_part), zorder=2)

    ax.plot(x_com_ave_continuous / community_member_num, linestyle="dashed",
            lw=7, color='dodgerblue', label=r'Continuous-Time Control $(\mathcal{V}_{%d})$' % (group_part), zorder=3)

    ax.plot(x_com_ave_control / community_member_num, linestyle="solid",
            lw=7, color='crimson', label=r'Event-Triggered Control $(\mathcal{V}_{%d})$' % (group_part), zorder=4)

    ax.plot(d_table_list.T[group_part - 1], lw=7, linestyle="dashdot",
            label=r'Threshold $(\bar{x}_%d = %.2f)$' % (group_part, d_table[group_part - 1]), color='darkorange', zorder=1)

    # # # plot setting
    ax.set_xlabel(r'$t$', fontsize=60)
    ax.set_ylabel(
        r'$\frac{1}{|\mathcal{V}_%d|}\sum_{i\in \mathcal{V}_%d} x_i(t)$' % (group_part, group_part), fontsize=60)
    # plt.title(r'Transition of $x$', fontsize=25)
    ax.set_xticks([0, 200000, 400000, 600000, 800000])
    ax.xaxis.set_major_formatter(FixedOrderFormatter(4, useMathText=True))
    ax.xaxis.offsetText.set_fontsize(0)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(4, 4))
    ax.tick_params(axis='x', labelsize=60)
    ax.set_yticks([0, 0.25, 0.5])
    ax.set_yticklabels([r'$0$', r'$0.25$', r'$0.5$'])
    ax.set_yticks([0.1], minor=True)
    ax.set_yticklabels([r'$0.1$'], minor=True)
    ax.set_ylim(0, 0.58)
    ax.tick_params(axis='y', labelsize=60, which='both')
    '''
    ax.set_yscale('log')
    ax.set_yticks([0.01, 0.1, 1])
    ax.set_yticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$1$'])
    ax.tick_params(axis='y', labelsize=60)
    '''
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
              borderaxespad=0, fontsize=48, ncol=1)

    # ax2.set_yscale('log')
    # ax2.set_yticks([0, 0.25, 0.5])
    # ax2.set_ylim(0, 0.65)
    # ax2.axis("off")
    # ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
    #           borderaxespad=0, fontsize=48, ncol=1)
    # ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0),
    #          borderaxespad=0, fontsize=43, ncol=1)
    ax.grid(which='major', alpha=0.8, linestyle='dashed')
    fig.savefig("./images/x_group{}_normal.pdf".format(group_part), bbox_inches="tight", dpi=300)


if __name__ == '__main__':

    D_base_max = 1.8
    D = np.load('./data/matrix/D.npy')
    D /= (D_base_max * 10)
    B = np.load('./data/matrix/B.npy')
    B /= (D_base_max * 10)
    K = np.load('./data/matrix/K.npy')
    K /= (D_base_max * 10)
    L = np.load('./data/matrix/L.npy')
    L /= (D_base_max * 10)
    sigma = np.load('./data/matrix/sigma.npy')
    eta = np.load('./data/matrix/eta.npy')
    W = np.load('data/matrix/W.npy')
    d_table = np.load('data/matrix/d_table.npy')

    # plot data
    plot_data_community(K, L, sigma, eta, d_table, group_part=2)
