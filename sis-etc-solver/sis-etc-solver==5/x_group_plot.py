import math
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
np.random.seed(0)
rc('text', usetex=True)
rc('font', **{'family': "sans-serif"})
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
plt.rcParams.update(params)
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

# number of nodes (max: 50)
n = 50

# preparation
INF = 1e9
epsilon = 1e-15
In = np.identity(n)
On = np.zeros((n, n))


def event_trigger_func(x, xk, sigma, eta):
    # define triggering rule
    if math.fabs(x - xk) > sigma * x + eta:
        return 1
    else:
        return 0


def plot_data_community(K, L, sigma, eta, d_table, group_part):
    # define time and gap
    Time = 100000
    h = 0.000030

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
        x_continuous[k + 1] = x_continuous[k] + h * (-(D + K.dot(np.diag(x_continuous[k]))).dot(x_continuous[k]) + (
            In - np.diag(x_continuous[k])).dot(B.T - L.dot(np.diag(x_continuous[k]).T)).dot(x_continuous[k]))
        for i in range(n):
            # # choice 3 is the case of event-triggered controller
            if event_trigger_func(x_control[k][i], xk[i], sigma[i], eta[i]) == 1:
                xk[i] = x_control[k][i]
                event[k + 1][i] = 1
        x_control[k + 1] = x_control[k] + h * (-(D + K.dot(np.diag(xk))).dot(x_control[k]) + (
            In - np.diag(x_control[k])).dot(B.T - L.dot(np.diag(xk).T)).dot(x_control[k]))

    # plot data
    # # subplot 1 is the transition data of x
    fig, ax = plt.subplots(figsize=(16, 9.7))

    # initialize variable
    x_com_ave_noinput = 0
    x_com_ave_continuous = 0
    x_com_ave_control = 0
    community_member_num = 0

    # compute
    for i in range(n):
        if W[group_part - 1][i] == 1:
            x_com_ave_noinput += x_noinput.T[i]
            x_com_ave_continuous += x_continuous.T[i]
            x_com_ave_control += x_control.T[i]
            community_member_num += 1
    ax.plot(x_com_ave_noinput / community_member_num, linestyle="dotted",
            lw=7, color='lime', label=r'Zero Control Inputs $(\mathcal{V}_%d)$' % (group_part), zorder=2)
    ax.plot(x_com_ave_continuous / community_member_num, linestyle="dashed",
            lw=7, color='cyan', label=r'Continuous-Time Control $(\mathcal{V}_%d)$' % (group_part), zorder=3)
    ax.plot(x_com_ave_control / community_member_num, linestyle="solid",
            lw=7, color='deeppink', label=r'Event-Triggered Control $(\mathcal{V}_%d)$' % (group_part), zorder=4)
    ax.plot(d_table_list.T[group_part - 1], lw=7, linestyle="dashdot",
            label=r'Threshold $(\bar{x}_%d = %.2f)$' % (group_part, d_table[group_part - 1]), color='navy', zorder=1)

    # # # plot setting
    plt.xlabel(r'$t$', fontsize=60)
    plt.ylabel(
        r'$\frac{1}{|\mathcal{V}_{%d}|}\sum_{i\in \mathcal{V}_{%d}}x_i(t)$' % (group_part, group_part), fontsize=60)
    # plt.title(r'Transition of $x$', fontsize=25)
    ax.xaxis.offsetText.set_fontsize(60)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=60)
    # plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=45)
    plt.yticks(fontsize=60)
    plt.ylim(0, 0.59)
    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
               borderaxespad=0, fontsize=43)
    plt.tight_layout()
    plt.grid()
    # plt.savefig("./images/transition_of_x_noinput.png", dpi=300)
    plt.savefig("./images/x_group{}.pdf".format(group_part), dpi=300)


if __name__ == '__main__':

    D = np.load('./data/matrix/D.npy')
    B = np.load('./data/matrix/B.npy')
    K = np.load('./data/matrix/K.npy')
    L = np.load('./data/matrix/L.npy')
    sigma = np.load('./data/matrix/sigma.npy')
    eta = np.load('./data/matrix/eta.npy')
    W = np.load('data/matrix/W.npy')
    d_table = np.load('data/matrix/d_table.npy')

    # plot data
    plot_data_community(K, L, sigma, eta, d_table, group_part=3)
