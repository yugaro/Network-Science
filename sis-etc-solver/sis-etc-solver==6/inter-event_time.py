import math
import random
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
from palettable.cubehelix import Cubehelix
np.random.seed(0)
random.seed(8)
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

palette = Cubehelix.make(n=(n + 5), start=0.5, rotation=-1.5,
                         max_sat=2.5,).colors
for i in range(n + 5):
    for j in range(3):
        palette[i][j] /= 255
palette = palette[:n]
random.shuffle(palette)
# palette.reverse()


def event_trigger_func(x, xk, sigma, eta):
    # define triggering rule
    if math.fabs(x - xk) > sigma * x + eta:
        return 1
    else:
        return 0


def plot_data_community(K, L, sigma, eta, W, d_table, group_part):
    # define time and gap
    Time = 400000
    h = 0.0001

    x_event = np.zeros([Time, n])

    # define initial state
    x0 = np.random.rand(n)
    x0[0] /= 3
    x_event[0] = x0
    xk = x0

    # define event and objective list
    event = np.zeros([Time, n])
    d_table_list = np.array([d_table for i in range(Time)])

    # create control input series (feedback, event-trigger)
    u_transition_event = np.zeros([Time - 1, n])
    v_transition_event = np.zeros([Time - 1, n, n])

    # define inter-time events
    inter_time_events = np.zeros([Time, n])

    # collect transition data of propotion of infected pepole and triggerring event
    for k in range(Time - 1):
        for i in range(n):
            # # event-triggered control
            if event_trigger_func(x_event[k][i], xk[i], sigma[i], eta[i]) == 1:
                xk[i] = x_event[k][i]
                event[k + 1][i] = 1
        x_event[k + 1] = x_event[k] + h * (-(D + K.dot(np.diag(xk))).dot(x_event[k]) + (
            In - np.diag(x_event[k])).dot(B.T - L.dot(np.diag(xk)).T).dot(x_event[k]))

    for i in range(n):
        triggering_tmp = 0
        for k in range(Time):
            if event[k][i] == 1:
                inter_time_events[k][i] = k * h - triggering_tmp
                triggering_tmp = k * h
            else:
                inter_time_events[k][i] = None

    # subplot of inter-time event
    fig, ax = plt.subplots(figsize=(16, 9.7))
    # cm = plt.cm.get_cmap('cubehelix_r', n)
    for i in range(n):
        if W[group_part - 1][i] == 1:
            ax.scatter(np.arange(Time),
                       inter_time_events.T[i], color=palette[i], s=250, alpha=1)
    plt.xlabel(r'$t$', fontsize=60)
    plt.ylabel(
        r'$t_{\ell + 1}^i - t_\ell^i$', fontsize=60)
    plt.xticks([0, 100000, 200000, 300000, 400000])
    ax.xaxis.offsetText.set_fontsize(0)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(4, 4))
    plt.setp(ax.get_xticklabels(), fontsize=60)

    ax.set_yscale('log')
    ax.tick_params(axis='y', labelsize=60)
    ax.set_ylim(0.5, 45)
    plt.tight_layout()
    ax.grid(which='major', alpha=0.8, linestyle='dashed')
    plt.savefig("./images/inter-event_time_group1.pdf", dpi=300)


if __name__ == "__main__":
    # load parameters of the event-triggered controller
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
    plot_data_community(K, L, sigma, eta, W, d_table, group_part=1)
