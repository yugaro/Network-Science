import math
import random
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
from palettable.cubehelix import Cubehelix
np.random.seed(0)
random.seed(0)
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
            v_transition_event[k][i] = L[i] * xk
        x_event[k + 1] = x_event[k] + h * (-(D + K.dot(np.diag(xk))).dot(x_event[k]) + (
            In - np.diag(x_event[k])).dot(B.T - L.dot(np.diag(xk)).T).dot(x_event[k]))
        # # # record event-triggered control input
        u_transition_event[k] = K.dot(xk)

    for i in reversed(range(n)):
        triggering_tmp = 0
        for k in range(Time):
            if event[k][i] == 1:
                inter_time_events[k][i] = k - triggering_tmp
                triggering_tmp = k
            else:
                inter_time_events[k][i] = None

    fig, ax = plt.subplots(figsize=(16, 9.7))
    color_num = 0
    for i in range(n):
        if W[group_part - 1][i] == 1:
            ax.plot(u_transition_event.T[i],
                    lw=4, color=palette[i], alpha=1)
            color_num += 1
    # # # plot setting
    plt.xlabel(r'$t$', fontsize=60)
    plt.ylabel(
        r'$u_i(t)$', fontsize=60)
    plt.xticks([0, 100000, 200000, 300000, 400000])
    ax.xaxis.offsetText.set_fontsize(0)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(4, 4))
    plt.setp(ax.get_xticklabels(), fontsize=60)
    plt.yticks([0, 0.05, 0.10, 0.15])
    plt.ylim(0, 0.155)
    ax.yaxis.offsetText.set_fontsize(60)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.setp(ax.get_yticklabels(), fontsize=60)
    plt.tight_layout()
    plt.grid(which='major', alpha=0.8, linestyle='dashed')
    plt.savefig(
        "./images/control_input_recovery_rate_group{}.pdf".format(group_part), dpi=300)

    # subplot of the transition data of V (event)
    fig, ax = plt.subplots(figsize=(16, 9.7))
    color_num = 0
    v_argmax = np.argmax(v_transition_event[0], axis=0)
    for i in reversed(range(n)):
        if W[group_part - 1][i] == 1:
            ax.plot(v_transition_event.T[i][v_argmax[i]],
                    lw=4, color=palette[i], alpha=1)

    # # # plot setting
    plt.xlabel(r'$t$', fontsize=60)
    plt.ylabel(
        r'$v_{ij}(t)$', fontsize=60)
    plt.xticks([0, 100000, 200000, 300000, 400000])
    ax.xaxis.offsetText.set_fontsize(0)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(4, 4))
    plt.setp(ax.get_xticklabels(), fontsize=60)
    plt.yticks([0, 0.008, 0.016, 0.024])
    # ax.set_yticks([0, 0.008, 0.016, 0.024])
    # ax.set_yticklabels([r'0', r'0.008', r'0.016', r'0.024'])
    ax.yaxis.offsetText.set_fontsize(60)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.setp(ax.get_yticklabels(), fontsize=60)
    # ax.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0,
    #           fontsize=20, title=r'airport code $(\mathcal{V}_1)$', ncol=2).get_title().set_fontsize(30)
    plt.tight_layout()
    plt.grid(which='major', alpha=0.8, linestyle='dashed')
    plt.savefig(
        "./images/control_input_infection_rate_group{}.pdf".format(group_part), dpi=300)
    '''
    # subplot of inter-time event
    fig, ax = plt.subplots(figsize=(16, 9.7))
    color_num = 0
    for i in (range(n)):
        # if W[group_part - 1][i] == 1:
        ax.scatter(np.arange(Time),
                   inter_time_events.T[i], color=cm(i), s=100, alpha=1)
        # color_num += 1
    plt.xlabel(r'$t$', fontsize=60)
    plt.ylabel(
        r'$t_{\ell + 1}^i - t_\ell^i$', fontsize=60)
    plt.xticks([0, 100000, 200000, 300000, 400000])
    ax.xaxis.offsetText.set_fontsize(0)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(4, 4))
    plt.setp(ax.get_xticklabels(), fontsize=60)

    ax.set_yscale('log')
    ax.tick_params(axis='y', labelsize=60)
    # plt.yticks([0, 500000])
    # ax.yaxis.offsetText.set_fontsize(0)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(4, 4))
    # plt.setp(ax.get_yticklabels(), fontsize=60)
    # ax.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0,
    #          fontsize=20, title=r'airport code $(\mathcal{V}_1)$', ncol=2).get_title().set_fontsize(30)
    plt.tight_layout()
    plt.grid()
    plt.savefig("./images/inter-event_time_all.pdf", dpi=300)
    '''

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

    '''
    color_map = ['darkred', 'red', 'darkorange', 'orange',
                 'gold', 'darkgreen', 'green',
                 'darkcyan', 'cyan', 'deepskyblue', 'navy', 'darkviolet',
                 'magenta', 'olive', 'lightskyblue', 'lime',
                 'crimson', 'peachpuff', 'mediumspringgreen', 'cornflowerblue',
                 ]
    '''
