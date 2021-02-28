import math
import numpy as np
# import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
import community
import networkx as nx
from sklearn.neural_network import MLPRegressor
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


def create_control_objectives(B):
    # define weighted multi-directed graph
    G = nx.Graph()

    # create SIS network
    edge_list = [(i, j, B[i][j])
                 for i in range(n) for j in range(n) if B[i][j] != 0]
    G.add_weighted_edges_from(edge_list)

    # create community
    partition = community.best_partition(G, weight='weight', resolution=0.85)
    # partition = community.best_partition(G, weight='weight', resolution=1)
    partition = dict(sorted(partition.items()))

    # choice target community
    community_list = [[4], [2, 3, 4], [0, 1, 2, 3, 4]]

    # count the numbers of control objective
    M = len(community_list)

    # define threshold of each node in the target
    d_table = np.array([0.1, 0.085, 0.07])

    # define target nodes according to community
    W = np.zeros((M, n))
    for m in range(M):
        for i in range(n):
            if partition[i] in community_list[m]:
                W[m][i] = 1

    # define threshold of each taget
    d = np.zeros(M)
    for m in range(M):
        for i in range(n):
            if W[m][i] == 1:
                d[m] += d_table[m]

    return W, M, d_table, d


def lyapunov_param_solver(B, D):
    # define SVM
    mlp = MLPRegressor(activation='relu', alpha=0.0001, max_iter=500)

    # define target vector
    target = np.ones(n) * 0.001

    # model fitting
    mlp.fit(B - D, target)

    # calculate Lyapunov param
    p = mlp.predict(B - D).dot(np.linalg.inv(B - D))

    # add bias and implement normalization
    p += np.abs(p.min()) + 1
    p /= np.linalg.norm(p)

    # caluculate rc
    rc = (B.T - D).dot(p)

    return p, rc


def event_trigger_func(x, xk, sigma, eta):
    # define triggering rule
    if math.fabs(x - xk) > sigma * x + eta:
        return 1
    else:
        return 0


def plot_data(K, L, sigma, eta, M, d_table, community_part):
    # define time and gap
    Time = 50000
    h = 0.000035

    # define propotion of infected pepole
    x = np.zeros([Time, n])
    x0 = np.random.rand(n)
    x0[0] /= 3
    x[0] = x0
    xk = x0

    # define event and objective list
    event = np.zeros([Time, n])
    d_table_list = np.array([d_table for i in range(Time)])
    triggerring_time_gap = np.zeros([Time, n])

    # collect transition data of propotion of infected pepole and triggerring event
    for k in range(Time - 1):
        for i in range(n):
            if event_trigger_func(x[k][i], xk[i], sigma[i], eta[i]) == 1:
                xk[i] = x[k][i]
                event[k][i] = 1
        x[k + 1] = x[k] + h * (-(D + K.dot(np.diag(xk))).dot(x[k]) + (
            In - np.diag(x[k])).dot(B.T - L.dot(np.diag(xk).T)).dot(x[k]))

    for i in range(n):
        triggering_tmp = 0
        for k in range(Time - 1):
            if event[k][i] == 1:
                triggerring_time_gap[k][i] = k - triggering_tmp
                triggering_tmp = k
            else:
                triggerring_time_gap[k][i] = None

    # plot data
    color_map = ['darkred', 'red', 'darkorange', 'orange',
                 'gold', 'darkgreen', 'green',
                 'darkcyan', 'cyan', 'deepskyblue', 'navy', 'darkviolet',
                 'magenta', 'olive', 'lightskyblue', 'lime',
                 'crimson', 'peachpuff', 'mediumspringgreen', 'cornflowerblue']

    # cm = plt.cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(16, 9.7))
    color_num = 0
    for i in range(n):
        if W[community_part][i] == 1:
            ax.scatter(np.arange(Time),
                       triggerring_time_gap.T[i], color=color_map[color_num], s=200, alpha=1)
            color_num += 1
    plt.xlabel(r'$t$', fontsize=60)
    plt.ylabel(
        r'$t_{\ell + 1}^i - t_\ell^i,$ for $i\in\mathcal{V}_1$', fontsize=60)
    ax.xaxis.offsetText.set_fontsize(60)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=60)
    plt.yticks([0, 10000, 20000, 30000, 40000])
    ax.yaxis.offsetText.set_fontsize(60)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.setp(ax.get_yticklabels(), fontsize=60)
    plt.xlim(0, 49650)
    plt.ylim(0, 40000)
    plt.tight_layout()
    plt.grid()
    # plt.savefig("./images/transition_of_x_noinput.png", dpi=300)
    plt.savefig("./images/triggering_time_gap.pdf", dpi=300)


if __name__ == '__main__':
    # load design parameter
    # D_base_max = 1.8
    D = np.load('./data/matrix/D.npy')
    B = np.load('./data/matrix/B.npy')
    K = np.load('./data/matrix/K.npy')
    L = np.load('./data/matrix/L.npy')
    sigma = np.load('./data/matrix/sigma.npy')
    eta = np.load('./data/matrix/eta.npy')
    # define control objectives according to network
    W, M, d_table, d = create_control_objectives(B)

    # plot data
    plot_data(K, L, sigma, eta, M, d_table, community_part=0)


'''
    # # subplot 1 is the transition data of x
    fig, ax = plt.subplots(figsize=(16, 9.7))
    for i in range(n):
        ax.plot(x.T[i], lw=1)

    # colors = ['navy', 'deepskyblue', 'magenta', 'darkgreen']
    colors = ['navy', 'magenta', 'darkgreen']

    for m in range(M):
        x_com_ave = 0
        community_member_num = 0
        for i in range(n):
            if W[m][i] == 1:
                x_com_ave += x.T[i]
                community_member_num += 1
        ax.plot(x_com_ave / community_member_num, linestyle="dashdot",
                lw=5, label='Average in Community {}'.format(m + 1), color=colors[m])

    ax.plot(d_table_list.T[0], lw=4, linestyle="dotted",
            label=r'$\bar{x}_1 = 0.10$,   Threshold for Community 1', color=colors[0])
    ax.plot(d_table_list.T[1], lw=4, linestyle="dotted",
            label=r'$\bar{x}_2 = 0.085$, Threshold for Community 2', color=colors[1])
    ax.plot(d_table_list.T[2], lw=4, linestyle="dotted",
            label=r'$\bar{x}_3 = 0.070$, Threshold for Community 3', color=colors[2])

    # # # plot setting
    plt.xlabel(r'Time $[t]$', fontsize=45)
    plt.ylabel(r'$x_i(t)$', fontsize=45)
    # plt.title(r'Transition of $x$', fontsize=25)
    ax.xaxis.offsetText.set_fontsize(45)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=40)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=45)
    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
               borderaxespad=0, fontsize=35)
    plt.tight_layout()
    plt.grid()
    # plt.savefig("./images/transition_of_x_noinput.png", dpi=300)
    plt.savefig("./images/test_zeno.pdf", dpi=300)
    '''
