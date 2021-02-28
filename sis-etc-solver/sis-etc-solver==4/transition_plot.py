import math
import numpy as np
# import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
import networkx as nx
import community
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
    # community_list = [[0, 1, 2, 3], [2, 3, 4], [4], [0, 1, 2, 3, 4]]
    # community_list = [[4], [0, 1, 2, 3], [2, 3, 4], [0, 1, 2, 3, 4]]
    community_list = [[4], [2, 3, 4], [0, 1, 2, 3, 4]]
    # community_list = [[0], [1], [2]]

    # count the numbers of control objective
    M = len(community_list)

    # define threshold of each node in the target
    # d_table = np.array([0.1, 0.09, 0.08, 0.07])
    d_table = np.array([0.1, 0.085, 0.07])
    # d_table = np.array([0.1, 0.1, 0.1])

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


def event_trigger_func(x, xk, sigma, eta):
    # define triggering rule
    if math.fabs(x - xk) > sigma * x + eta:
        return 1
    else:
        return 0


def plot_data_community(K, L, sigma, eta, W, M, d_table, community_part):
    # define time and gap
    Time = 50000
    h = 0.000035

    # define propotion of infected pepole
    # x_noinput = np.zeros([Time, n])
    # x_feedback = np.zeros([Time, n])
    x_event = np.zeros([Time, n])

    # define initial state
    x0 = np.random.rand(n) / 2
    x0[0] /= 3
    # x_noinput[0] = x0
    # x_feedback[0] = x0
    x_event[0] = x0
    xk = x0

    # define event and objective list
    event = np.zeros([Time, n])
    d_table_list = np.array([d_table for i in range(Time)])

    # create control input series (feedback, event-trigger)
    # u_transition_feedback = np.zeros([Time - 1, n])
    # v_transition_feedback = np.zeros([Time - 1, n])
    u_transition_event = np.zeros([Time - 1, n])
    v_transition_event = np.zeros([Time - 1, n, n])

    # collect transition data of propotion of infected pepole and triggerring event
    for k in range(Time - 1):
        for i in range(n):
            # # event-triggered control
            if event_trigger_func(x_event[k][i], xk[i], sigma[i], eta[i]) == 1:
                xk[i] = x_event[k][i]
                event[k + 1][i] = 1
            v_transition_event[k][i] = L[i] * xk
        x_event[k + 1] = x_event[k] + h * (-(D + K.dot(np.diag(xk))).dot(x_event[k]) + (
            In - np.diag(x_event[k])).dot(B.T - L.dot(np.diag(xk).T)).dot(x_event[k]))
        # # # record event-triggered control input
        u_transition_event[k] = K.dot(xk)

    v_argmax = np.argmax(v_transition_event[0], axis=0)

    # # subplot 2 is the transition data of triggerring event
    color_map = ['darkred', 'red', 'darkorange', 'orange',
                 'gold', 'darkgreen', 'green',
                 'darkcyan', 'cyan', 'deepskyblue', 'navy', 'darkviolet',
                 'magenta', 'olive', 'lightskyblue', 'lime',
                 'crimson', 'peachpuff', 'mediumspringgreen', 'cornflowerblue']
    '''
    fig, ax = plt.subplots(figsize=(16, 9.7))
    color_num = 0
    for i in reversed(range(n)):
        if W[community_part][i] == 1:
            ax.plot(event.T[i], lw=4, color=color_map[color_num], alpha=1)
            color_num += 1
    # # # plot setting
    plt.xlabel(r'Time $[t]$', fontsize=45)
    plt.ylabel('Triggering Event in Community 1', fontsize=45)
    # plt.title('Transition of Triggering Event', fontsize=25)
    ax.xaxis.offsetText.set_fontsize(45)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=45)
    plt.yticks([0, 1], fontsize=45)
    plt.tight_layout()
    plt.grid()
    plt.savefig("./images/triggering_event.pdf", dpi=300)
    '''

    # # subplot 5 is the transition data of U (event)
    fig, ax = plt.subplots(figsize=(16, 9.7))
    color_num = 0
    for i in reversed(range(n)):
        if W[community_part][i] == 1:
            ax.plot(u_transition_event.T[i] / (10 * D_base_max),
                    lw=4, color=color_map[color_num], alpha=1)
            color_num += 1
    # # # plot setting
    plt.xlabel(r'$t$', fontsize=60)
    plt.ylabel(
        r'$u_i(t),$ for $i\in\mathcal{V}_1$', fontsize=60)
    # plt.title(
    #    r'Transition of Event-Triggered Control Input of Recovery Rate $(U)$', fontsize=25)
    ax.xaxis.offsetText.set_fontsize(60)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=60)
    plt.yticks([0, 0.04, 0.08, 0.12, 0.16])
    ax.yaxis.offsetText.set_fontsize(60)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.setp(ax.get_yticklabels(), fontsize=60)
    plt.ylim(0, 0.18)
    plt.tight_layout()
    plt.grid()
    plt.savefig(
        "./images/control_input_of_recovery_rate.pdf", dpi=300)

    # subplot 6 is the transition data of V (event)
    fig, ax = plt.subplots(figsize=(16, 9.7))
    color_num = 0
    for i in reversed(range(n)):
        if W[community_part][i] == 1:
            ax.plot(v_transition_event.T[i]
                    [v_argmax[i]] / (10 * D_base_max), lw=4, color=color_map[color_num], alpha=1)
            color_num += 1
            # for j in range(n):
            #    if B[i][j] != 0:
            #        ax.plot(v_transition_event.T[i][j] / (10 * D_base_max), lw=2)
    # # # plot setting
    plt.xlabel(r'$t$', fontsize=60)
    plt.ylabel(
        r'$v_{ij}(t),$ for $i\in\mathcal{V}_1$', fontsize=60)
    # plt.title(
    #    r'Transition of Event-Triggered Control Input of Infection Rate $(V)$', fontsize=25)
    ax.xaxis.offsetText.set_fontsize(60)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=60)
    # plt.yticks(fontsize=50)
    ax.yaxis.offsetText.set_fontsize(60)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.setp(ax.get_yticklabels(), fontsize=60)
    plt.ylim(0, 0.011)
    plt.tight_layout()
    plt.grid()
    plt.savefig(
        "./images/control_input_of_infection_rate.pdf", dpi=300)


if __name__ == "__main__":
    # load parameters of the event-triggered controller
    D_base_max = 1.8
    D = np.load('./data/matrix/D.npy')
    B = np.load('./data/matrix/B.npy')
    K = np.load('./data/matrix/K.npy')
    L = np.load('./data/matrix/L.npy')
    sigma = np.load('./data/matrix/sigma.npy')
    eta = np.load('./data/matrix/eta.npy')

    # define control objectives according to network
    W, M, d_table, d = create_control_objectives(B)

    # plot data
    plot_data_community(K, L, sigma, eta, W, M, d_table, community_part=0)

    # # no control input
    # x_noinput[k + 1] = x_noinput[k] + h * (-D.dot(x_noinput[k]) + (
    #     In - np.diag(x_noinput[k])).dot(B.T).dot(x_noinput[k]))

    # # feedback control
    # x_feedback[k + 1] = x_feedback[k] + h * (-(D + K.dot(np.diag(x_feedback[k]))).dot(x_feedback[k]) + (
    #     In - np.diag(x_feedback[k])).dot(B.T - L.dot(np.diag(x_feedback[k]).T)).dot(x_feedback[k]))
    # # # record feedback control input
    # u_transition_feedback[k] = K.dot(x_feedback[k])
    # v_transition_feedback[k] = L.dot(x_feedback[k])

    '''
    # caltulate average state transition of targert community
    x_noinput_com_ave = 0
    x_feedback_com_ave = 0
    x_event_com_ave = 0
    community_member_num = 0
    for i in range(n):
        if W[community_part][i] == 1:
            x_noinput_com_ave += x_noinput.T[i]
            x_feedback_com_ave += x_feedback.T[i]
            x_event_com_ave += x_event.T[i]
            community_member_num += 1

    # plot data
    # # subplot 1 is the transition data of x
    fig, ax = plt.subplots(figsize=(16, 9.7))
    ax.plot(x_noinput_com_ave / community_member_num,
            lw=4, label='No Control Input')
    ax.plot(x_feedback_com_ave / community_member_num,
            lw=4, label='Continuous-Time Control')
    ax.plot(x_event_com_ave / community_member_num,
            lw=4, label='Event-Triggered Control')
    ax.plot(d_table_list.T[community_part], lw=4, label=r'$\bar{x}_1 = 0.10$, Threshold for Community1',
            linestyle="dotted", color='navy')

    # # # plot setting
    plt.xlabel(r'Time $[t]$', fontsize=25)
    plt.ylabel(
        r'Average in Community1 $sum_{i in {\rm supp}(w_1)}x_i$', fontsize=25)
    # plt.title(r'Transition of $x$', fontsize=25)
    ax.xaxis.offsetText.set_fontsize(25)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=25)
    # plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()
    plt.grid()
    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
               borderaxespad=0, fontsize=20)
    plt.savefig("./images/transition_of_x_community.png", dpi=300)
    '''

    '''
    # # subplot 3 is the transition data of U (continuous)
    fig, ax = plt.subplots(figsize=(16, 9.7))
    for i in range(n):
        if W[community_part][i] == 1:
            ax.plot(u_transition_feedback.T[i] / (10 * D_base_max), lw=1)
    # # # plot setting
    plt.xlabel(r'Time $[t]$', fontsize=25)
    plt.ylabel(r'Control Input of Recovery Rate $(U)$', fontsize=25)
    plt.title(
        r'Transition of Continuous Control Input of Recovery Rate $(U)$', fontsize=25)
    ax.xaxis.offsetText.set_fontsize(25)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()
    plt.grid()
    plt.savefig(
        "./images/transition_of_continuous_control_input_of_recovery_rate_community.png", dpi=300)

    # subplot 4 is the transition data of V (continuous)
    fig, ax = plt.subplots(figsize=(16, 9.7))
    for i in range(n):
        if W[community_part][i] == 1:
            ax.plot(v_transition_feedback.T[i] / (10 * D_base_max), lw=1)
    # # # plot setting
    plt.xlabel(r'Time $[t]$', fontsize=25)
    plt.ylabel(r'Control Input of Infection Rate $(V)$', fontsize=25)
    plt.title(
        r'Transition of Continuous Control Input of Infection Rate $(V)$', fontsize=25)
    ax.xaxis.offsetText.set_fontsize(25)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()
    plt.grid()
    plt.savefig(
        "./images/transition_of_continuous_control_input_of_infection_rate_community.png", dpi=300)
    '''
