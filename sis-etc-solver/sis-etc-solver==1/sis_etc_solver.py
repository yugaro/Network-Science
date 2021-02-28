import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cvxpy as cp
from sklearn.neural_network import MLPRegressor
from matplotlib import rc
np.random.seed(0)
rc('text', usetex=True)
rc('font', **{'family': "sans-serif"})
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

# number of nodes (max: 50)
n = 50

# preparation
INF = 1e9
epsilon = 1e-8
In = np.identity(n)
On = np.zeros((n, n))

# matrix of infection rates (air route matrix in case of the passengers and flights)
B = pd.read_csv('./data/usa_airport_ad_matrix.csv',
                index_col=0, nrows=n, usecols=[i for i in range(n + 1)]).values

# matrix of recovery rates
D_base_max = 5
D_base_min = 4
D = np.diag((D_base_max - D_base_min) * np.random.rand(n) + D_base_min)

# maximum range of control paramter of infection rate (L)
barl = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if B[i][j] != 0:
            barl[i][j] = B[i][j] - epsilon

# maximum range of control paramter of recovery rate (K)
bark_max = 6
bark_min = 4
bark = np.diag((bark_max - bark_min) * np.random.rand(n) + bark_min)

# define control objectives
# # number of control objectives
M = 1

# # define target nodes
W = np.ones((M, n))

# # define threshold of each node in the target
d_table = np.array([0.1])

# # define threshold of each taget
d = np.zeros(M)
for m in range(M):
    for i in range(n):
        if W[m][i] == 1:
            d[m] += d_table[m]


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


def p_min_supp_w(p):
    # caluculate p_min in each target
    p_min = []
    for i in range(M):
        tmp_p = INF
        for j in range(n):
            if W[i][j] == 1 and tmp_p > p[j]:
                tmp_p = p[j]
        p_min += [tmp_p]
    return p_min


def analyse_theta(p, B, D, K, L, G, H):
    # define variables of the state of nodes
    x = cp.Variable(n)

    # define parameter of theorem 1
    s = (K + L.T).dot(In - G).dot(p)
    S = np.diag(s)
    Q = S + 1 / 2 * np.diag(p).dot(L.T).dot(In - G).dot(G + H)
    r = ((B - D.T) + (K + L.T).dot(H)).dot(p)

    # define constraint in theorem 1
    if np.all(Q):
        constranit_theta = [x @ Q @ x - r.T @ x <= 0,
                            0 <= x, x <= 1]
    else:
        constranit_theta = [- r.T @ x <= 0,
                            0 <= x, x <= 1]

    # define objective function in theorem 1
    theta = p.T @ x

    # solve program of theorem 1 and caluculate theta*
    prob_theta = cp.Problem(cp.Maximize(theta), constranit_theta)
    prob_theta.solve()

    return prob_theta.value


def control_parameter_solver_gp(p, rc):
    # define varialbe
    tildek = cp.Variable((n, n), pos=True)
    tildel = cp.Variable((n, n), pos=True)
    s = cp.Variable(n, pos=True)
    xi = cp.Variable(1, pos=True)

    # define constant
    # # c1 = p bark + sum(p barl)
    c1 = np.zeros(n)
    for i in range(n):
        c1[i] = p[i] * bark[i][i]
        for j in range(n):
            if B[i][j] != 0:
                c1[i] += p[j] * barl[i][j]

    # # c2 = 2 p_m *d - sum_{rc < 0} p * rc / c1
    c2 = np.zeros(M)
    p_m = p_min_supp_w(p)
    for m in range(M):
        c2[m] = 2 * p[m] * d[m]
        for i in range(n):
            if rc[i] < 0:
                c2[m] -= p[i] * rc[i] / c1[i]

    # create constraints
    # # tildeK + epsilon <= barK
    gp_const1_c = [tildek[i][i] + epsilon <= bark[i][i] for i in range(n)]

    # # tildeL + epsilon <= barL
    gp_const2_c = [tildel[i][j] + epsilon <= barl[i][j]
                   for i in range(n) for j in range(n) if B[i][j] != 0]

    # # s + p tildek + sum(p tildel) <= c1
    gp_const3_c = []
    for i in range(n):
        tmp_const3_c = 0
        for j in range(n):
            if B[i][j] != 0:
                tmp_const3_c += p[j] * tildel[i][j]
        gp_const3_c += [s[i] + p[i] * tildek[i][i] + tmp_const3_c <= c1[i]]

    # # sum(p^2/s) * sum(r^2/s) <= xi
    tmp1_const4_c = 0
    tmp2_const4_c = 0
    for i in range(n):
        tmp1_const4_c += p[i]**2 / s[i]
        tmp2_const4_c += rc[i]**2 / s[i]
    gp_const4_c = [tmp1_const4_c * tmp2_const4_c <= xi]

    # # xi + sum_{rc >= 0}( p * rc / s) <= c2
    gp_const5_c = []
    for m in range(M):
        tmp_const5_c = 0
        for i in range(n):
            if rc[i] >= 0:
                tmp_const5_c += p[i] * rc[i] / s[i]
        gp_const5_c += [xi**0.5 + tmp_const5_c <= c2[m]]

    # configure gp constraints of control paramters
    gp_consts_c = gp_const1_c + gp_const2_c + \
        gp_const3_c + gp_const4_c + gp_const5_c

    # create objective func and solve GP (control parameters)
    gp_fc = 1
    for i in range(n):
        gp_fc += bark[i][i] / tildek[i][i]
        for j in range(n):
            if B[i][j] != 0:
                gp_fc += barl[i][j] / tildel[i][j]
    gp_prob_c = cp.Problem(cp.Maximize(1 / gp_fc), gp_consts_c)
    gp_prob_c.solve(gp=True)
    print("GP status (control parameters):", gp_prob_c.status)

    # get value of K and L
    K = bark - np.array(tildek.value)
    L = barl - np.array(tildel.value)
    for i in range(n):
        for j in range(n):
            if i != j:
                K[i][j] = 0
            if B[i][j] == 0:
                L[i][j] = 0

    return K, L


def triggered_parameter_solver_gp(p, rc, Kstar, Lstar):
    # define variable
    tildesigma = cp.Variable(n, pos=True)
    eta = cp.Variable(n, pos=True)
    r = cp.Variable(n, pos=True)
    s = cp.Variable(n, pos=True)
    xi = cp.Variable(1, pos=True)

    # c3 = p K* + sum(p L*)
    c3 = np.zeros(n)
    for i in range(n):
        c3[i] = p[i] * Kstar[i][i]
        for j in range(n):
            if B[i][j] != 0:
                c3[i] += p[j] * Lstar[i][j]

    # create constraints
    # # tildesigma + epsilon <= 1
    gp_const1_e = [tildesigma[i] + epsilon <= 1 for i in range(n)]

    # # eta + epsilon <= 1
    gp_const2_e = [eta[i] + epsilon <= 1 for i in range(n)]

    # # s / tildesigma <= c3
    gp_const3_e = [s[i] / tildesigma[i] <= c3[i] for i in range(n)]

    # # max{0, rc} + c3 * eta <= r
    gp_const4_e = []
    for i in range(n):
        if rc[i] >= 0:
            gp_const4_e += [rc[i] + c3[i] * eta[i] <= r[i]]
        elif rc[i] < 0 and -rc[i] < c3[i]:
            gp_const4_e += [eta[i] == -rc[i] / c3[i]]
        else:
            gp_const4_e += [c3[i] * eta[i] <= r[i]]

    # # (sum p^2 / s) * (sum r^2 / s) <= xi1
    tmp1_const5_e = 0
    tmp2_const5_e = 0
    for i in range(n):
        tmp1_const5_e += p[i] ** 2 / s[i]
        tmp2_const5_e += r[i] ** 2 / s[i]
    gp_const5_e = [tmp1_const5_e * tmp2_const5_e <= xi]

    # # xi ** 0.5 + sum p * r / s <= 2 * p_m * d
    gp_const6_e = []
    p_m = p_min_supp_w(p)
    for m in range(M):
        tmp_const6_e = 0
        for i in range(n):
            tmp_const6_e += p[i] * r[i] / s[i]
        gp_const6_e += [xi**0.5 + tmp_const6_e <= 2 * p_m[m] * d[m]]

    # # configure gp constraints
    gp_consts_e = gp_const1_e + gp_const2_e + gp_const3_e + \
        gp_const4_e + gp_const5_e + gp_const6_e

    # create objective funcition and solve GP (triggered paramters)
    gp_fe = 1
    for i in range(n):
        gp_fe *= eta[i] / tildesigma[i]
    gp_prob_e = cp.Problem(cp.Maximize(gp_fe), gp_consts_e)
    gp_prob_e.solve(gp=True)
    print("GP status (event-triggered paramters) :", gp_prob_e.status)

    # get value of sigma and eta
    sigma = 1 - np.array(tildesigma.value)
    eta = np.array(eta.value)
    return sigma, eta


def data_info(p, K, L, sigma, eta):
    print("W\n", W)
    print("d\n", d_table)
    print("B\n", B)
    print("D\n", D)
    print("p\n", p)
    print("K\n", K)
    print("L\n", L)
    print("sigma\n", sigma)
    print("eta\n", eta)


def event_trigger_func(x, xk, sigma, eta):
    # define triggering rule
    if math.fabs(x - xk) > sigma * x + eta:
        return 1
    else:
        return 0


def plot_data(K, L, sigma, eta, choice):
    # define time and gap
    Time = 30000
    h = 0.00005

    # define propotion of infected pepole
    x = np.zeros([Time, n])
    x0 = np.random.rand(n)
    x[0] = x0
    xk = x0

    # define event and objective list
    event = np.zeros([Time, n])
    d_table_list = np.array([d_table for i in range(Time)])
    u_transition = np.zeros([Time - 1, n])
    v_transition = np.zeros([Time - 1, n])

    # collect transition data of propotion of infected pepole and triggerring event
    for k in range(Time - 1):
        for i in range(n):
            # # choice 1 has no control input
            if choice == 1:
                x[k + 1] = x[k] + h * \
                    (-D.dot(x[k]) + (In - np.diag(x[k])).dot(B.T).dot(x[k]))

            # # In the case of using feedback controller
            else:
                # # choice 2 is the case of continuous controller
                if choice == 2 and event_trigger_func(x[k][i], xk[i], 0, 0) == 1:
                    xk[i] = x[k][i]
                    event[k + 1][i] = 1

                # # choice 3 is the case of event-triggered controller
                elif choice == 3 and event_trigger_func(x[k][i], xk[i], sigma[i], eta[i]) == 1:
                    xk[i] = x[k][i]
                    event[k + 1][i] = 1
                x[k + 1] = x[k] + h * (-(D + K.dot(np.diag(xk))).dot(x[k]) + (
                    In - np.diag(x[k])).dot(B.T - L.dot(np.diag(xk).T)).dot(x[k]))
                u_transition[k] = K.dot(xk)
                v_transition[k] = L.dot(xk)

    # plot data
    # # subplot 1 is the transition data of x
    fig, ax = plt.subplots(figsize=(16, 9.7))
    for i in range(n):
        ax.plot(x.T[i], lw=1)
    for m in range(M):
        ax.plot(d_table_list.T[m], lw=5, linestyle="dashdot")

    # # # plot setting
    plt.xlabel('Time', fontsize=35)
    plt.ylabel(r'$x$', fontsize=35)
    plt.title(r'Transition of $x$', fontsize=35)
    ax.xaxis.offsetText.set_fontsize(25)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=35)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=35)
    plt.tight_layout()
    plt.grid()
    plt.savefig("transition_of_x.png")

    # # subplot 2 is the transition data of triggerring event
    fig, ax = plt.subplots(figsize=(16, 9.7))
    for i in range(n):
        ax.plot(event.T[i], lw=1)
    # # # plot setting
    plt.xlabel('Time', fontsize=35)
    plt.ylabel('triggering event', fontsize=35)
    plt.title('Transition of triggering event', fontsize=35)
    ax.xaxis.offsetText.set_fontsize(25)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=35)
    plt.yticks([0, 1], fontsize=35)
    plt.tight_layout()
    plt.grid()
    plt.savefig("triggering_event.png")

    # # subplot 3 is the transition data of U
    fig, ax = plt.subplots(figsize=(16, 9.7))
    for i in range(n):
        ax.plot(u_transition.T[i], lw=1)
    # # # plot setting
    plt.xlabel('Time', fontsize=35)
    plt.ylabel(r'$U$', fontsize=40)
    plt.title(r'Transition of $U$', fontsize=35)
    ax.xaxis.offsetText.set_fontsize(25)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=35)
    plt.yticks(fontsize=35)
    plt.tight_layout()
    plt.grid()
    plt.savefig("transition_of_control_input_about_recovery_rate.png")

    # subplot 4 is the transition data of V
    fig, ax = plt.subplots(figsize=(16, 9.7))
    for i in range(n):
        ax.plot(v_transition.T[i], lw=1)
    # # # plot setting
    plt.xlabel('Time', fontsize=35)
    plt.ylabel(r'$V$', fontsize=35)
    plt.title(r'Transition of $V$', fontsize=35)
    ax.xaxis.offsetText.set_fontsize(25)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=35)
    plt.yticks(fontsize=35)
    plt.tight_layout()
    plt.grid()
    plt.savefig("transition_of_control_input_about_infection_rate.png")


if __name__ == '__main__':
    # design Lyapunov parameter
    p, rc = lyapunov_param_solver(B, D)

    # analyse thetastar in the case of no input
    thetastar_noinput = analyse_theta(p, B, D, On, On, On, On)

    # judge whether control objectives can be achieve in the case of no control input
    p_min = p_min_supp_w(p)
    judge_noinput = [thetastar_noinput <= d[m] * p_min[m] for m in range(M)]

    # design event-triggered controller
    if np.all(judge_noinput):
        print('Control objectives can be achieved without control inputs.')
    else:
        # design control paramters
        K, L = control_parameter_solver_gp(p, rc)

        # design triggered prapeters
        sigma, eta = triggered_parameter_solver_gp(p, rc, K, L)

        # print parameter info
        data_info(p, K, L, sigma, eta)

        # plot data
        plot_data(K, L, sigma, eta, choice=3)
