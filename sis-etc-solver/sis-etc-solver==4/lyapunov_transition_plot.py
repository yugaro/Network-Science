import math
import numpy as np
import matplotlib as mpl
import cvxpy as cp
from matplotlib import pyplot as plt
from matplotlib import rc
# from sklearn.neural_network import MLPRegressor
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


def analyse_theta(p, B, D, K, L, G, H):
    # define variables of the state of nodes
    x = cp.Variable(n)

    # define parameter of theorem 1
    s = (K + L.T).dot(In - G).dot(p)
    S = np.diag(s)
    Q = S + 1 / 2 * np.diag(p).dot(L.T).dot(In - G).dot(G + H)
    Q = (Q.T + Q) / 2
    r = ((B.T - D) + (K + L.T).dot(H)).dot(p)

    # define constraint in theorem 1
    constranit_theta = [cp.quad_form(x, Q) - r.T @ x <= 0,
                        0 <= x, x <= 1]

    # define objective function in theorem 1
    theta = p.T @ x

    # solve program of theorem 1 and caluculate theta*
    prob_theta = cp.Problem(cp.Maximize(theta), constranit_theta)
    prob_theta.solve(solver=cp.MOSEK)

    return prob_theta.value


def analyse_theta2(p, B, D, K, L, G, H):
    # define variables of the state of nodes
    x = cp.Variable(n)

    # define parameter of theorem 1
    s = (K + L.T).dot(In - G).dot(p)
    S = np.diag(s)
    # Q = S + 1 / 2 * np.diag(p).dot(L.T).dot(In - G).dot(G + H)
    # Q = (Q.T + Q) / 2
    r = ((B.T - D) + (K + L.T).dot(H)).dot(p)

    # define constraint in theorem 1
    constranit_theta = [cp.quad_form(x, S) - r.T @ x <= 0,
                        0 <= x, x <= 1]

    # define objective function in theorem 1
    theta = p.T @ x

    # solve program of theorem 1 and caluculate theta*
    prob_theta = cp.Problem(cp.Maximize(theta), constranit_theta)
    prob_theta.solve(solver=cp.MOSEK)

    return prob_theta.value


def analyse_theta3(p, B, D, K, L, G, H):
    # define variables of the state of nodes
    x = cp.Variable(n)

    # define parameter of theorem 1
    s = (K + L.T).dot(p)
    S = np.diag(s)
    # Q = S + 1 / 2 * np.diag(p).dot(L.T).dot(In - G).dot(G + H)
    # Q = (Q.T + Q) / 2
    r = (B.T - D).dot(p)

    # define constraint in theorem 1
    constranit_theta = [cp.quad_form(x, S) - r.T @ x <= 0,
                        0 <= x, x <= 1]

    # define objective function in theorem 1
    theta = p.T @ x

    # solve program of theorem 1 and caluculate theta*
    prob_theta = cp.Problem(cp.Maximize(theta), constranit_theta)
    prob_theta.solve(solver=cp.MOSEK)

    return prob_theta.value


def event_trigger_func(x, xk, sigma, eta):
    # define triggering rule
    if math.fabs(x - xk) > sigma * x + eta:
        return 1
    else:
        return 0


def plot_data(p, K, L, sigma, eta):
    # define time and gap
    Time = 100000
    h = 0.000035

    # define propotion of infected pepole
    x = np.zeros([Time, n])
    x0 = np.random.rand(n)
    x0[0] /= 3
    x[0] = x0
    xk = x0

    # define event and objective list
    # d_table_list = np.array([d_table for i in range(Time)])
    Lyapunov_transition = np.zeros([Time - 1])
    thetastar_transition = np.zeros([Time - 1])
    # thetastar2_transition = np.zeros([Time - 1])
    # thetastar3_transition = np.zeros([Time - 1])

    theta_value = analyse_theta(p, B, D, K, L, np.diag(sigma), np.diag(eta))
    print(theta_value / 1.5)
    # thetastar2 = analyse_theta2(p, B, D, K, L, np.diag(sigma), np.diag(eta))
    # thetastar3 = analyse_theta3(p, B, D, K, L, np.diag(sigma), np.diag(eta))
    # print(theta_value)

    # collect transition data of propotion of infected pepole and triggerring event
    for k in range(Time - 1):
        # collect value of Lyapunov func
        Lyapunov_transition[k] = p.dot(x[k]) / 1.8
        thetastar_transition[k] = theta_value / 1.5
        # thetastar2_transition[k] = thetastar2
        # thetastar3_transition[k] = thetastar3
        for i in range(n):
            if event_trigger_func(x[k][i], xk[i], sigma[i], eta[i]) == 1:
                xk[i] = x[k][i]
        x[k + 1] = x[k] + h * (-(D + K.dot(np.diag(xk))).dot(x[k]) + (
            In - np.diag(x[k])).dot(B.T - L.dot(np.diag(xk).T)).dot(x[k]))

    # transition Lyapunov
    fig, ax = plt.subplots(figsize=(16, 9.7))
    ax.plot(Lyapunov_transition, lw=5, color='royalblue')
    ax.plot(thetastar_transition, lw=5, linestyle="dashdot",
            color='magenta', label=r'$\theta^* = 0.304$')
    ax.xaxis.offsetText.set_fontsize(60)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=60)
    plt.xlabel(r'$t$', fontsize=60)
    plt.ylabel(r'$V(x(t))$', fontsize=60)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=60)
    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
               borderaxespad=0, fontsize=60)
    plt.tight_layout()
    plt.grid()
    plt.savefig("./images/transition_of_Lyapunov_func.pdf", dpi=300)


if __name__ == '__main__':
    # load design parameter
    D = np.load('./data/matrix/D.npy')
    B = np.load('./data/matrix/B.npy')
    K = np.load('./data/matrix/K.npy')
    L = np.load('./data/matrix/L.npy')
    sigma = np.load('./data/matrix/sigma.npy')
    eta = np.load('./data/matrix/eta.npy')
    p = np.load('./data/matrix/p.npy')

    # plot data
    plot_data(p, K, L, sigma, eta)
