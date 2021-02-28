import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
import cvxpy as cp
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

# matrix of recovery rates
D_base_max = 1.8
D_base_min = 1.5
D = np.diag(np.sort((D_base_max - D_base_min) *
                    np.random.rand(n) + D_base_min)[::-1])

# matrix of infection rates (air route matrix in case of the passengers and flights)
B = pd.read_csv('./data/US_Airport_Ad_Matrix.csv',
                index_col=0, nrows=n, usecols=[i for i in range(n + 1)]).values


def lyapunov_param_solver(B, D):
    # define SVM
    mlp = MLPRegressor(activation='relu',
                       early_stopping=False,
                       alpha=0.000100,
                       max_iter=500,)

    # define target vector
    target = (B - D).dot(np.ones(n)) * 1
    # print(target)

    # model fitting
    mlp.fit(B - D, target)

    # calculate Lyapunov param
    p = mlp.predict(B - D).dot(np.linalg.inv(B - D))

    # add bias and implement normalization
    # p += np.abs(p.min()) + 1
    p /= np.linalg.norm(p)

    # caluculate rc
    rc = (B.T - D).dot(p)

    return p, rc

def lyapunov_param_solver2(B, D):
    p_v = cp.Variable(n)
    rc_v = cp.Variable(n)

    lyapunov_cons1 = [rc_v == (B - D) @ p_v]
    lyapunov_cons2 = [p_v[i] >= 0.00000000001 for i in range(n)]

    lyapunov_constraints = lyapunov_cons1 + lyapunov_cons2

    f_ly = 0
    for i in range(n):
        f_ly += cp.pos(rc_v[i])

    prob_lyapunov = cp.Problem(cp.Minimize(f_ly), lyapunov_constraints)
    prob_lyapunov.solve(solver=cp.MOSEK)

    p = np.array(p_v.value)
    rc = np.array(rc_v.value)
    p = p / np.linalg.norm(p, 2)
    rc = rc / np.linalg.norm(p, 2)
    return p, rc


def lyapunov_param_solver3(B, D):
    p_v = cp.Variable(n, pos=True)
    rc_v = cp.Variable(n)

    lyapunov_cons1 = [rc_v == (B - D) @ p_v]
    lyapunov_cons2 = [p_v[i] >= 0.0000001 for i in range(n)]
    # lyapunov_constraints = lyapunov_cons1
    # lyapunov_constraints = lyapunov_cons2
    lyapunov_constraints = lyapunov_cons1 + lyapunov_cons2

    f_ly = 0
    for i in range(n):
        f_ly += cp.logistic(rc_v[i])

    prob_lyapunov = cp.Problem(cp.Minimize(f_ly), lyapunov_constraints)
    prob_lyapunov.solve(solver=cp.MOSEK)

    p = np.array(p_v.value)
    rc = np.array(rc_v.value)
    p = p / np.linalg.norm(p, 2)
    rc = rc / np.linalg.norm(p, 2)
    return p, rc


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
    if np.all(Q == 0):
        constranit_theta = [- r.T @ x <= 0,
                            0 <= x, x <= 1]
    else:
        constranit_theta = [cp.quad_form(x, Q) - r.T @ x <= 0,
                            0 <= x, x <= 1]

    # define objective function in theorem 1
    theta = p.T @ x

    # solve program of theorem 1 and caluculate theta*
    prob_theta = cp.Problem(cp.Maximize(theta), constranit_theta)
    prob_theta.solve(solver=cp.MOSEK)

    return prob_theta.value

if __name__ == '__main__':

    # design Lyapunov parameter
    # p, rc = lyapunov_param_solver(B, D)
    # p, rc = lyapunov_param_solver2(B, D)
    p, rc = lyapunov_param_solver3(B, D)
    print(p)
    # load data
    # D = np.load('./data/matrix/D.npy')
    # B = np.load('./data/matrix/B.npy')
    # K = np.load('./data/matrix/K.npy')
    # L = np.load('./data/matrix/L.npy')
    # sigma = np.load('./data/matrix/sigma.npy')
    # eta = np.load('./data/matrix/eta.npy')

    theta = analyse_theta(p, B, D, On, On, On, On)
    print(theta)
