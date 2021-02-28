import numpy as np
import cvxpy as cp
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

if __name__ == "__main__":
    D = np.load('./data/matrix/D.npy')
    B = np.load('./data/matrix/B.npy')
    K = np.load('./data/matrix/K.npy')
    L = np.load('./data/matrix/L.npy')
    sigma = np.load('./data/matrix/sigma.npy')
    eta = np.load('./data/matrix/eta.npy')
    p = np.load('./data/matrix/p.npy')

    thetastar = analyse_theta(p, B, D, K, L, np.diag(sigma), np.diag(eta))
    print('theta*\n', thetastar)

    thetastar2 = analyse_theta2(p, B, D, K, L, np.diag(sigma), np.diag(eta))
    print('theta2*\n', thetastar2)

    thetastar3 = analyse_theta3(p, B, D, K, L, np.diag(sigma), np.diag(eta))
    print('theta3*\n', thetastar3)

    eigenvalue, eigenvector = np.linalg.eig(B - D)
    print('eigenvalue (B - D)\n', eigenvalue)

    '''
    # caluculate eigenvalue and eigenvector
    eigenvalue, eigenvector = np.linalg.eig(B - D)
    print('eigenvalue (B - D)\n', eigenvalue)

    # caluculate eigenvalue and eigenvector
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if B[i][j] != 0:
                A[i][j] = 1
    eigenvalue, eigenvector = np.linalg.eig(A)
    print('eigenvalue A \n', np.array(eigenvalue).max())

    # caluculate spectral radius (B)
    spectral_radius = np.linalg.norm(B, 2)
    print('spectral radius B \n', spectral_radius)

    # caluculate spectral radius (B - D)
    spectral_radius = np.linalg.norm((B - D), 2)
    print('spectral radius (B - D)\n', spectral_radius)
    '''
    # caluculate Lyapunov paramters
    # p, rc = lyapunov_param_solver(B, D)
    # print('spectral radius D^{-1}B\n', np.linalg.norm(np.linalg.inv(D).dot(B), 2))
    # calculate thetastar
