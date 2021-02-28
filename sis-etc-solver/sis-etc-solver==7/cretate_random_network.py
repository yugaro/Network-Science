import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import networkx as nx
np.random.seed(0)
# number of nodes (max: 50)
n = 250
M = 1

# preparation
INF = 1e9
epsilon = 1e-15
In = np.identity(n)
On = np.zeros((n, n))

B_max = 0.03
B_min = 0.001
B = (B_max - B_min) * np.random.rand(n, n) + B_min
edge_num = 0
for i in range(n):
    B[i][i] = 0
    for j in range(n):
        if np.random.normal() > -1.9:
            B[i][j] = 0
        elif i != j:
            edge_num += 1
print(edge_num)

for i in range(n):
    if np.all(B[i] == 0) or np.all(B.T[i] == 0):
        print('no')

D = np.diag(np.ones(n)) * 0.1

bark_const = 10
bark = np.diag(np.ones(n) * bark_const)

barl = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if B[i][j] != 0:
            barl[i][j] = (B[i][j] - epsilon)

W = np.ones((M, n))
d_table = [0.1]
d = np.zeros((M))
for m in range(M):
    for i in range(n):
        if W[m][i] == 1:
            d[m] += d_table[m]


def depict_graph(B):
    # define weighted multi-directed graph
    G = nx.Graph()

    # create SIS network
    edge_list = [(i, j, B[i][j])
                 for i in range(n) for j in range(n) if B[i][j] != 0]
    G.add_weighted_edges_from(edge_list)

    # define node size
    # eigv_cent = nx.eigenvector_centrality_numpy(G)
    lambda_b, v_b = np.linalg.eig(B.T)
    index_b = np.where(lambda_b == max(lambda_b))[0][0]
    eigenvector_cent = np.abs(v_b[:, index_b])
    node_size = np.array([(size ** 4)
                          for size in eigenvector_cent]) * 40000000
    # node_size = np.array([(size ** 4)
    #                       for size in list(eigenvector_cent.values())]) * 20000000

    # define edge width
    width = np.array([d['weight'] for (u, v, d) in G.edges(data=True)])
    width_std = 14 * (((width - min(width)) / (max(width) - min(width)))) + 0.5

    # define label name
    # plot graph
    plt.figure(figsize=(60, 60))
    # pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    pos = nx.spring_layout(G, k=1)

    # set drawing nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
    )

    # set drawing edges
    nx.draw_networkx_edges(
        G,
        pos,
        width=width_std
    )

    # plt.legend(bbox_to_anchor=(0.3, 1), loc='upper left', borderaxespad=0, fontsize=19.5)
    plt.tight_layout()
    plt.gca().set_axis_off()
    plt.savefig('./images/random_network.png')


def lyapunov_param_solver(B, D):
    p_v = cp.Variable(n, pos=True)
    rc_v = cp.Variable(n)

    lyapunov_cons1 = [rc_v == (B.T - D) @ p_v]
    lyapunov_cons2 = [0.000001 <= p_v[i] for i in range(n)]
    lyapunov_cons3 = [p_v[i] <= 0.0000012 for i in range(n)]
    lyapunov_constraints = lyapunov_cons1 + lyapunov_cons2 + lyapunov_cons3

    f_ly = 0
    for i in range(n):
        f_ly += rc_v[i]

    prob_lyapunov = cp.Problem(cp.Minimize(f_ly), lyapunov_constraints)
    prob_lyapunov.solve(solver=cp.MOSEK)

    p = np.array(p_v.value)
    p = p / np.linalg.norm(p)
    rc = (B.T - D).dot(p)

    return p, rc


def lyapunov_param_solver2(B, D):
    p = np.ones(n)
    rc = (B.T - D).dot(p)
    return p, rc


def p_min_supp_w(p, W, M):
    # caluculate p_min in each target
    p_min = []
    for m in range(M):
        tmp_p = INF
        for j in range(n):
            if W[m][j] == 1 and tmp_p > p[j]:
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


def control_parameter_solver_gp(p, rc, W, M, d):
    # define varialbe
    tildek = cp.Variable((n, n), pos=True)
    tildel = cp.Variable((n, n), pos=True)
    s = cp.Variable((M, n), pos=True)
    xi = cp.Variable((M, 1), pos=True)

    # define constant
    # # c1 = p bark + sum(p barl)
    c1 = np.zeros((M, n))
    for m in range(M):
        for i in range(n):
            if W[m][i] == 1:
                c1[m][i] = p[i] * bark[i][i]
                for j in range(n):
                    if B[i][j] != 0:
                        c1[m][i] += p[j] * barl[i][j]

    # # c2 = 2 p_m *d - sum_{rc < 0} p * rc / c1
    c2 = np.zeros(M)
    p_m = p_min_supp_w(p, W, M)
    for m in range(M):
        c2[m] = 2 * p[m] * d[m]
        for i in range(n):
            if rc[i] < 0 and W[m][i] == 1:
                c2[m] -= p[i] * rc[i] / c1[m][i]

    # create constraints
    # # tildeK + epsilon <= barK
    gp_const1_c = [tildek[i][i] + epsilon <= bark[i][i] for i in range(n)]

    # # tildeL + epsilon <= barL
    gp_const2_c = [tildel[i][j] + epsilon <= barl[i][j]
                   for i in range(n) for j in range(n) if B[i][j] != 0]

    # # s + p tildek + sum(p tildel) <= c1
    gp_const3_c = []
    for m in range(M):
        for i in range(n):
            if W[m][i] == 1:
                tmp_const3_c = 0
                for j in range(n):
                    if B[i][j] != 0:
                        tmp_const3_c += p[j] * tildel[i][j]
                gp_const3_c += [s[m][i] + p[i] * tildek[i]
                                [i] + tmp_const3_c <= c1[m][i]]

    # # sum(p^2/s) * sum(r^2/s) <= xi
    gp_const4_c = []
    for m in range(M):
        tmp1_const4_c = 0
        tmp2_const4_c = 0
        for i in range(n):
            if W[m][i] == 1:
                tmp1_const4_c += p[i]**2 / s[m][i]
                tmp2_const4_c += rc[i]**2 / s[m][i]
        gp_const4_c += [tmp1_const4_c * tmp2_const4_c <= xi[m]]

    # # xi + sum_{rc >= 0}( p * rc / s) <= c2
    gp_const5_c = []
    for m in range(M):
        tmp_const5_c = 0
        for i in range(n):
            if rc[i] >= 0 and W[m][i] == 1:
                tmp_const5_c += p[i] * rc[i] / s[m][i]
        gp_const5_c += [xi[m]**0.5 + tmp_const5_c <= c2[m]]

    # configure gp constraints of control paramters
    gp_consts_c = gp_const1_c + gp_const2_c + \
        gp_const3_c + gp_const4_c + gp_const5_c

    # create objective func and solve GP (control parameters)

    gp_fc = 1
    k_weight = 100000000000000
    for i in range(n):
        gp_fc += bark[i][i] * k_weight / tildek[i][i]
        for j in range(n):
            if B[i][j] != 0:
                gp_fc += barl[i][j] / tildel[i][j]

    gp_prob_c = cp.Problem(cp.Maximize(1 / gp_fc), gp_consts_c)
    gp_prob_c.solve(gp=True, solver=cp.MOSEK)
    print("GP status (control parameters):", gp_prob_c.status)

    # get value of K and L
    Kstar = bark - np.array(tildek.value)
    Lstar = barl - np.array(tildel.value)
    for i in range(n):
        for j in range(n):
            if i != j:
                Kstar[i][j] = 0
            if B[i][j] == 0:
                Lstar[i][j] = 0

    return Kstar, Lstar


def triggered_parameter_solver_gp(p, rc, Kstar, Lstar, W, M, d):
    # define variable
    sigma = cp.Variable(n, pos=True)
    eta = cp.Variable(n, pos=True)
    r = cp.Variable((M, n), pos=True)
    s = cp.Variable((M, n), pos=True)
    xi = cp.Variable((M, 1), pos=True)
    xi2 = cp.Variable((M, n), pos=True)

    # c3 = p K* + sum(p L*)
    c3 = np.zeros((M, n))
    for m in range(M):
        for i in range(n):
            if W[m][i] == 1:
                c3[m][i] = p[i] * Kstar[i][i]
                for j in range(n):
                    if B[i][j] != 0:
                        c3[m][i] += p[j] * Lstar[i][j]

    # create constraints
    # # sigma + epsilon <= 1
    gp_const1_e = [sigma[i] + epsilon <= 1 for i in range(n)]

    # # eta + epsilon <= 1
    gp_const2_e = [eta[i] + epsilon <= 1 for i in range(n)]

    # # s / tildesigma <= c3
    gp_const3_e = [sigma[i] + 1 / xi2[m][i] <=
                   1 for m in range(M) for i in range(n) if W[m][i] == 1]
    gp_const3_e += [s[m][i] * xi2[m][i] <= c3[m][i]
                    for m in range(M) for i in range(n) if W[m][i] == 1]

    # # max{0, rc} + c3 * eta <= r
    gp_const4_e = []
    for m in range(M):
        for i in range(n):
            if rc[i] >= 0 and W[m][i] == 1:
                gp_const4_e += [rc[i] + c3[m][i] * eta[i] <= r[m][i]]
            elif rc[i] < 0 and -rc[i] < c3[m][i] and W[m][i] == 1:
                gp_const4_e += [eta[i] == -rc[i] / c3[m][i]]
            elif W[m][i] == 1:
                gp_const4_e += [c3[m][i] * eta[i] <= r[m][i]]

    # # (sum p^2 / s) * (sum r^2 / s) <= xi1
    gp_const5_e = []
    for m in range(M):
        tmp1_const5_e = 0
        tmp2_const5_e = 0
        for i in range(n):
            if W[m][i] == 1:
                tmp1_const5_e += p[i] ** 2 / s[m][i]
                tmp2_const5_e += r[m][i] ** 2 / s[m][i]
        gp_const5_e += [tmp1_const5_e * tmp2_const5_e <= xi[m]]

    # # xi ** 0.5 + sum p * r / s <= 2 * p_m * d
    gp_const6_e = []
    p_m = p_min_supp_w(p, W, M)
    for m in range(M):
        tmp_const6_e = 0
        for i in range(n):
            if W[m][i] == 1:
                tmp_const6_e += p[i] * r[m][i] / s[m][i]
        gp_const6_e += [xi[m]**0.5 + tmp_const6_e <= 2 * p_m[m] * d[m]]

    # # configure gp constraints
    gp_consts_e = gp_const1_e + gp_const2_e + gp_const3_e + \
        gp_const4_e + gp_const5_e + gp_const6_e

    # create objective funcition and solve GP (triggered paramters)
    gp_fe = 1
    for i in range(n):
        gp_fe += 1 / (sigma[i]) + 1 / (eta[i])
    gp_prob_e = cp.Problem(cp.Maximize(1 / gp_fe), gp_consts_e)
    gp_prob_e.solve(gp=True, solver=cp.MOSEK)
    print("GP status (event-triggered paramters) :", gp_prob_e.status)

    # get value of sigma and eta
    sigmastar = np.array(sigma.value)
    etastar = np.array(eta.value)
    return sigmastar, etastar


def data_info(p, K, L, sigma, eta, W, M, d_table):
    print("W\n", W)
    print("d\n", d_table)
    print("B\n", B)
    print("D\n", D)
    print("p\n", p)
    print("K\n", K)
    print("L\n", L)
    print("sigma\n", sigma)
    print("eta\n", eta)


if __name__ == '__main__':

    # depict graph structure
    # depict_graph(B)

    # design Lyapunov parameter
    p, rc = lyapunov_param_solver2(B, D)

    # analyse thetastar in the case of no input
    thetastar_noinput = analyse_theta(p, B, D, On, On, On, On)
    print('theta star (noinput):', thetastar_noinput)

    # judge whether control objectives can be achieve in the case of no control input
    p_min = p_min_supp_w(p, W, M)
    judge_noinput = [thetastar_noinput <= d[m] * p_min[m] for m in range(M)]

    # design event-triggered controller
    if np.all(judge_noinput):
        print('Control objectives can be achieved without control inputs.')
    else:
        # design control paramters
        K, L = control_parameter_solver_gp(p, rc, W, M, d)

        # design triggered prapeters
        sigma, eta = triggered_parameter_solver_gp(p, rc, K, L, W, M, d)
        print('theta star (control):', analyse_theta(
            p, B, D, K, L, np.diag(sigma), np.diag(eta)))

        # print parameter info
        data_info(p, K, L, sigma, eta, W, M, d_table)

        np.save('data/matrix/D', D)
        np.save('data/matrix/B', B)
        np.save('data/matrix/K', K)
        np.save('data/matrix/L', L)
        np.save('data/matrix/sigma', sigma)
        np.save('data/matrix/eta', eta)
        np.save('data/matrix/p', p)
        np.save('data/matrix/W', W)
        np.save('data/matrix/d_table', d_table)
        np.save('data/matrix/M', M)
        np.save('data/matrix/d', d)
        np.save('data/matrix/rc', rc)
