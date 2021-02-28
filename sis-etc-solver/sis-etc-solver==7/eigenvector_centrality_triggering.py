import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
np.random.seed(0)
np.random.seed(0)
rc('text', usetex=True)
rc('font', **{'family': "sans-serif"})
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

# number of nodes (max: 50)
n = 250
M = 1

# preparation
INF = 1e9
epsilon = 1e-15
In = np.identity(n)
On = np.zeros((n, n))


def plot_data(B, D, K, L, sigma, eta, d_table):

    lambda_b, v_b = np.linalg.eig(B.T)
    index_b = np.where(lambda_b == max(lambda_b))[0][0]
    eigenvector_cent = np.abs(v_b[:, index_b])

    B_sum = np.zeros(n)
    L_sum = np.zeros(n)
    for i in range(n):
        for j in range(n):
            B_sum[i] += B[j][i]
            L_sum[i] += L[j][i]

    # eigen - sigma
    fig = plt.figure(figsize=(16, 9.7))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlabel(r'eigenvector centrality', fontsize=60)
    ax.set_ylabel(r'$\sigma_i$', fontsize=60)
    ax.scatter(eigenvector_cent,
               sigma, s=100, alpha=1)
    # ax.set_yscale('log')
    ax.set_xticks([0, 0.1, 0.2])
    ax.set_xticklabels([r'$0$', r'$0.1$', r'$0.2$'])
    ax.set_yticks([0.025, 0.075, 0.125])
    ax.set_yticklabels([r'$0$', r'$0.1$', r'$0.2$'])
    ax.tick_params(axis='x', labelsize=60)
    ax.tick_params(axis='y', labelsize=60)
    ax.grid(which='major', alpha=0.8, linestyle='dashed')
    fig.savefig("./images/eigen_sigma.pdf", bbox_inches="tight", dpi=300)

    # eigen - eta
    fig = plt.figure(figsize=(16, 9.7))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlabel(r'eigenvector centrality', fontsize=60)
    ax.set_ylabel(r'$\eta_i$', fontsize=60)
    ax.scatter(eigenvector_cent,
               eta, s=100, alpha=1)
    ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=60)
    ax.tick_params(axis='y', labelsize=60)
    fig.savefig("./images/eigen_eta.pdf", bbox_inches="tight", dpi=300)

if __name__ == '__main__':
    # load data
    D = np.load('./data/matrix/D.npy')
    B = np.load('./data/matrix/B.npy')
    K = np.load('./data/matrix/K.npy')
    L = np.load('./data/matrix/L.npy')
    sigma = np.load('./data/matrix/sigma.npy')
    eta = np.load('./data/matrix/eta.npy')
    p = np.load('./data/matrix/p.npy')
    W = np.load('./data/matrix/W.npy')
    d_table = np.load('./data/matrix/d_table.npy')
    W = np.load('./data/matrix/W.npy')

    # plot data
    plot_data(B, D, K, L, sigma, eta, d_table)

'''
    # B - K
    fig = plt.figure(figsize=(16, 9.7))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlabel(
        r'$\sum_{j\in \mathcal{N}_i^{\rm in}}\bar{\beta}_{ji}$', fontsize=60)
    ax.set_ylabel(r'$k_i$', fontsize=60)
    ax.scatter(B_sum,
               np.diag(K), s=100, alpha=1)
    ax.tick_params(axis='x', labelsize=60)
    ax.tick_params(axis='y', labelsize=60)
    fig.savefig("./images/b_k.pdf", bbox_inches="tight", dpi=300)

    # B - L
    fig = plt.figure(figsize=(16, 9.7))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlabel(
        r'$\sum_{j\in \mathcal{N}_i^{\rm in}}\bar{\beta}_{ji}$', fontsize=60)
    ax.set_ylabel(
        r'$\sum_{j\in \mathcal{N}_i^{\rm in}}l_{ji}$', fontsize=60)
    ax.scatter(B_sum,
               L_sum, s=100, alpha=1)
    ax.tick_params(axis='x', labelsize=60)
    ax.tick_params(axis='y', labelsize=60)
    fig.savefig("./images/b_l.pdf", bbox_inches="tight", dpi=300)

    '''
'''
    # b - sigma
    fig = plt.figure(figsize=(16, 9.7))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlabel(r'$\sum_{j\in \mathcal{N}_i^{\rm in}}\bar{\beta}_{ji}$', fontsize=60)
    ax.set_ylabel(r'$\sigma_i$', fontsize=60)
    ax.scatter(B_sum,
               sigma, s=100, alpha=1)
    ax.tick_params(axis='x', labelsize=60)
    ax.tick_params(axis='y', labelsize=60)
    fig.savefig("./images/b_sigma.pdf", bbox_inches="tight", dpi=300)

    # b - eta
    fig = plt.figure(figsize=(16, 9.7))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlabel(r'$\sum_{j\in \mathcal{N}_i^{\rm in}}\bar{\beta}_{ji}$', fontsize=60)
    ax.set_ylabel(r'$\eta_i$', fontsize=60)
    ax.scatter(B_sum,
               eta, s=100, alpha=1)
    ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=60)
    ax.tick_params(axis='y', labelsize=60)
    fig.savefig("./images/b_eta.pdf", bbox_inches="tight", dpi=300)
    '''
'''
    # eigen - k
    fig = plt.figure(figsize=(16, 9.7))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlabel(r'eigenvector centrality', fontsize=60)
    ax.set_ylabel(r'$k_i$', fontsize=60)
    ax.scatter(eigv_cent,
               np.diag(K), s=100, alpha=1)
    ax.tick_params(axis='x', labelsize=60)
    ax.tick_params(axis='y', labelsize=60)
    fig.savefig("./images/eigen_k.pdf", bbox_inches="tight", dpi=300)

    # sigma - eta
    fig = plt.figure(figsize=(16, 9.7))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlabel(r'$\sigma_i$', fontsize=60)
    ax.set_ylabel(r'$\eta_i$', fontsize=60)
    ax.scatter(sigma,
               eta, s=100, alpha=1)
    ax.tick_params(axis='x', labelsize=60)
    ax.tick_params(axis='y', labelsize=60)
    fig.savefig("./images/sigma_eta.pdf", bbox_inches="tight", dpi=300)
    '''
