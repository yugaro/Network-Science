import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc
np.random.seed(0)
rc('text', usetex=True)
rc('font', **{'family': "sans-serif"})
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

# number of nodes (max: 50)
n = 50

# matrix of infection rates (air route matrix in case of the passengers and flights)
B = pd.read_csv('./data/US_Airport_Ad_Matrix.csv',
                index_col=0, nrows=n, usecols=[i for i in range(n + 1)]).values

# matrix of recovery rates
D_base_max = 4
D_base_min = 2
D = np.diag((D_base_max - D_base_min) * np.random.rand(n) + D_base_min)

# number of control objectives
M = 1

# define target nodes
W = np.ones((M, n))

# define threshold of each node in the target
d_table = np.array([0.05])

# load parameters of the event-triggered controller
K = np.load('./data/matrix/K.npy')
L = np.load('./data/matrix/L.npy')
sigma = np.load('./data/matrix/sigma.npy')
eta = np.load('./data/matrix/eta.npy')
