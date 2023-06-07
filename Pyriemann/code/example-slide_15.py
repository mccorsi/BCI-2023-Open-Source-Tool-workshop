
import numpy as np
from pyriemann.datasets import sample_gaussian_spd
from pyriemann.utils.mean import mean_riemann
import matplotlib.pyplot as plt

# fix seed
np.random.seed(100)

# generate a set of random SPD matrices
C = sample_gaussian_spd(n_matrices=10, mean=np.eye(2), sigma=3)
M = mean_riemann(C)

theta = np.linspace(0, 2*np.pi, 50)
r = np.linspace(0, 8, 20)
X = []
for ri in r:
    for tj in theta:
        Q = np.array([[np.cos(tj), -np.sin(tj)], [np.sin(tj), np.cos(tj)]])
        D = np.diag([ri, 0])
        Xij = Q @ D @ Q.T
        X.append(Xij)
X = np.stack(X)        

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for Ci in C:
    ax.scatter(Ci[0,0], Ci[1,1], Ci[0,1], c='blue', s=50)
for Xi in X:
    ax.scatter(Xi[0,0], Xi[1,1], Xi[0,1], c='black', s=2, alpha=0.50)
ax.scatter(M[0,0], M[1,1], M[0,1], c='red', s=50)
ax.view_init(elev=-50., azim=23, roll=0)
ax.set_xlim(1.4, 5)
ax.set_ylim(0.5, 4)
ax.set_zlim(-3.4, +0.2)
plt.savefig('example-slide_15.pdf', format='pdf')