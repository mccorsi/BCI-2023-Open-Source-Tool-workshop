
import numpy as np
from pyriemann.datasets import generate_random_spd_matrix
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.distance import distance_riemann
import matplotlib.pyplot as plt

# fix seed
np.random.seed(100)

# generate two random SPD matrices
A = generate_random_spd_matrix(n_dim=2)
B = generate_random_spd_matrix(n_dim=2)

# walk over the geodesic between A and B
C = []
for gamma in np.linspace(0, 1, 100):
    C.append(geodesic_riemann(A, B, gamma))

# calculate the distances
d = [distance_riemann(A, Ci) for Ci in C]

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
ax.scatter(A[0,0], A[1,1], A[0,1], c='blue', s=50)
ax.scatter(B[0,0], B[1,1], B[0,1], c='blue', s=50)
for Ci in C:
    ax.scatter(Ci[0,0], Ci[1,1], Ci[0,1], c='red', s=10)
for Xi in X:
    ax.scatter(Xi[0,0], Xi[1,1], Xi[0,1], c='black', s=2, alpha=0.50)
ax.view_init(elev=-50., azim=23, roll=0)
ax.set_xlim(1.4, 5)
ax.set_ylim(0.5, 4)
ax.set_zlim(-3.4, +0.2)
plt.savefig('example-slide_14.pdf', format='pdf')
