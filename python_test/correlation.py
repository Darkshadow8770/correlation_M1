#
"""Gaussian Correlation
play with parameters:
 - Number of coordinates
 - lx (the decorrelation lenght scale)
 - multiply by the correlation instead of the covariance and plot
 - add your own test
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import math

def se_corr(x, lx):
    """Square exponential correlation kernel or Gaussian kernel in 1D
    x -- vector of coordinates
    lx -- decorrelation lenght scale
    In this example, the decorrelation lenght scale is constant.
    However, in the general case, the decorrelation lenght scale is not constant. It changes from one point to another.
    """
    n = len(x)
    C = np.zeros((n,n,))
    two_lx2 = 2.0*lx*lx
    for i in range(n):
        for j in range(n):
            dist = x[i] - x[j]
            C[i,j] = np.exp( -dist**2/two_lx2 )
            pass
        pass
    return C

def make_simple_covariance(C):
    """Make a simple covariance by normalizing a correlation matrix.
    this builds a pseudo covariance that can be used for Gaussian denoising
    """
    n = C.shape[0]
    W = np.zeros_like(C)
    for k in range(n):
        W[k,k] = 1./np.sqrt(sum(C[k,:]))
        pass
    B = np.dot( W, np.dot( C, W ) )
    return B


#
B_std = 1.0
R_std = 1.0
#
nCoord = 200
coordMin = 0.
coordMax = 1.
coord = np.linspace(coordMin, coordMax, num=nCoord)
dCoord = coord[1]-coord[0]
lx = 10.0*dCoord
#
C = se_corr(coord, lx)
B = make_simple_covariance(C)
#
#rnd = np.array( [random.random() for _ in coord] )
# generating Gaussian noise of mean 0 and standard deviation 0.2
rnd = np.array( [random.gauss(0., 0.2) for _ in coord] )
# making a Sine wave
sinx = np.array([ math.sin( 2.*math.pi*(x-0.5) ) for x in coord] )
noisy = sinx + rnd
# Denoising
denoised = np.dot(B, noisy)
#

fig, axs = plt.subplots(1, 2)
ax = axs[0]
ax.plot(coord, noisy, "r--", lw = 2.0, label="Noisy")
ax.plot(coord, sinx, "g-", lw = 4.0, label="Truth")
ax.plot(coord, denoised, "k-", lw = 2.0, label="denoised")
ax.legend( loc=3 )
#plt.ylim( 0., 1.2 )
ax.set_title('Random and correlated')


ax = axs[1]
cplot = ax.contourf(coord, coord, C)
#ax.axis('equal')
ax.set_aspect('equal', 'box')
ax.set_xlabel('X')
ax.set_ylabel('X')
ax.set_title('SE correlation')
fig.colorbar(cplot, shrink=0.5, aspect=5)


plt.show()
plt.close()

exit()
