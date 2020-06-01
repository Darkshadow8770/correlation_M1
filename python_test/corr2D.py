#
"""Application of the 2D square exponential correlation kernel
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import math
#
def apply_se_corr2D(x, y, lx, f):
    """Apply the 2D square exponential correlation kernel 
    x -- 2d array of coordinates in the first dimension
    y -- 2d array of coordinates in the second dimension
    lx -- constant decorrelation lenght scale
    f -- 2D array discretized function to apply the correlation to
    In this example, the decorrelation lenght scale is constant.
    However, in the general case, the decorrelation lenght scale is not constant. It changes from one point to another.
    """
    if( x.shape != y.shape or x.shape != f.shape):
        print (x.shape)
        print (y.shape)
        print (f.shape)
        raise ValueError("Incompatible shape of coordinates arrays")
    #
    nx = x.shape[0]
    ny = x.shape[1]
    #
    g = np.zeros_like(f)
    two_lx2 = 2.0*lx*lx
    #
    for i in range(nx):
        for j in range(ny):
            coef_sum = 0
            for k1 in range(nx):
                for k2 in range(ny):
                    dx = x[k1,k2] - x[i,j]
                    dy = y[k1,k2] - y[i,j]
                    coef = np.exp( -(dx*dx+dy*dy)/two_lx2 )
                    g[i,j] = g[i,j] + coef*f[k1,k2]
                    coef_sum = coef_sum + coef
                    pass # k2
                pass #k1
            # Normalization if needed
            g[i,j] = g[i,j]/coef_sum
            pass # j
        pass # i
    return g
#
nx = 20
ny = 40
xmin, xmax = 0., 1.0
ymin, ymax = 0., 2.0
x1d = np.linspace(xmin, xmax, num=nx)
y1d = np.linspace(ymin, ymax, num=ny)
#
xmean, ymean = (xmin+xmax)/2., (ymin+ymax)/2.
#
dx = x1d[1]-x1d[0]
dy = y1d[1]-y1d[0]
lx = 2.0*np.sqrt(dx*dx + dy*dy)
# meshgrid uses 'xy' indexing by default, so w
x2d, y2d = np.meshgrid(x1d, y1d, indexing='ij')
#
print lx
#
# generating Gaussian noise of mean 0 and standard deviation lx
rnd = np.random.normal(0., .1, (nx,ny))
# making a Sine wave
sinx = np.zeros_like( rnd )

for i in range(nx):
    for j in range(ny):
        coordij = np.sqrt( (x2d[i,j]-xmean)**2 + (y2d[i,j]-ymean)**2 )
        sinx[i,j] = math.sin( 2.*math.pi*coordij )
        pass # j
    pass #i
noisy = sinx + rnd
# Denoising
denoised = apply_se_corr2D(x2d, y2d, lx, noisy)
#
vmin=min(np.amin(sinx), np.amin(noisy))
vmax=max(np.amax(sinx), np.amax(noisy))
levels=np.linspace(vmin, vmax, num=10)

fig, axs = plt.subplots(1, 3)
ax = axs[0]
cplot1 = ax.contourf( x2d, y2d, sinx, levels )
ax.set_aspect('equal', 'box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Truth')
#fig.colorbar(cplot, shrink=0.5, aspect=5)
#
ax = axs[1]
cplot = ax.contourf( x2d, y2d, noisy, levels )
ax.set_aspect('equal', 'box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Noisy')
#fig.colorbar(cplot, shrink=0.5, aspect=5)
#
ax = axs[2]
cplot = ax.contourf( x2d, y2d, denoised, levels )
ax.set_aspect('equal', 'box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('denoised')
#fig.colorbar(cplot, shrink=0.5, aspect=5)
fig.colorbar(cplot, ax=axs)
#
plt.show()
plt.close()

exit()
