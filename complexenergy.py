import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
import os

rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern']})
rc('text', usetex=True)

dpi = 300

r = np.linspace(0.8, 6, 1000)
k = 1.0
L = 1.0
m = 0.5
V = -k/r + 0.5*L**2/(m*r**2)
minV = np.min(V)
maxV = np.max(V)
r0 = r[V.argmin(axis=0)]
#Vp = k/r**2 - L**2/(m*r**3)
Vpp = -2*k/r0**3 + 3*L**2/(m*r0**4)
Vapprox = minV + 0.5*Vpp*(r-r0)**2

def make_plot(fname=None):
    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    plt.plot(r, V, lw=2, ls='-', color='k', label=r"$V(r) = -\frac{k}{r} + \frac{1}{2}\frac{L^2}{m r^2}$")
    plt.plot(r, Vapprox, lw=3, ls='-', color='r', label=r"$V_{approx}(r) = V(r_0) + \frac{1}{2} V''(r_0) (r - r_0)^2$")

    ax = plt.gca()
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.xlim(0.0, r[-1])
    plt.ylim(minV-0.01, maxV)
    plt.xlabel(r"$r$", horizontalalignment='right', x=1.0)
    plt.ylabel("Potential energy", horizontalalignment='right', y=1.0)
    #plt.grid()
    plt.legend(frameon=False)
    plt.savefig(fname, dpi=dpi)

make_plot("complexenergy.png")
