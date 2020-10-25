import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import rc
import matplotlib.gridspec as gridspec
import os

rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 20, 0.01
t = np.arange(0, tmax+dt, dt)

# Pendulum equilibrium spring length (m), spring constant (N.m)
L0 = 5.0
k = 5.0
m = 1.0

# Initial conditions: x, p
u0 = [L0 + 2.0, 0.0]

# for the animation
fps = 4
di = int(1/fps/dt)
dpi = 200
# Plotted bob circle radius
r = 0.40

def dudt(u, t, L0, k, m):
    """Return the first derivatives of u = x, p"""
    x, p = u
    dxdt = p/m
    dpdt = -k*(x - L0)
    return dxdt, dpdt

# Do the numerical integration of the equations of motion
u = odeint(dudt, u0, t, args=(L0, k, m))
# Unpack z and theta as a function of time
x = u[:,0]
p = u[:,1]

# Force, kinetic and potential energy
F = -k*(x - L0)
T = p**2/(2.0*m)
V = 0.5*k*(x - L0)**2

def plot_spring(x0, y0, x, y, L, ax):
    """Plot the spring from (x0,y0) to (x,y) as the projection of a helix."""
    # Spring turn radius, number of turns
    rs, ns = 0.10, 20
    # Number of data points for the helix
    Ns = 1000
    # We don't draw coils all the way to the end of the pendulum:
    # pad a bit from the anchor and from the bob by these number of points
    ipad1, ipad2 = 100, 150
    w = np.linspace(x0, x, Ns)
    # Set up the helix along the x-axis ...
    h = np.zeros(Ns)
    h[ipad1:-ipad2] = y0 + rs * np.sin(2*np.pi * ns * w[ipad1:-ipad2] / L)
    ax.plot(w, h, c='k', lw=2)

def make_plot(i, ax=None, axp=None, fname=None):
    """
    Plot and save an image of the spring pendulum configuration for time
    point i.
    """

    y0 = 0.0
    x0 = 0.0
    minX = np.min(x)
    maxX = np.max(x)
    if ax is not None:
        plot_spring(x0, y0, x[i], y0, x[i], ax)
        #ax.arrow(x[i], y0+0.5, 0.2*F[i], y0)
        ax.annotate("", xy=(x[i]+0.2*F[i], y0+0.5), xytext=(x[i], y0+0.5),
                    arrowprops=dict(arrowstyle="-|>", shrinkA=4, shrinkB=4))
        ax.text(0.5*(2*x[i] + 0.2*F[i]), y0+0.8, r"$\vec{F}$", fontsize='xx-large')
        ax.annotate("", xy=(x[i]+0.5*p[i], y0-0.8), xytext=(x[i], y0-0.8),
                    arrowprops=dict(arrowstyle="-|>", shrinkA=4, shrinkB=4))
        ax.text(0.5*(2*x[i] + 0.5*p[i]), y0-1.5, r"$\vec{p}$", fontsize='xx-large')
        # Circles representing the anchor point of rod 1 and the bobs
        c0 = Circle((0, 0), r/2, fc='k', zorder=10)
        c1 = Circle((x[i], 0), r, fc='r', ec='r', zorder=10)
        ax.add_patch(c0)
        ax.add_patch(c1)
        rg = Rectangle((-0.5, -1.0), 0.5, 2.0, color='k')
        ax.add_patch(rg)

        ax.set_aspect(aspect='equal', adjustable='box')
        ax.set_xlim(0.0, maxX*2)
        ax.set_ylim(y0-2, y0+2)
        ax.axis('off')

    if axp is not None:
        minV = np.min(V)
        maxV = np.max(V)+0.1
        xp = np.linspace(-2.0-0.3, 2.0+0.3, 100)
        Vp = 0.5*k*xp**2
        Ep = (T+V)[0]
        Tp = Ep - Vp
        Tp[Tp < 0] = np.nan

        axp.plot(xp, Tp, lw=2, ls='--', color='red', label="Kinetic (T)")
        axp.plot(xp, Vp, lw=2, ls='-', color='blue', label="Potential (V)")
        axp.plot(xp, Ep*np.ones_like(xp), lw=1, ls='-', color='k', label="Total (E)")

        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        axp.spines['left'].set_position('center')
        axp.spines['bottom'].set_position('zero')
        
        # Eliminate upper and right axes
        axp.spines['right'].set_color('none')
        axp.spines['top'].set_color('none')
        
        # Show ticks in the left and lower axes only
        axp.xaxis.set_ticks_position('bottom')
        axp.yaxis.set_ticks_position('left')
        if isinstance(i, int):
            ce = Circle((x[i]-L0, V[i]), 0.05, fc='r', ec='r', zorder=10)
            axp.add_patch(ce)
        #axp.set_xlim(minX-L0, maxX-L0)
        #axp.set_ylim(minV, maxV)
        axp.set_xlabel(r"$x$ [m]", horizontalalignment='right', x=1.0)
        axp.set_ylabel("Energy [J]", horizontalalignment='right', y=1.0)
        #axp.grid()
        axp.legend(loc='center left', frameon=False)
    if fname is None:
        fname = f'frames/img-{i//di:04d}.png'
    plt.savefig(fname, dpi=dpi)
    # Clear the Axes ready for the next image.
    if ax is not None:
        ax.cla()
    if axp is not None:
        axp.cla()
    plt.cla()

def plot(fname, mode, i='all'):
    # Make an image every di time points, corresponding to a frame rate of fps
    # frames per second.
    # Frame rate, s-1
    # This figure size (inches) and dpi give an image of 600x450 pixels.
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    gs = gridspec.GridSpec(4, 1)
    ax = None
    axp = None
    if mode == 1:
        ax = fig.add_subplot(gs[:, :])
    elif mode == 2:
        axp = fig.add_subplot(gs[:, :])
    else:
        ax = fig.add_subplot(gs[:1, :])
        axp = fig.add_subplot(gs[1:, :])
    
    if i == 'all':
        for i in range(0, t.size, di):
            print(i // di, '/', t.size // di)
            make_plot(i, ax=ax, axp=axp)
        os.system(f"convert -delay {fps} -loop 0 frames/img-*.png {fname}")
    else:
        make_plot(i, ax=ax, axp=axp, fname=fname)

plot(fname='springmass_energy.png', mode=2, i='energy')
plot(fname="springmass.gif", mode=3)
