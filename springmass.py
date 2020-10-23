import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Pendulum equilibrium spring length (m), spring constant (N.m)
L0 = 1.0
k = 40.0
m = 1.0

def dudt(u, t, L0, k, m):
    """Return the first derivatives of u = x, p"""
    x, p = u
    dxdt = p/m
    dpdt = -k*(x - L0)
    return dxdt, dpdt

# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 20, 0.01
t = np.arange(0, tmax+dt, dt)
# Initial conditions: x, p
u0 = [L0 + 1.0, 0.0]

# Do the numerical integration of the equations of motion
u = odeint(dudt, u0, t, args=(L0, k, m))
# Unpack z and theta as a function of time
x = u[:,0]
p = u[:,1]

# Convert to Cartesian coordinates of the two bob positions.
F = -k*x

# Plotted bob circle radius
r = 0.05

def plot_spring(x0, y0, x, y, L):
    """Plot the spring from (x0,y0) to (x,y) as the projection of a helix."""
    # Spring turn radius, number of turns
    rs, ns = 0.05, 25
    # Number of data points for the helix
    Ns = 1000
    # We don't draw coils all the way to the end of the pendulum:
    # pad a bit from the anchor and from the bob by these number of points
    ipad1, ipad2 = 100, 150
    w = np.linspace(0, L, Ns)
    # Set up the helix along the x-axis ...
    xp = np.zeros(Ns)
    xp[ipad1:-ipad2] = rs * np.sin(2*np.pi * ns * w[ipad1:-ipad2] / L)
    ax.plot(x0 + w, y0 + xp, c='k', lw=2)

def make_plot(i):
    """
    Plot and save an image of the spring pendulum configuration for time
    point i.
    """

    y0 = 0.0
    x0 = 0.0
    plot_spring(x0, y0, x[i], y0, L0)
    # Circles representing the anchor point of rod 1 and the bobs
    c0 = Circle((0, 0), r/2, fc='k', zorder=10)
    c1 = Circle((x[i], 0), r, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)

    ax.set_xlim(x0, 1.2*(L0+np.max(x)))
    ax.set_ylim(y0-1, y0+1)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('frames/img-{:04d}.png'.format(i//di), dpi=300)
    # Clear the Axes ready for the next image.
    plt.cla()


# Make an image every di time points, corresponding to a frame rate of fps
# frames per second.
# Frame rate, s-1
fps = 10
di = int(1/fps/dt)
# This figure size (inches) and dpi give an image of 600x450 pixels.
fig = plt.figure(figsize=(8, 6), dpi=300)
ax = fig.add_subplot(111)

for i in range(0, t.size, di):
    print(i // di, '/', t.size // di)
    make_plot(i)

