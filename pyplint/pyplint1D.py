"""
Author: Sunao Sugiyama
Date: 2024-06-25
Description: This is a python implementation of the Pichard Lefschetz algorithm

References:
https://p-lpi.github.io/
"""
import numpy as np
import matplotlib.pyplot as plt

class Points:
    dtype = [('ID', int), ('x', complex), ('active', bool)]
    def __init__(self, *points):
        """
        """
        # initialize points
        n = len(points)
        self.data = np.zeros(n, dtype=self.dtype)
        self.data['ID']     = np.arange(n)
        self.data['x']      = np.array(points, dtype=complex)
        self.data['active'] = np.ones(n, dtype=bool)

    def size(self):
        """
        Get the number of points
        """
        return data.size

    def get_ID(self):
        """
        Get the ID of the point
        """
        return self.data['ID']

    def get_x(self, ID=None):
        """
        Get the coordinate of the point
        """
        if ID is None:
            return self.data['x']
        else:
            idx = self.get_idx(ID)
            return self.data['x'][idx]

    def get_active(self, ID=None):
        """
        Get the activity of the point
        """
        if ID is None:
            return self.data['active']
        else:
            idx = self.get_idx(ID)
            return self.data['active'][idx]

    def get_idx(self, ID):
        """
        Get the index of the point
        """
        if isinstance(ID, int):
            return self.data['ID'].tolist().index(ID)
        else:
            return np.searchsorted(self.data['ID'], ID)

    def move_x(self, dx, ID=None):
        """
        Move the point
        """
        if ID is None:
            self.x += dx
        else:
            idx = self.get_idx(ID)
            self.data['x'][idx] += dx

    def append(self, *points):
        """
        Append the points
        """
        n = len(points)
        data = np.zeros(n, dtype=self.dtype)
        data['ID']     = np.max(self.data['ID']) + np.arange(n) + 1
        data['x']      = np.array(points, dtype=complex)
        data['active'] = np.ones(n, dtype=bool)
        self.data = np.append(self.data, data)
        newID = data['ID']
        return newID

    def inactivate(self, ID):
        """
        Inactivate the point
        """
        idx = self.get_idx(ID)
        self.data['active'][idx] = False

    def clean(self):
        """
        Remove the inactive points
        """
        idx = self.data['active']
        self.data = self.data[idx]

    def copy(self):
        """
        Copy the points
        """
        points = Points()
        points.data = self.data.copy()
        return points

class Simplices:
    dtype = [('l1', int), ('l2', int), ('size', float), ('active', bool)]
    def __init__(self, l1, l2):
        """
        l1 (int): first point id
        l2 (int): second point id
        """
        self.data = np.zeros(1, dtype=self.dtype)
        self.data['l1'] = l1
        self.data['l2'] = l2
        self.data['size'] = 0
        self.data['active'] = True

    def set_size(self, points):
        """
        Set the size of the simplices
        """
        x1 = points.get_x(self.data['l1'])
        x2 = points.get_x(self.data['l2'])
        dx = x1 - x2
        s  = np.abs(dx)
        self.data['size'] = s

    def append(self, l1, l2):
        """
        Append the simplices
        """
        n = len(l1)
        data = np.zeros(n, dtype=self.dtype)
        data['l1'] = l1
        data['l2'] = l2
        data['size']= 0
        data['active'] = True
        self.data = np.append(self.data, data)

    def inactivate(self, points):
        """
        Inactivate the simplices based on the inactive points
        """
        a1 = points.get_active(self.data['l1'])
        a2 = points.get_active(self.data['l2'])
        a  = a1 & a2
        self.data['active'] = a & self.data['active']

    def subdivide(self, points, delta, max_depth=10):
        """
        Subdivide the simplices into smaller simplices
        """
        self.set_size(points)
        w = (self.data['size']>delta) & self.data['active']
        n = 0
        while np.any(w) and n < max_depth:
            self.data['active'][w] = False
            l1 = self.data['l1'][w]
            l2 = self.data['l2'][w]
            x1 = points.get_x(l1)
            x2 = points.get_x(l2)
            x  = (x1 + x2) / 2
            newID = points.append(*x)
            self.append(l1, newID)
            self.append(newID, l2)
            self.set_size(points)
            w = (self.data['size']>delta) & self.data['active']
            n+= 1

    def clean(self, points):
        """
        Remove the inactive simplices
        """
        # remove inactive simplices
        w = self.data['active']
        self.data = self.data[w]

    def copy(self):
        """
        Copy the simplices
        """
        simplices = Simplices(0, 1)
        simplices.data = self.data.copy()
        return simplices

class Thimble:
    def __init__(self, xmin, xmax, delta=0.1, tau=0.05, thre=-20, epsilon=1e-6, niter=30):
        """
        """
        # set the parameters
        self.xmin = xmin
        self.xmax = xmax
        self.delta = delta
        self.tau = tau
        self.thre= thre
        self.epsilon = epsilon
        self.niter = niter
        # initialize the points and simplices
        self.reset()

    def reset(self):
        self.points    = Points(self.xmin, self.xmax)
        self.simplices = Simplices(0, 1)

    def subdivide(self):
        """
        Subdivide the simplices into smaller simplices
        """
        self.simplices.subdivide(self.points, self.delta)

    def clean(self):
        """
        Remove the inactive simplices and points
        """
        self.simplices.clean(self.points)
        self.points.clean()

    def copy(self):
        """
        Copy the thimble
        """
        thimble = Thimble(self.xmin, self.xmax, self.delta, self.tau, self.thre, self.epsilon, self.niter)
        thimble.points    = self.points.copy()
        thimble.simplices = self.simplices.copy()
        return thimble

    def set_phi(self, phi, **kwargs):
        """
        Set the function phi
        """
        self.phi_func  = phi
        self.phi_kwargs= kwargs

    def integrand(self, x):
        """
        Compute the integrand of the Pichard Lefschetz algorithm at x.
        """
        phi = self.phi_func(x, **self.phi_kwargs)
        return np.exp(phi)

    def h(self, x):
        """
        """
        phi = self.phi_func(x, **self.phi_kwargs)
        return np.real(phi)

    def H(self, x):
        """
        """
        phi = self.phi_func(x, **self.phi_kwargs)
        return np.imag(phi)

    def gradient(self, x):
        """
        """
        # gradiant along real axis
        gr = (self.h(x + self.epsilon) - self.h(x - self.epsilon)) / (2*self.epsilon)
        # gradiant along imag axis
        gi = (self.h(x + 1j*self.epsilon) - self.h(x - 1j*self.epsilon)) / (2*self.epsilon)
        return gr + 1j*gi

    def flow(self):
        # move points on flow
        ID = self.points.get_ID()
        a  = self.points.get_active()
        x  = self.points.get_x()
        g  = self.gradient(x[a])
        dx =-self.tau * np.divide(g, np.abs(g), where=np.abs(g)>0, out=g)
        self.points.move_x(dx, ID[a])

    def inactivate(self):
        """
        Inactivate the simplices and points
        """
        ID = self.points.get_ID()
        a  = self.points.get_active()
        x  = self.points.get_x()
        h  = self.h(x[a])
        self.points.inactivate(ID[a][h < self.thre])
        self.simplices.inactivate(self.points)

    def routine(self):
        self.inactivate()
        self.flow()
        self.subdivide()
        self.clean()

    def find(self, niter=None):
        for _ in range(niter or self.niter):
            self.routine()

    def plot(self, fig_and_ax=None, return_artists=False):
        """
        Plot the simplices
        """
        if fig_and_ax is None:
            fig, ax = plt.subplots(1,1, figsize=(5,5))
        else:
            fig, ax = fig_and_ax
        ax.axis('equal')
        ax.axhline(0, color='k', ls='--')
        ax.axvline(0, color='k', ls='--')
        ax.set_xlabel(r'$\mathcal{R}[x]$')
        ax.set_ylabel(r'$\mathcal{I}[x]$')
        x1 = self.points.get_x(self.simplices.data['l1'])
        x2 = self.points.get_x(self.simplices.data['l2'])
        artists = ax.plot([x1.real, x2.real], [x1.imag, x2.imag], 'C0-', marker='o', ms=2)
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.xmin, self.xmax)
        ax.grid()
        if return_artists:
            return fig, ax, artists
        else:
            return fig, ax

    def plot_gif(self, fname, xyrange=None, niter=None, time=3):
        """
        Plot thimble as a function of iteration by GIF

        Parameters:
        - fname: the filename of the GIF.
        - xyrange: the range of the x and y axis.
        - niter: the number of iterations.
        - time: the time for the GIF.
        """
        import matplotlib.animation as animation
        from PIL import Image
        fig, ax = plt.subplots(figsize=(5,5))

        def update(frame):
            # update the data
            self.routine()
            # update the plot
            ax.clear()
            art = self.plot((fig,ax), return_artists=True)[2]
            return art

        niter = niter or self.niter
        self.reset()
        ani = animation.FuncAnimation(fig, update, \
            frames=np.arange(0, niter), blit=True)

        ani.save(fname, writer='pillow', fps=niter/time)
        plt.close()

    def romberg(self, N=3, integrand=None):
        if integrand is None:
            integrand = self.integrand
        x1 = self.points.get_x(self.simplices.data['l1'])
        x2 = self.points.get_x(self.simplices.data['l2'])
        I  = romberg(integrand, x1, x2, max_iter=N).sum()
        return I

def romberg(func, xmin, xmax, tol=1e-10, max_iter=10):
    """
    Perform Romberg integration to approximate the integral of `func` from `xmin` to `xmax`.
    
    Parameters:
    - func: the integrand function, must be callable.
    - xmin: the lower limit of integration.
    - xmax: the upper limit of integration.
    - tol: the tolerance for stopping the algorithm.
    - max_iter: the maximum number of iterations.
    
    Returns:
    - The estimated value of the integral.
    """
    if xmin.ndim == 0:
        n = 1
    else:
        n = xmin.shape[0]
    
    R = np.zeros((n, max_iter, max_iter), dtype=complex)
    h = xmax - xmin
    
    # First estimate with the trapezoidal rule
    R[:, 0, 0] = 0.5 * h * (func(xmin) + func(xmax))
    
    for i in range(1, max_iter):
        h /= 2
        
        # Composite trapezoidal rule for 2^i panels
        sum_f = np.sum([func(xmin + (2*k - 1) * h) for k in range(1, 2**(i-1) + 1)], axis=0)
        R[:, i, 0] = 0.5 * R[:, i-1, 0] + sum_f * h
        
        # Romberg extrapolation
        for k in range(1, i+1):
            R[:, i, k] = (4**k * R[:, i, k-1] - R[:, i-1, k-1]) / (4**k - 1)
        
        # Check for convergence
        if np.all(np.abs(R[:, i, i] - R[:, i-1, i-1])) < tol:
            return R[:, i, i]

    return R[:, max_iter-1, max_iter-1]