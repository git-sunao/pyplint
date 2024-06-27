"""
Author: Sunao Sugiyama
Date: 2024-06-25
Description: This is a python implementation of the Pichard Lefschetz algorithm

References:
https://p-lpi.github.io/
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.collections import PolyCollection

print('==== NOTE: STILL IN DEVELOPMENT STAGE ====')

class Points:
    dtype = [('ID', int), ('x', complex), ('y', complex), ('active', bool)]
    def __init__(self, *points):
        """
        """
        n = len(points)
        self.data = np.zeros(n, dtype=self.dtype)
        self.data['ID']     = np.arange(n)
        self.data['x']      = np.array([p[0] for p in points], dtype=complex)
        self.data['y']      = np.array([p[1] for p in points], dtype=complex)
        self.data['active'] = np.ones(n, dtype=bool)

    def size(self):
        """
        Get the number of points
        """
        return len(self.data)

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

    def get_y(self, ID=None):
        """
        Get the coordinate of the point
        """
        if ID is None:
            return self.data['y']
        else:
            idx = self.get_idx(ID)
            return self.data['y'][idx]
    
    def get_xy(self, ID=None):
        """
        Get the coordinate of the point
        """
        if ID is None:
            x = self.data['x']
            y = self.data['y']
        else:
            idx = self.get_idx(ID)
            x = self.data['x'][idx]
            y = self.data['y'][idx]
        xy = np.transpose([x,y])
        return xy

    def get_active(self, ID=None):
        """
        Get the active flag
        """
        if ID is None:
            return self.data['active']
        else:
            idx = self.get_idx(ID)
            return self.data['active'][idx]

    def get_idx(self, ID=None):
        """
        Get the index of the point
        """
        if isinstance(ID, int):
            return np.where(self.data['ID'] == ID)[0]
        else:
            return np.searchsorted(self.data['ID'], ID)

    def move_x(self, dx, ID=None):
        """
        Move the point
        """
        if ID is None:
            self.data['x'] += dx
        else:
            idx = self.get_idx(ID)
            self.data['x'][idx] += dx

    def move_y(self, dy, ID=None):
        """
        Move the point
        """
        if ID is None:
            self.data['y'] += dy
        else:
            idx = self.get_idx(ID)
            self.data['y'][idx] += dy

    def move_xy(self, dxdy, ID=None):
        """
        Move the point
        """
        if ID is None:
            self.data['x'] += dxdy[:,0]
            self.data['y'] += dxdy[:,1]
        else:
            idx = self.get_idx(ID)
            self.data['x'][idx] += dxdy[:,0]
            self.data['y'][idx] += dxdy[:,1]

    def append(self, *points):
        """
        Append the points
        """
        n = len(points)
        data = np.zeros(n, dtype=self.dtype)
        data['ID']     = np.max(self.data['ID']) + np.arange(n) + 1
        data['x']      = np.array([p[0] for p in points], dtype=complex)
        data['y']      = np.array([p[1] for p in points], dtype=complex)
        data['active'] = np.ones(n, dtype=bool)
        self.data = np.append(self.data, data)
        newID = data['ID']
        return newID

    def inactivate(self, ID):
        """
        Inactivate a point
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
    dtype    = [('l1', int), ('l2', int), \
                ('l3', int), ('l4', int), \
                ('size12', float), ('size34', float), \
                ('size13', float), ('size24', float), \
                ('active', bool)]
    def __init__(self, l1, l2, l3, l4):
        """
        l1 (int): ID of the first point
        l2 (int): ID of the second point
        l3 (int): ID of the third point
        l4 (int): ID of the fourth point

        l1 -- l2
        |      |
        l3 -- l4
        """
        self.data = np.zeros(1, dtype=self.dtype)
        self.data['l1'] = l1
        self.data['l2'] = l2
        self.data['l3'] = l3
        self.data['l4'] = l4
        self.data['size12'] = 0
        self.data['size34'] = 0
        self.data['size13'] = 0
        self.data['size24'] = 0
        self.data['active'] = True

    def set_size(self, points):
        """
        Set the size of the simplex
        """
        p1 = points.get_xy(self.data['l1'])
        p2 = points.get_xy(self.data['l2'])
        p3 = points.get_xy(self.data['l3'])
        p4 = points.get_xy(self.data['l4'])
        self.data['size12'] = np.sum(np.abs(p2-p1)**2, axis=1)**0.5
        self.data['size34'] = np.sum(np.abs(p4-p3)**2, axis=1)**0.5
        self.data['size13'] = np.sum(np.abs(p3-p1)**2, axis=1)**0.5
        self.data['size24'] = np.sum(np.abs(p4-p2)**2, axis=1)**0.5

    def append(self, l1, l2, l3, l4):
        """
        Append simplices
        """
        n = len(l1)
        data = np.zeros(n, dtype=self.dtype)
        data['l1'] = l1
        data['l2'] = l2
        data['l3'] = l3
        data['l4'] = l4
        data['size12'] = 0
        data['size34'] = 0
        data['size13'] = 0
        data['size24'] = 0
        data['active'] = np.ones(n, dtype=bool)
        self.data = np.append(self.data, data)

    def inactivate(self, points):
        """
        Inactivate the simplices based on the inactive points
        """
        a1 = points.get_active(self.data['l1'])
        a2 = points.get_active(self.data['l2'])
        a3 = points.get_active(self.data['l3'])
        a4 = points.get_active(self.data['l4'])
        a  = a1 & a2 & a3 & a4
        self.data['active'] = a & self.data['active']

    def subdivide(self, points, delta, max_depth=10):
        """
        Subdivide the simplices into smaller simplices
        """
        self.set_size(points)
        # subdivide in the first dimension
        w = (np.max([self.data['size12'], self.data['size34']], axis=0) > delta) & self.data['active']
        n = 0
        while np.any(w) and n < max_depth:
            self.data['active'][w] = False
            l1 = self.data['l1'][w]
            l2 = self.data['l2'][w]
            l3 = self.data['l3'][w]
            l4 = self.data['l4'][w]
            p1 = points.get_xy(l1)
            p2 = points.get_xy(l2)
            p3 = points.get_xy(l3)
            p4 = points.get_xy(l4)
            p12 = (p1+p2)/2
            p34 = (p3+p4)/2
            newID12 = points.append(*p12)
            newID34 = points.append(*p34)
            self.append(l1, newID12, l3, newID34)
            self.append(newID12, l2, newID34, l4)
            self.set_size(points)
            w = (np.max([self.data['size12'], self.data['size34']], axis=0) > delta) & self.data['active']
            n += 1
        # subdivide in the second dimension
        w = (np.max([self.data['size13'], self.data['size24']], axis=0) > delta) & self.data['active']
        n = 0
        while np.any(w) and n < max_depth:
            self.data['active'][w] = False
            l1 = self.data['l1'][w]
            l2 = self.data['l2'][w]
            l3 = self.data['l3'][w]
            l4 = self.data['l4'][w]
            p1 = points.get_xy(l1)
            p2 = points.get_xy(l2)
            p3 = points.get_xy(l3)
            p4 = points.get_xy(l4)
            p13 = (p1+p3)/2
            p24 = (p2+p4)/2
            newID13 = points.append(*p13)
            newID24 = points.append(*p24)
            self.append(l1, l2, newID13, newID24)
            self.append(newID13, newID24, l3, l4)
            self.set_size(points)
            w = (np.max([self.data['size13'], self.data['size24']], axis=0) > delta) & self.data['active']
            n += 1
        
    def clean(self, points):
        """
        Remove the inactive simplices
        """
        w = self.data['active']
        self.data = self.data[w]

    def copy(self):
        """
        Copy the simplices
        """
        simplices = Simplices(0,1,2,3)
        simplices.data = self.data.copy()
        return simplices

class Thimble:
    def __init__(self, xmin, xmax, delta=0.1, tau=0.02, thre=-20, epsilon=1e-6, niter=20, max_len=2e3):
        """
        """
        # set the parameters
        self.xmin = xmin
        self.xmax = xmax
        self.delta = delta
        self.tau = tau
        self.thre = thre
        self.epsilon = epsilon
        self.niter = niter
        self.max_len = max_len
        # initialize the points and simplices
        self.reset()

    def reset(self):
        """
        Reset the points and simplices
        """
        # initialize the points
        self.points = Points([self.xmin, self.xmax], \
                             [self.xmax, self.xmax], \
                             [self.xmin, self.xmin], \
                             [self.xmax, self.xmin])
        # initialize the simplices
        self.simplices = Simplices(0, 1, 2, 3)

    def subdivide(self):
        """
        Subdivide the simplices
        """
        self.simplices.subdivide(self.points, self.delta)

    def clean(self):
        """
        Remove the inactive points and simplices
        """
        self.simplices.clean(self.points)
        self.points.clean()

    def copy(self):
        """
        Copy the thimble
        """
        thimble = Thimble(self.xmin, self.xmax, self.delta, self.tau, self.thre, self.epsilon, self.niter, self.max_len)
        thimble.points = self.points.copy()
        thimble.simplices = self.simplices.copy()
        return thimble
        
    def set_phi(self, phi, **kwargs):
        """
        Set the function phi
        """
        self.phi_func   = phi
        self.phi_kwargs = kwargs

    def integrand(self, x, y):
        """
        Compute the integrand of the Picard-Lefschetz formula
        """
        phi = self.phi_func(x, y, **self.phi_kwargs)
        return np.exp(phi)

    def h(self, x, y):
        """
        Compute the function h
        """
        phi = self.phi_func(x, y, **self.phi_kwargs)
        return np.real(phi)

    def H(self, x, y):
        """
        Compute the function H
        """
        phi = self.phi_func(x, y, **self.phi_kwargs)
        return np.imag(phi)

    def gradient(self, x, y):
        """
        Compute the gradient of the function phi
        """
        # For dim1
        gr = (self.h(x+   self.epsilon, y) - self.h(x-   self.epsilon, y))/(2*self.epsilon)
        gi = (self.h(x+1j*self.epsilon, y) - self.h(x-1j*self.epsilon, y))/(2*self.epsilon)
        gx = gr + 1j*gi
        # For dim2
        gr = (self.h(x, y+   self.epsilon) - self.h(x, y-   self.epsilon))/(2*self.epsilon)
        gi = (self.h(x, y+1j*self.epsilon) - self.h(x, y-1j*self.epsilon))/(2*self.epsilon)
        gy = gr + 1j*gi
        # stack
        g = np.transpose([gx, gy])
        return g

    def flow(self):
        # move points on flow
        ID = self.points.get_ID()
        a  = self.points.get_active()
        x  = self.points.get_x()
        y  = self.points.get_y()
        g  = self.gradient(x[a], y[a])
        n  = (np.abs(g[:,0])**2 + np.abs(g[:,1])**2)**0.5
        n  = np.transpose([n, n])
        dxdy = - self.tau * np.divide(g, n, where=n>0, out=g)
        self.points.move_xy(dxdy, ID[a])
    
    def inactivate(self):
        """
        Inactivate the points and simplices
        """
        ID = self.points.get_ID()
        a  = self.points.get_active()
        x  = self.points.get_x()
        y  = self.points.get_y()
        h  = self.h(x[a], y[a])
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
    
    def _plot(self, dim, fig_and_ax=None, return_artists=False, down_sample=0, xmin=None, xmax=None):
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
        ax.set_xlabel(r'$\mathcal{{R}}[{}]$'.format(['x', 'y'][dim]))
        ax.set_ylabel(r'$\mathcal{{I}}[{}]$'.format(['x', 'y'][dim]))
        p1 = self.points.get_xy(self.simplices.data['l1'])[:,dim]
        p2 = self.points.get_xy(self.simplices.data['l2'])[:,dim]
        p3 = self.points.get_xy(self.simplices.data['l3'])[:,dim]
        p4 = self.points.get_xy(self.simplices.data['l4'])[:,dim]
        if down_sample > 0 and len(p1) > down_sample:
            c = np.random.choice(len(p1), down_sample, replace=False)
            p1 = p1[c]
            p2 = p2[c]
            p3 = p3[c]
            p4 = p4[c]
        polygons = np.transpose([[p1.real, p2.real, p3.real, p4.real, p1.real], \
            [p1.imag, p2.imag, p3.imag, p4.imag, p1.imag]])
        poly_collection = PolyCollection(polygons, alpha=0.3)
        ax.add_collection(poly_collection)
        ax.set_xlim(xmin or self.xmin, xmax or self.xmax)
        ax.set_ylim(xmin or self.xmin, xmax or self.xmax)
        ax.grid()
        if return_artists:
            return fig, ax, [poly_collection]
        else:
            return fig, ax

    def plot(self, fig_and_axes=None, return_artists=False, down_sample=0, xmin=None, xmax=None):
        """
        Plot the simplices
        """
        if fig_and_axes is None:
            fig, axes = plt.subplots(1,2, figsize=(10,5))
        else:
            fig, axes = fig_and_axes
        fig.subplots_adjust(wspace=0.3)
        artists = []
        for dim, ax in enumerate(axes):
            o = self._plot(dim, (fig, ax), return_artists=return_artists, down_sample=down_sample, xmin=xmin, xmax=xmax)
            if return_artists:
                artists += o[2]
        if return_artists:
            return fig, axes, artists
        else:
            return fig, axes

    def plot_gif(self, fname, niter=None, time=3, **kwargs):
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
        fig, axes = plt.subplots(1,2,figsize=(10,5))

        def update(frame):
            # update the data
            self.routine()
            # update the plot
            axes[0].clear()
            axes[1].clear()
            arts = self.plot((fig, axes), return_artists=True, **kwargs)[2]
            return arts
        
        niter = niter or self.niter
        self.reset()
        ani = animation.FuncAnimation(fig, update, \
            frames=np.arange(0, niter), blit=True)

        ani.save(fname, writer='pillow', fps=niter/time)
        plt.close()

    def romberg(self, N=3, integrand=None):
        if integrand is None:
            integrand = self.integrand
        p1 = self.points.get_xy(self.simplices.data['l1'])
        p2 = self.points.get_xy(self.simplices.data['l2'])
        p3 = self.points.get_xy(self.simplices.data['l3'])
        p4 = self.points.get_xy(self.simplices.data['l4'])
        I  = romberg(integrand, [p1, p2, p3, p4], N=N).sum()
        return I

def _normalized_integrand(u, v, pnts, func):
    # compute jacobian
    hess00 = (pnts[3][:,0] - pnts[1][:,0])*(1-v) + (pnts[2][:,0] - pnts[0][:,0])*v
    hess01 = (pnts[3][:,1] - pnts[1][:,1])*(1-v) + (pnts[2][:,1] - pnts[0][:,1])*v
    hess10 = (pnts[0][:,0] - pnts[1][:,0])*(1-u) + (pnts[2][:,0] - pnts[3][:,0])*u
    hess11 = (pnts[0][:,1] - pnts[1][:,1])*(1-u) + (pnts[2][:,1] - pnts[3][:,1])*u
    jacob = hess00*hess11 - hess01*hess10
    # move to the original simplex domain
    xy= pnts[0] * (1-u)*v + \
        pnts[1] * (1-u)*(1-v) + \
        pnts[2] * u*v + \
        pnts[3] * u*(1-v)
    # compute the integrand at x(u,v)
    i = func(xy[:,0], xy[:,1])
    return i*jacob

def romberg(func, pnts, tol=1e-10, N=10):
    n = pnts[0].shape[0]
    
    h = np.zeros(N + 1)
    r = np.zeros((n, N + 1, N + 1), dtype=complex)

    for i in range(1, N + 1):
        h[i] = 1. / (2 ** (i - 1))

    r[:, 1, 1] = h[1] ** 2 * (
        _normalized_integrand(0, 0, pnts, func) + 
        _normalized_integrand(h[1], 0, pnts, func) +
        _normalized_integrand(0, h[1], pnts, func) +
        _normalized_integrand(h[1], h[1], pnts, func)
    ) / 4.0

    for i in range(2, N + 1):
        coeff = 0
        for k in range(1, int(2 ** (i - 2)) + 1):
            coeff += 2 * (
                _normalized_integrand((2 * k - 1) * h[i], 0, pnts, func) +
                _normalized_integrand((2 * k - 1) * h[i], h[1], pnts, func) +
                _normalized_integrand(0, (2 * k - 1) * h[i], pnts, func) +
                _normalized_integrand(h[1], (2 * k - 1) * h[i], pnts, func)
            )
        for k in range(1, int(2 ** (i - 2)) + 1):
            for l in range(1, int(2 ** (i - 1))):
                coeff += 4 * _normalized_integrand(2 * h[i] * k, (2 * l - 1) * h[i], pnts, func)
        for k in range(1, int(2 ** (i - 2))):
            for l in range(1, int(2 ** (i - 2)) + 1):
                coeff += 4 * _normalized_integrand((2 * k - 1) * h[i], 2 * h[i] * l, pnts, func)

        r[:, i, 1] = 0.25 * (r[:,i - 1, 1] + h[i] ** 2 * coeff)

    for i in range(2, N + 1):
        for j in range(2, i + 1):
            r[:, i, j] = r[:, i, j - 1] + (r[:, i, j - 1] - r[:, i - 1, j - 1]) / (4 ** (j - 1) - 1)

    return r[:, N, N]
