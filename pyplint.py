"""
Author: Sunao Sugiyama
Date: 2020-10-06
Description: This is a python implementation of the Pichard Lefschetz algorithm

References:
https://p-lpi.github.io/
"""
import numpy as np
import matplotlib.pyplot as plt

class Thimble:
    """
    A class for the Pichard Lefschetz algorithm.

    Parameters:
    - xmin: the lower limit of the domain.
    - xmax: the upper limit of the domain.
    - delta: the minimum separation of x points.
    - thre: the threshold for h.
    - tau: the step size of the flow.
    - epsilon: the epsilon for gradient computation.
    """
    def __init__(self, xmin, xmax, delta, thre=30, tau=0.05, epsilon=1e-6):
        self.xmin = xmin
        self.xmax = xmax

        # minimum separations of x points
        self.x_delta = delta 
        # threshold for h
        self.h_thre = thre
        # epsilon of gradient
        self.epsilon = epsilon
        # step size of flow
        self.tau = tau

        self.initialize_points()

    def set_phi(self, phi_func, **kwargs):
        """
        Set the function phi that defines the thimble.

        Parameters:
        - phi_func: the function that defines the thimble.
        - kwargs: the keyword arguments for the function.
        """
        self._phi_func = phi_func
        self._phi_kwargs = kwargs

    def initialize_points(self):
        """
        Initialize the points in the domain.
        """
        N = int((self.xmax-self.xmin)/self.x_delta)
        self.x = np.linspace(self.xmin, self.xmax, N) + 1j*np.zeros(N)
        self.active = np.ones(self.x.size, dtype=bool)

    def plot(self, xyrange=None):
        """
        Plot the points in the domain.

        Parameters:
        - xyrange: the range of the x and y axis.
        """
        plt.figure(figsize=(5,5))
        plt.axhline(0, color='k', ls='--')
        plt.axvline(0, color='k', ls='--')
        dx = 0.1*(self.xmax-self.xmin)
        x = self.x[self.active]
        plt.plot(x.real, x.imag, label='active', marker='o', ms=2)
        plt.xlim(xyrange)
        plt.ylim(xyrange)
        plt.legend()

    def plot_gif(self, fname, xyrange=None, niter=50, time=3):
        """
        Plot thimble as a function of iteration by GIF

        Parameters:
        - fname: the filename of the GIF.
        - xyrange: the range of the x and y axis.
        - niter: the number of iterations.
        - time: the time for the GIF.
        """
        self.initialize_points()

        import matplotlib.animation as animation
        from PIL import Image
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlim(xyrange)
        ax.set_ylim(xyrange)
        ax.axhline(0, color='k', ls='--')
        ax.axvline(0, color='k', ls='--')
        ax.set_xlabel(r'$\mathcal{R}[x]$')
        ax.set_ylabel(r'$\mathcal{I}[x]$')
        line, = ax.plot([], [], 'o-', ms=2)

        def update(frame):
            self.flow()
            self.subdivide()
            self.clean()
            x = self.x[self.active]
            line.set_data(x.real, x.imag)
            return line,
        
        ani = animation.FuncAnimation(fig, update, \
            frames=np.arange(0, niter), blit=True)

        ani.save(fname, writer='pillow', fps=niter/time)
        plt.close()

    def phi(self, x):
        """
        Compute the function phi at x.

        Parameters:
        - x: the point at which to evaluate phi.

        Returns:
        - The value of phi at x.
        """
        return self._phi_func(x, **self._phi_kwargs)

    def h(self, x):
        """
        Compute the function h at x.
        
        Parameters:
        - x: the point at which to evaluate h.

        Returns:
        - The value of h at x.
        """
        return np.real(self._phi_func(x, **self._phi_kwargs))

    def H(self, x):
        """
        Compute the function H at x.

        Parameters:
        - x: the point at which to evaluate H.

        Returns:
        - The value of H at x.
        """
        return np.imag(self._phi_func(x, **self._phi_kwargs))

    def gradient(self, x):
        """
        Compute the gradient of h at x.

        Parameters:
        - x: the point at which to evaluate the gradient.

        Returns:
        - The gradient of h at x.
        """
        # gradiant along real axis
        gr = (self.h(x + self.epsilon) - self.h(x - self.epsilon)) / (2*self.epsilon)
        # gradiant along imag axis
        gi = (self.h(x + 1j*self.epsilon) - self.h(x - 1j*self.epsilon)) / (2*self.epsilon)
        return gr + 1j*gi

    def run(self, niter=30):
        """
        Run the Pichard Lefschetz algorithm.

        Parameters:
        - niter: the number of iterations to run.
        """
        self.initialize_points()
        for i in range(niter):
            self.flow()
            self.subdivide()
            self.clean()

    def flow(self):
        """
        Perform the flow of the points in the domain.
        """
        # update point status
        h = self.h(self.x[self.active])
        self.active[self.active] = h > self.h_thre
        # move point on flow
        g  = self.gradient(self.x[self.active])
        dx = self.tau * np.divide(g, np.abs(g), where=np.abs(g)>0)
        self.x[self.active] -= dx

    def is_simplex_active(self):
        """
        Check if the simplices are active.

        Returns:
        - A boolean array indicating if the simplices are active.
        """
        # simplex is active if the either point that belong to the simplex is active
        return self.active[:-1] & self.active[1:]

    def simplex_size(self):
        """
        Compute the size of the active simplices.

        Returns:
        - The size of the active simplices.
        """
        # compute the size of the active simplices
        return np.abs(np.diff(self.x))

    def active_simplex_left(self):
        """
        Get the left point of the active simplices.

        Returns:
        - The left point of the active simplices.
        """
        i = np.where(self.is_simplex_active())[0]
        return self.x[:-1][i]
    
    def active_simplex_right(self):
        """
        Get the right point of the active simplices.

        Returns:
        - The right point of the active simplices.
        """
        i = np.where(self.is_simplex_active())[0]
        return self.x[1:][i]

    def subdivide(self):
        """
        Subdivide the simplices by inserting new points at the midpoint.

        Returns:
        - The new points.
        """
        # gets simplex info
        a = self.is_simplex_active()
        s = self.simplex_size()
        # find indices where active simplex satisfy size > delta
        i = np.where(a & (s>self.x_delta))[0]
        # create new points as midpoint
        xm= (self.x[i] + self.x[i+1])/2
        # insert new points
        self.x = np.insert(self.x, i+1, xm)
        self.active = np.insert(self.active, i+1, True)

    def clean(self):
        """
        Removes points by contracting the succeeding inactive points
        to an single inactive point. This is equivalent to keep a point
        if the point is active or a point before the point is active.
        """
        i = self.active[:-1] | self.active[1:]
        i = np.append(True, i)
        self.x = self.x[i]
        self.active = self.active[i]
        
    def romberg(self, N=3, phi=None):
        """
        Perform Romberg integration for each active simplex and sum it
        to get the Pichard Lefschetz integral.
        """
        if phi is None:
            phi = self.phi
        # get left and right points of active simplices
        x_left  = self.active_simplex_left()
        x_right = self.active_simplex_right()
        # compute the integral for each simplex
        I = np.array([romberg(phi, x_left[i], x_right[i], max_iter=N) for i in range(x_left.size-1)])
        return np.sum(I)

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
    R = np.zeros((max_iter, max_iter), dtype=complex)
    h = xmax - xmin
    
    # First estimate with the trapezoidal rule
    R[0, 0] = 0.5 * h * (func(xmin) + func(xmax))
    
    for i in range(1, max_iter):
        h /= 2
        
        # Composite trapezoidal rule for 2^i panels
        sum_f = np.sum(func(xmin + (2*k - 1) * h) for k in range(1, 2**(i-1) + 1))
        R[i, 0] = 0.5 * R[i-1, 0] + sum_f * h
        
        # Romberg extrapolation
        for k in range(1, i+1):
            R[i, k] = (4**k * R[i, k-1] - R[i-1, k-1]) / (4**k - 1)
        
        # Check for convergence
        if np.abs(R[i, i] - R[i-1, i-1]) < tol:
            return R[i, i]
    
    return R[max_iter-1, max_iter-1]