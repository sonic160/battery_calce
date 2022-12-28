import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm


class pf_class:
    '''
    This is the definition of particle filetering class.

    Methods:
    * __init__: Initialization function.
    * particle_filter(): Implement the particle filtering.
    '''

    def __init__(self, Ns, t, nx, gen_x0, sys, p_yk_given_xk, gen_sys_noise):
        '''
        Initialization function of the PF class.

        Args:
        - Ns: Particle size.
        - t: An array of the measuring time.
        - nx: Number of state variables, a scalar value.
        - w: weights   (Ns x T)
        - particles: particles (nx x Ns x T)
        - gen_x0: function handle of a procedure that samples from the initial pdf p_x0
        - p_yk_given_xk: function handle of the observation likelihood PDF p(y[k] | x[k])
        - gen_sys_noise: function handle of a procedure that generates system noise
        - sys: function handle to process equation
        '''
        self.k = 0 # Step.
        self.Ns = Ns # Particle size.
        T = len(t) # Get the number of measuring points.
        self.w = np.zeros((Ns, T)) # Initial the weights.
        self.particles = np.zeros((nx, Ns, T)) # Initial the particles.
        self.gen_x0 = gen_x0
        self.sys = sys
        self.p_yk_given_xk = p_yk_given_xk
        self.gen_sys_noise = gen_sys_noise


    def particle_filter(self, yk, resampling_strategy='multinomial_resampling'):
        """
        This function implement a Generic particle filter. Note: when resampling is performed on each step this algorithm is called
        the Bootstrap particle filter.

        Usage:
        xhk = pf.particle_filter(yk, resamping_strategy)

        Args:
        * yk   = observation vector at time k (column vector)
        * resampling_strategy = resampling strategy. Set it either to 
                                'multinomial_resampling' or 'systematic_resampling'

        Outputs:
        * xhk   = estimated state

        Reference: Arulampalam et. al. (2002).  A tutorial on particle filters for online nonlinear/non-gaussian bayesian tracking. IEEE Transactions on Signal Processing. 50 (2). p 174--188
        
        Original verision in Matlab programmed by: Diego Andres Alvarez Marin (diegotorquemada@gmail.com), February 29, 2012.
        
        Converted to Python: Zhiguo Zeng, (zhiguo.zeng@centralesupelec.fr), 26/12/2022.
        """   

        k = self.k
        if k == 0:
            raise ValueError('error: Cannot have only one step!')

        # Initialize variables
        Ns = self.Ns  # number of particles
        if k == 1:
            self.particles[:, :, 0] = self.gen_x0(Ns)  # Initial particles.
            self.w[:, 0] = np.repeat(1 / Ns, Ns)  # all particles have the same weight

        # Separate memory
        xkm1 = self.particles[:, :, k-1]  # extract particles from last iteration;
        wkm1 = self.w[:, k - 1]  # weights of last iteration
        xk = np.zeros_like(xkm1)  # = zeros(nx,Ns);
        wk = np.zeros_like(wkm1)  # = zeros(Ns,1);

        # Algorithm 3 of Ref [1]
        uk = self.gen_sys_noise(Ns)
        for i in range(Ns):
            # Using the PRIOR PDF: pf.p_xk_given_xkm1: eq 62, Ref 1.
            xk[:, i] = self.sys(t[k], t[k-1], xkm1[:, i], uk[:, i])

            # weights (when using the PRIOR pdf): eq 63, Ref 1
            wk[i] = wkm1[i] * self.p_yk_given_xk(yk, xk[:, i])


        # Normalize weight vector
        wk = wk / sum(wk)

        # Calculate effective sample size: eq 48, Ref 1
        Neff = 1 / sum(wk**2)

        # Resampling
        # remove this condition and sample on each iteration:
        # [xk, wk] = resample(xk, wk, resampling_strategy);
        # if you want to implement the bootstrap particle filter
        resample_percentaje = 0.50
        Nt = resample_percentaje * Ns
        if Neff < Nt:
            print('Resampling ...')
            xk, wk = self.resample(xk, wk, resampling_strategy)
            # {xk, wk} is an approximate discrete representation of p(x_k | y_{1:k})

        # Compute estimated state
        xhk = np.matmul(xk, wk)

        # Store new weights and particles
        self.w[:, k] = wk
        self.particles[:, :, k] = xk

        return xhk


    def resample(self, xk, wk, resampling_strategy='multinomial_resampling'):
        '''
        This function implements the resampling of PF.

        Args:
        - xk: The original particles. 
        - wk: Weights.
        - resampling_strategy: A string of the resamping strategy:
            -- 'multinomial_resampling' (default): Sampling with replacement.
            -- 'systematic_resampling': Latin hypercube sampling

        Outputs:
        - xk: The particles after resampling.
        - wk: The new weights.

        '''
        Ns = len(wk)  # Ns = number of particles
        # wk = wk / sum(wk)  # normalize weight vector (already done)

        if resampling_strategy == 'multinomial_resampling':
            with_replacement = True
            idx = np.random.choice(np.arange(Ns), Ns, p=wk, replace=with_replacement)
            # THIS IS EQUIVALENT TO:
            # edges = np.minimum(np.cumsum(wk), 1)  # protect against accumulated round-off
            # edges[-1] = 1  # get the upper edge exact
            # # this works like the inverse of the empirical distribution and returns
            # # the interval where the sample is to be found
            # _, idx = np.histogram(np.sort(np.random.rand(Ns, 1)), edges)
        elif resampling_strategy == 'systematic_resampling':
            # this is performing latin hypercube sampling on wk
            edges = np.minimum(np.cumsum(wk), [1]*len(wk))  # protect against accumulated round-off
            edges[-1] = 1  # get the upper edge exact
            u1 = np.random.random()/Ns
            # this works like the inverse of the empirical distribution and returns
            # the interval where the sample is to be found
            _, idx = np.histogram(np.arange(u1, 1, 1/Ns), edges)
        else:
            raise ValueError('Resampling strategy not implemented!')
        
        xk = xk[:, idx]  # extract new particles
        wk = np.ones(Ns)/Ns  # now all particles have the same weight

        return xk, wk


# Here we test the PF.
if __name__ == '__main__':
    tic = time.process_time()

    # Process equation x[k] = sys(k, x[k-1], u[k]);
    nx = 5  # number of states
    ny = 1  # number of observations
    nu = 4  # size of the vector of process noise
    sigma_u = 1*np.array([1e-2, 1e-6, 1e-4, 1e-5])
    nv = 1  # size of the vector of observation noise
    sigma_v = 5e-2
    t = np.arange(1, 140+1, 2)
    T = len(t) # Number of time steps

    def degradation_path(x, t):
        return x[0] * np.exp(x[1] * t) + x[2] * np.exp(x[3] * t)    
    
    def degradation_incr(xk, tk, tkm1):
        return degradation_path(xk, tk) - degradation_path(xk, tkm1)

    def sys(tk, tkm1, xkm1, uk):
        xk = np.zeros(nx)
        xk[0] = xkm1[0] + uk[0]
        xk[1] = xkm1[1] + uk[1]
        xk[2] = xkm1[2] + uk[2]
        xk[3] = xkm1[3] + uk[3]
        xk[4] = xkm1[4] + degradation_incr(xk, tk, tkm1)

        return xk

    # Observation equation y[k] = obs(k, x[k], v[k]);    
    def obs(xk, vk):
        return xk[4] + vk

    # PDF of process noise and noise generator function
    def p_sys_noise(u):
        return norm.pdf(u, 0, sigma_u)

    def gen_sys_noise(Ns=1):
        if Ns == 1:
            sample = np.random.normal(0, sigma_u)
        else:
            sample = np.zeros((nu, Ns))
            for i in range(nu):
                sample[i, :] = np.random.normal(0, sigma_u[i], size=Ns)
        
        return sample

    # PDF of observation noise and noise generator function
    def p_obs_noise(v):
        return norm.pdf(v, 0, sigma_v)

    def gen_obs_noise():
        return np.random.normal(0, sigma_v)

    # Initial PDF
    def gen_x0(Ns=1):
        x0 = np.zeros((nx, Ns))
        x0[0, :] = np.random.uniform(.88, .89, size=Ns)
        x0[1, :] = np.random.uniform(-9e-4, -8e-4, size=Ns)
        x0[2, :] = np.random.uniform(-3e-4, -2e-4, size=Ns)
        x0[3, :] = np.random.uniform(.03, .05, size=Ns)
        x0[4, :] = x0[0, :] * np.exp(x0[1, :] * t[0]) + x0[2, :] * np.exp(x0[3, :] * t[0])

        return x0
    
    # Observation likelihood.
    def p_yk_given_xk(yk, xk):
        return p_obs_noise(yk - obs(xk, 0))


    # Separate memory space
    x = np.zeros((nx, T))
    y = np.zeros((ny, T))
    u = np.zeros((nu, T))
    v = np.zeros((nv, T))
    yReal = np.zeros((ny, T))

    # Simulate a system trajectory
    xh0 = np.array([.887, -8.86e-4, -2.32e-4, .0458])  # initial state, true value
    u[:, 0] = gen_sys_noise()  # initial process noise
    v[:, 0] = gen_obs_noise()  # initial observation noise
    x[:4, 0] = xh0
    x[4, 0] = degradation_path(xh0, t[0])
    y[:, 0] = x[4, 0]

    for k in range(1, T):
        u[:, k] = gen_sys_noise()
        v[:, k] = gen_obs_noise()
        x[:, k] = sys(t[k], t[k-1], x[:, k-1], u[:, k])
        y[:, k] = obs(x[:, k], v[:, k])

    # Run particle filtering to estimate the state variables.
    xh = np.zeros((nx, T))
    xh[:4, 0] = xh0
    xh[4, 0] = degradation_path(xh0, t[0])
    yh = np.zeros((ny, T))
    yh[:, 0] = obs(xh[:, 0], 0)

    pf = pf_class(
        Ns=int(1e3), t=t, nx=nx, gen_x0=gen_x0, sys=sys,
        p_yk_given_xk=p_yk_given_xk, gen_sys_noise=gen_sys_noise
    )

    for k in range(1, T):
        print('Iteration = {}/{}'.format(k, T))
        pf.k = k
        xh[:, k] = pf.particle_filter(y[:,k])
        yh[:, k] = obs(xh[:, k], 0)

    for k in range(T):
        yReal[:, k] = obs(x[:, k], 0)

    # Plot the data
    plt.plot(t, y.reshape(-1), 'bo', t[1:], yh.reshape(-1)[1:], 'r-', t, yReal.reshape(-1), 'k-')
    plt.show()

    toc = time.process_time()
    print("Computation time = "+str(1000*(toc - tic ))+"ms")




