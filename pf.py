import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from scipy.stats import gaussian_kde
from tqdm import tqdm


class pf_class:
    '''
    This is the definition of particle filetering class.

    Methods:
    - __init__: Initialization function.
    - state_estimation(): Implement the particle filtering.
    - rul_prediction(): Predict RUL.
    - resample(): Perform resampling in state estimation.
    - get_state_estimation(): This function allows estimating the mean and CI based on samples and weights.
    '''

    def __init__(self, Ns, t, nx, gen_x0, sys, obs, p_yk_given_xk, gen_sys_noise, initial_outlier_quota=3):
        '''
        Initialization function of the PF class.

        Args:
        - Ns: Particle size.
        - t: An array of the measurement time.
        - nx: Number of state variables, a scalar value.
        - w: weights   (Ns x T), where T is the total number of measurements.
        - particles: particles (nx x Ns x T).
        - gen_x0: function handle of a procedure that samples from the initial pdf p_x0 to create initial particles.
        - p_yk_given_xk: function handle of the observation likelihood PDF p(y[k] | x[k]).
        - gen_sys_noise: function handle of a procedure that generates system noise.
        - sys: function handle to process equation.
        - obs: function handle to observation equation.
        - initial_outlier_quota: The number of concecutive outliers allowed, before reinitiating the particles.
        '''
        self.k = 0 # Current step.
        self.Ns = Ns # Particle size.
        T = len(t) # Get the number of measuring points.
        # Memory assignment.
        self.w = np.zeros((Ns, T)) # Weights.
        self.t = t # Observation time
        self.particles = np.zeros((nx, Ns, T)) # Particles.
        # Store the function handles of the state space model.
        self.gen_x0 = gen_x0
        self.sys = sys
        self.obs = obs
        self.p_yk_given_xk = p_yk_given_xk
        self.gen_sys_noise = gen_sys_noise
        self.outlier_quota = initial_outlier_quota
        self.initial_outlier_quota = initial_outlier_quota


    def state_estimation(self, yk, resampling_strategy='multinomial_resampling'):
        """
        This function implement a generic particle filter to estimate the system states. Note: when resampling is performed on each step this algorithm is called
        the Bootstrap particle filter.

        Usage:
        xhk = pf.state_estimation(yk, resamping_strategy)
        This function has to be called sequentially, at k=1,2,\cdots, 

        Args:
        - yk   = observation vector at time k (column vector)
        - resampling_strategy = resampling strategy. Set it either to 
                                'multinomial_resampling' (default) or 'systematic_resampling'

        Outputs:
        * xhk   = estimated state

        Reference: Arulampalam et. al. (2002).  A tutorial on particle filters for online nonlinear/non-gaussian bayesian tracking. IEEE Transactions on Signal Processing. 50 (2). p 174--188
        
        This function was developed based on an original verision in Matlab programmed by: Diego Andres Alvarez Marin (diegotorquemada@gmail.com), February 29, 2012.
        """   
        # Get the current step.
        t = self.t
        k = self.k
        if k == 0:
            raise ValueError('error: Cannot have only one step!')

        # Initialize variables
        Ns = self.Ns  # number of particles
        # If it is the first data point, we need to create the initial particles.
        if k == 1:
            self.particles[:, :, 0] = self.gen_x0(Ns)  # Generate the initial particles.
            self.w[:, 0] = np.repeat(1 / Ns, Ns)  # All particles have the same weight initially.

        # Separate memory
        xkm1 = self.particles[:, :, k-1]  # extract particles from last iteration;
        wkm1 = self.w[:, k-1]  # weights of last iteration
        xk = np.zeros_like(xkm1)  # Initial values for the current particles.
        wk = np.zeros_like(wkm1)  # _ for the current weights.

        # Algorithm 3 of Ref [1].
        uk = self.gen_sys_noise(Ns) # Generate the process noise. 
        for i in range(Ns): # For each particle, predict its state in the next time instant.
            xk[:, i] = self.sys(t[k], t[k-1], xkm1[:, i], uk[:, i]) # This is the state equation.
            wk[i] = wkm1[i] * self.p_yk_given_xk(yk, xk[:, i]) # Update the weights (when using the PRIOR pdf): eq 63, Ref 1.
        # Handle exception: 
        if sum(wk) == 0: # If sum(wk)==0: Keep the previous weigths.
            if self.outlier_quota == 1:
                print(f'\nReinitiate the particles: k={k}')
                xk = self.gen_x0(Ns, t[k])
                wk = np.repeat(1 / Ns, Ns)
                self.outlier_quota = self.initial_outlier_quota
            else:
                print(f'\nSmoothing due to weight NaN: k={k}')
                for i in range(Ns):
                    xk[:, i] = self.sys(t[k], t[k-1], xkm1[:, i], np.zeros(xkm1.shape[0]-1))
                wk = wkm1
                self.outlier_quota -= 1
        else: # Here we scrape the outlier.
            y_tmp = xkm1[-1, :] # This is the degradation estimation of each particles.
            y_w = wkm1
            y_mean = self.sys(t[k], t[k-1], np.matmul(xkm1, wkm1), np.zeros(xkm1.shape[0]-1))[-1]
            _, y_bands = self.get_state_estimation(y_tmp, y_w) # Get the 90% CI of the estimation.
            interval_width = y_bands[1]-y_bands[0]
            # A outlier is defined as exceeding 1.5 interval_width from the upper and lower bound.
            if (yk > y_mean+5*interval_width) | (yk < y_mean-5*interval_width):
                if self.outlier_quota == 1:
                    print(f'\nReinitiate the particles: k={k}')
                    xk = self.gen_x0(Ns, t[k])
                    wk = np.repeat(1 / Ns, Ns)
                    self.outlier_quota = self.initial_outlier_quota
                else:
                    print(f'\nSmoothing due to outlier: k={k}')
                    for i in range(Ns):
                        xk[:, i] = self.sys(t[k], t[k-1], xkm1[:, i], np.zeros(xkm1.shape[0]-1))
                    wk = wkm1
                    self.outlier_quota -= 1                    
            else:
                wk = wk/sum(wk)
                self.outlier_quota = self.initial_outlier_quota        

        # Resampling if necessary.
        resample_percentaje = 0.50
        Nt = resample_percentaje * Ns
        # Calculate effective sample size: eq 48, Ref 1
        Neff = 1 / sum(wk**2)
        if Neff < Nt:
            # print('Resampling ...')
            xk, wk = self.resample(xk, wk, resampling_strategy)

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

        if resampling_strategy == 'multinomial_resampling': # Sampling with replacement.
            with_replacement = True
            idx = np.random.choice(np.arange(Ns), Ns, p=wk, replace=with_replacement)
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


    def rul_prediction(self, threshold, idx_pred, t_pred, max_ite=70, max_RUL=145, alpha = .1):
        '''
        This function predicts the RUL based on the result of the PF.

        Args:
        - threshold: The failure threshold. Failure is defined when the degradation measures < threshold.
        - t_pred: An array representing the time horizon in which you would like to make prediction. Note:
            -- t_pred should cover t, i.e., the first part of the t_pred should contain all the measurement time instants.
            -- The second part of t_pred contains the times corresponds to the prediction steps in particle filtering. For example, suppose a step in PF represents 10 time units and the observation ends at t=100.
            If you wish to predict the degradation trajectory in the next 10 time step, then, t_pred = [t, 110, 120, \cdots, 210].
        - idx_pred: An array containing the indexes in t_pred, that represents the time instants you wish to perform RUL prediction.
            For example, t_pred = [10, 20, 30, \cdots, 100], and you have measurement data for the first three points $10, 20$ and $30$.
            If you wish to predict RUL at these three time instants, then you should set idx_pred = [0, 1, 2]. By doing so, three seperate RUL predictions will be performed. Each prediction will only use the degradation 
            measurements before it.
        - max_ite: Maximun time steps in the RUL prediction. We only extroplolate up to max_ite steps. If beyond this, the degradation still does not fall below threshold, we end the search and set RUL = max_RUL.
            The default value is $70$.
        - max_RUL: When the search for RUL ends without finding the failure time, we set the RUL = max_RUL. Default value is $145$.
        - alpha: Confidence level for calculating the confidence interval. Default value is .1.

        Outputs:
        - rul_mean: An array containing the mean value of the predicted RUL. Shape (n_pred), where n_pred is the number of predictions. 
        - rul_bands: An ndarry containting the upper and lower CI for the mean RUL. Shape: (n_pred, 2)
        - rul: An ndarray containing the predicted RULs for all the particles. Shape (Ns, n_pred) where Ns is the number of particles.
        - rul_weights: An ndarry containing the weights for the predicted RULs. Shape (Ns, n_pred).
        '''
        # Initialize the variables.
        n_pred = len(idx_pred)
        rul = max_RUL*np.ones((self.Ns, n_pred))
        rul_mean = np.zeros(n_pred)
        rul_bands = np.zeros((n_pred, 2))
        rul_weights = np.zeros((self.Ns, n_pred))

        # Do a loop to make RUL predictions:
        for i in tqdm(range(n_pred)):
            idx_pred_i = idx_pred[i] # Index of the prediction instant.
            rul_weights[:, i] = self.w[:, idx_pred_i] # Get the weights. 

            # Degradation state estimation:
            x_h = np.matmul(self.particles[:, :, idx_pred_i], self.w[:, idx_pred_i])
            # For each particle, repeat the state space model until we find failure or max_ite is reached.
            for j in range(self.Ns):
                counter = 1 # This is the step we extropolate into the future.
                x_cur = x_h
                # Repeatedly moving one step forward.
                while counter <= max_ite:
                    # Predict the future degradation.
                    x_pred = self.sys(t_pred[idx_pred_i+counter], t_pred[idx_pred_i+counter-1], x_cur, self.gen_sys_noise()) # State equation.
                    y_pred = self.obs(x_pred, 0) # Observation equation.
                    # Find failure time.
                    if y_pred < threshold: # If a failure is found.
                        rul[j, i] = t_pred[idx_pred_i+counter] - t_pred[idx_pred_i]
                        break
                    else: # Otherwise we move one step forward.
                        x_cur = x_pred
                        counter += 1

            # Calculate the mean and CI for the predicted RUL.
            rul_mean[i], rul_bands[i, :] = self.get_state_estimation(rul[:, i], rul_weights[:, i], alpha=alpha) 
            
        return rul_mean, rul_bands, rul, rul_weights


    def get_state_estimation(self, x_sample, weights, alpha=.1):
        ''' 
        This function estimates the mean and CI based on particles with weights.

        Args:
            - x: An array of the particles.
            - weights: weights.
            - alpha: confidence level, default is .1.

        Outputs:
            - est_mean: The mean value of the estimate.
            - est_bands: Confidence bands.
        '''
        # Get the mean estimation
        est_mean = np.dot(x_sample, weights)

        # Get the CI.
        idx_sorted = np.argsort(x_sample)
        x_sorted = x_sample[idx_sorted]
        w_sort = weights[idx_sorted]
        x_cdf = np.cumsum(w_sort)
        index_l = next(i for i, x in enumerate(x_cdf) if x > alpha/2)
        index_u = next(i for i, x in enumerate(x_cdf) if x > 1-alpha/2)
        est_bands = np.array([x_sorted[index_l], x_sorted[index_u]])

        return est_mean, est_bands


# Here we test the PF.
if __name__ == '__main__':
    tic = time.process_time()

    # Define the state space model.
    # Process equation x[k] = sys(k, x[k-1], u[k]):
    nx = 5  # number of states
    nu = 4  # size of the vector of process noise
    sigma_u = 1*np.array([1e-2, 1e-6, 1e-5, 1e-6])
    # sigma_u = 1e-1*np.array([1e-5, 1e-6, 1e-6, 1e-5])
    # Degradation model.
    def degradation_path(x, t):
        return x[0] * np.exp(x[1] * t) + x[2] * np.exp(x[3] * t)    
    # Degradation increments, calculated based on the degradation model.
    def degradation_incr(xk, tk, tkm1):
        return degradation_path(xk, tk) - degradation_path(xk, tkm1)
    # Process model.
    def sys(tk, tkm1, xkm1, uk):
        xk = np.zeros(nx)
        xk[0] = xkm1[0] + uk[0]
        xk[1] = xkm1[1] + uk[1]
        xk[2] = xkm1[2] + uk[2]
        xk[3] = xkm1[3] + uk[3]
        xk[4] = xkm1[4] + degradation_incr(xk, tk, tkm1)
        return xk
    # Generate system noise.
    def gen_sys_noise(Ns=1):
        if Ns == 1:
            sample = np.random.normal(0, sigma_u)
        else:
            sample = np.zeros((nu, Ns))
            for i in range(nu):
                sample[i, :] = np.random.normal(0, sigma_u[i], size=Ns)
        return sample
    # PDF of process noise and noise generator function
    def p_sys_noise(u):
        return norm.pdf(u, 0, sigma_u)   

    # Define observation equation.
    ny = 1  # number of observations
    nv = 1  # size of the vector of observation noise
    sigma_v = 1e-2
    # sigma_v = 5e-3
    # Observation equation y[k] = obs(k, x[k], v[k]);    
    def obs(xk, vk):
        return xk[4] + vk
    # PDF of observation noise and noise generator function
    def p_obs_noise(v):
        return norm.pdf(v, 0, sigma_v)
    # Generate observation noise.
    def gen_obs_noise():
        return np.random.normal(0, sigma_v)
    
    # Prepare particle filter.
    t = np.arange(1, 140+1, 1) # The observation times.
    T = len(t) # Number of time steps
    # Generating the initial particles.
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

    # Generate a simulated degradation process.
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
    # Iteratively generate the degardation path.
    for k in range(1, T):
        u[:, k] = gen_sys_noise()
        v[:, k] = gen_obs_noise()
        x[:, k] = sys(t[k], t[k-1], x[:, k-1], u[:, k])
        y[:, k] = obs(x[:, k], v[:, k])
    # Get the true degradation.
    for k in range(T):
        yReal[:, k] = obs(x[:, k], 0)

    # Run particle filtering to estimate the state variables.
    xh = np.zeros((nx, T)) # Estimate of the state variables.
    yh = np.zeros((ny, T)) # Estimate of the observation variable.
    # Create a particle filter object.
    pf = pf_class(
        Ns=int(1e3), t=t, nx=nx, gen_x0=gen_x0, sys=sys, obs=obs,
        p_yk_given_xk=p_yk_given_xk, gen_sys_noise=gen_sys_noise
    )
    # Do the filtering:
    for k in range(1, T):
        print('Iteration = {}/{}'.format(k, T))
        pf.k = k
        xh[:, k] = pf.state_estimation(y[:,k])        

    y_bands = np.zeros((2, T))
    for k in range(1,T):
        # Mean and CI of y.
        x_tmp = pf.particles[:, :, k]
        y_tmp = x_tmp[4, :]
        y_w = pf.w[:, k]
        yh[:, k], y_bands[:, k] = pf.get_state_estimation(y_tmp, y_w)

    # Visualize the result of state estimation.
    fig, ax = plt.subplots()
    ax.plot(t, y.reshape(-1), 'bo', label='Degradation measurement')
    ax.plot(t[1:], yh.reshape(-1)[1:], 'k-', label='PF estimation')
    ax.fill_between(t[1:], y_bands[0, 1:], y_bands[1, 1:], color='blue', alpha=.25, label='90% Confidence interval')
    ax.plot(t, yReal.reshape(-1), 'r--', label='True degradation')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('Degradation indicator')
    plt.show()

    # RUL prediction.
    threshold = .7 # Failure threshold.
    max_ite = 100 # Maximun number of prediction states.
    max_RUL = 100 # RUL when not failure found.
    idx_pred = np.arange(30, 140, 10) # Index of the prediction instants.
    # Create the time.
    t_pred = np.arange(t[-1]+1, t[-1] + max_ite + 1, 1) 
    t_pred = np.concatenate((t, t_pred))
    # Run the RUL prediction.
    rul_mean, rul_bands, rul, rul_weights = pf.rul_prediction(threshold, idx_pred, t_pred, max_ite=max_ite, max_RUL=max_RUL)
    
    # Visualize the result.
    fig, ax = plt.subplots()
    ax.plot(t_pred[idx_pred], rul_mean, '-ko', label='RUL prediction')
    ax.fill_between(t_pred[idx_pred], rul_bands[:, 0], rul_bands[:, 1], color='blue', alpha=.25, label='90% Confidence interval')
    # Get the true TTF.
    true_ttf = 140
    for i in range(T):
        if yReal[0, i] < threshold:
            true_ttf = t[i]
            break
    ax.plot(t_pred[idx_pred], (true_ttf-t_pred[idx_pred])*(true_ttf-t_pred[idx_pred]>=0), '--r', label='True RUL')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('RUL')

    # 3d plot of the predicted RULs.
    fig = plt.figure()
    fig.set_size_inches(16, 6)
    ax = fig.add_subplot(projection='3d')
    # Set the x and y data for the plot
    xi = t_pred[idx_pred]
    yi = np.linspace(0, 180, 1000)
    xx, yy = np.meshgrid(xi, yi)
    den = np.zeros_like(xx)
    # Plot.
    for i in range(len(idx_pred)):
        # for each time step perform a kernel density estimation
        try:
            kde = gaussian_kde(dataset=rul[:, i], weights=rul_weights[:,i])
            den[:, i] = kde.evaluate(yi)
            ax.plot(xi[i]*np.ones_like(yi), yi, kde.evaluate(yi))
        except np.linalg.LinAlgError:
            print('LinAlgError at ')
            print(i)
            continue

    # Show the plot
    ax.set_zlim(0, .1)
    ax.plot(t_pred[idx_pred], rul_mean, '-ko', zs=0, zdir='z', label='RUL prediction')
    ax.plot(t_pred[idx_pred], (true_ttf-t_pred[idx_pred])*(true_ttf-t_pred[idx_pred]>=0), '--r', zs=0, zdir='z', label='True RUL')
    ax.legend()
    ax.set_xlabel('$t$')
    ax.set_ylabel('RUL')
    ax.set_zlabel('Density')
    plt.show()

    toc = time.process_time()
    print("Computation time for RUL prediction = "+str(1000*(toc - tic ))+"ms")




