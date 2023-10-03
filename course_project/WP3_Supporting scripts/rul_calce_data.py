import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utility import drop_outlier_sw
from utility import cal_ttf
from scipy.stats import norm
from scipy.stats import gaussian_kde
from pf import pf_class
from tqdm import tqdm
import pickle


def create_mdl(sigma_u, sigma_v):
    '''
    This function defines the state space model describing the degardation. The model will be used in the PF.
    
    Args:
    - sigma_u: The stds of the process noises.
    - sigma_v: The std of the observation noise.

    Outputs:
    - nx: Number of state variables.
    - ny: Number of observations.
    - sys: Function handle to the state equation.
    - obs: Function handle to the observation equation.
    - p_yk_given_xk: Function handle to a function that give p(yk|xk).
    - gen_sys_noise: Function handle to generate the system noise.
    '''    
    # Degradation model.
    def degradation_path(x, t):
        return x[0] * np.exp(x[1] * t) + x[2] * np.exp(x[3] * t)    
    # Process model.
    nx = 5  # number of states
    nu = 4  # size of the vector of process noise
    def sys(tk, xkm1, uk):
        xk = np.zeros(nx)
        xk[0] = xkm1[0] + uk[0]
        xk[1] = xkm1[1] + uk[1]
        xk[2] = xkm1[2] + uk[2]
        xk[3] = xkm1[3] + uk[3]
        xk[4] = degradation_path(np.array([xk[0], xk[1], xk[2], xk[3]]), tk)
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
    # # PDF of process noise and noise generator function
    # def p_sys_noise(u):
    #     return norm.pdf(u, 0, sigma_u)   

    # Define observation equation.
    ny = 1  # number of observations
    # Observation equation y[k] = obs(k, x[k], v[k]);    
    def obs(xk, vk):
        return xk[-1] + vk
    # PDF of observation noise and noise generator function
    def p_obs_noise(v):
        return norm.pdf(v, 0, sigma_v)
    # # Generate observation noise.
    # def gen_obs_noise():
    #     return np.random.normal(0, sigma_v)

    # Observation likelihood.
    def p_yk_given_xk(yk, xk):
        return p_obs_noise(yk - obs(xk, 0))

    return nx, ny, sys, obs, degradation_path, p_yk_given_xk, gen_sys_noise


def individual_battery_run(t, y, sigma_u, sigma_v, Ns, threshold, idx_ttf, idx_pred, max_RUL, n_plt=5):
    '''
    This function run the PF for the data from a given battery and predict its RUL.

    Args:
    - t: The time of the observations.
    - y: Observation data.
    - sigma_u: Std of the process noises.
    - sigma_v: Std of the observation noise.
    - Ns: Number of particles.
    - threshold: Degradation threshold.
    - idx_pred: Index of the time instants that needs to predict RUL.
    - t_pred: A vector of all the time instants, including the observation data, and the time horizon until max_ite.
    - max_RUL: If max_ite is reached, set the RUL to max_RUL.

    Outputs:
    - xh: Estimates of the state variables from the PF.
    - yh: Estimate of the degradation from the PF.
    - y_bands: Confidence bands.
    - rul_mean: RUL prediciton.
    - rul_bands: Confidence bands of the RUL prediction.
    - rul: RUL prediction from all the particles.
    - rul_weights: Weights of the rul predicitons of all the particles.
    '''
    # Create the state space model.
    nx, ny, sys, obs, degradation_path, p_yk_given_xk, gen_sys_noise = create_mdl(sigma_u, sigma_v)

    # Prepare particle filter.       
    # Generating the initial particles.
    def gen_x0(Ns=1, t_0=t[0]):
        x0 = np.zeros((nx, Ns))
        x0[0, :] = np.random.uniform(1, 1.2, size=Ns)
        x0[1, :] = np.random.uniform(-1e-3, -2e-5, size=Ns)
        x0[2, :] = np.random.uniform(-2e-2, -1e-3, size=Ns)
        # x0[1, :] = np.random.uniform(-1e-3, 0, size=Ns)
        # x0[2, :] = np.random.uniform(-2e-1, 0, size=Ns)
        x0[3, :] = np.random.uniform(.001, .01, size=Ns)
        x0[4, :] = degradation_path(x0, t_0)
        return x0

    # Create a particle filter object.
    pf = pf_class(
        Ns=int(Ns), t=t, nx=nx, gen_x0=gen_x0, sys=sys, obs=obs,
        p_yk_given_xk=p_yk_given_xk, gen_sys_noise=gen_sys_noise,
        initial_outlier_quota=5, degradation_path=degradation_path
    )
    # Do the filtering:
    T = len(t) # Number of time steps
    xh = np.zeros((nx, T)) # Estimate of the state variables.
    yh = np.zeros((ny, T)) # Estimate of the observation variable.
    # State estimation.
    for k in tqdm(range(1, T)):
        pf.k = k
        xh[:, k] = pf.state_estimation(y[k])        
    # Estimate degradation and CI.
    y_bands = np.zeros((2, T))
    for k in range(1,T):
        # Mean and CI of y.
        x_tmp = pf.particles[:, :, k]
        y_tmp = x_tmp[4, :]
        y_w = pf.w[:, k]
        yh[:, k], y_bands[:, k] = pf.get_state_estimation(y_tmp, y_w)

    # RUL prediction.
    # Run the RUL prediction.
    rul_mean, rul_bands, rul, rul_weights, deg_mean, deg_bands = pf.rul_prediction(threshold, idx_pred, t, max_RUL=max_RUL)    

    # Visulize the results.
    # Create n_plt Figures of the predicted degradation trajectories.
    idx_plt = np.linspace(0, len(idx_pred)-1, n_plt, dtype=int)
    for i in range(n_plt):
        idx_pred_start = idx_pred[idx_plt[i]]
        deg_pred = deg_mean[idx_plt[i]]
        deg_pred_bands = deg_bands[idx_plt[i]]
        plot_deg_pred(t, y, threshold, idx_ttf, yh, y_bands, idx_pred_start, deg_pred, deg_pred_bands)    
    # RUL.
    true_ttf = t[idx_ttf]
    plot_rul_t(t, true_ttf, idx_pred, rul_mean, rul_bands)
    # 3d plot of the predicted RULs.
    plot_rul_density(t, idx_pred, max_RUL, rul_mean, rul, rul_weights, true_ttf)

    return xh, yh, y_bands, rul_mean, rul_bands, rul, rul_weights, pf


def plot_deg_pred(t, y, threshold, idx_ttf, yh, y_bands, idx_pred_start, deg_pred, deg_pred_bands):
    '''
    Create a plot for the estimated degradation + predicted prediction.

    Args:
    - t: An array of measurement times.
    - y: An array of measured degradation.
    - threshold: Failure threshold.
    - idx_ttf: Index of true time to failure.
    - yh: Estimated degradation.
    - y_bands: CI of yh.
    - idx_pred_start: Index of the start of RUL prediction.
    - deg_pred: An array of predicted degradation.
    - deg_pred_bands: CI of deg_pred.

    Output: None.
    '''
    ax1b = plt.subplot()
    ax1b.plot(t, y, 'b-', label='Measurement')
    ax1b.plot(t[idx_pred_start]+np.arange(1, len(deg_pred)+1, 1), deg_pred, 'k--', label='Degradation prediction')
    ax1b.plot(t[1:idx_pred_start], yh.reshape(-1)[1:idx_pred_start], 'k-', label='Degradation estimation')
    ax1b.fill_between(t[1:idx_pred_start], y_bands[0, 1:idx_pred_start], y_bands[1, 1:idx_pred_start], color='blue', alpha=.25)
    ax1b.fill_between(t[idx_pred_start]+np.arange(1, len(deg_pred)+1, 1), deg_pred_bands[:, 0], deg_pred_bands[:, 1], color='blue', alpha=.25, label='90% Confidence interval')
    ax1b.plot(t[idx_ttf], y[idx_ttf], 'rx', label='Time to failure')
    ax1b.plot(np.linspace(0, t[idx_pred_start]+len(deg_pred)+1, 20), threshold*np.ones(20), 'r--', label='Failure threshold')
    ax1b.plot(t[idx_pred_start]*np.ones(6), np.arange(.6, 1.2, .1), '--k')
    ax1b.legend()
    ax1b.set_xlabel('t')
    ax1b.set_ylabel('Capacity (Ah)')
    plt.show()


def plot_rul_t(t, true_ttf, idx_pred, rul_mean, rul_bands):
    '''
    Create a plot of predicted RUL(t) V.S. t.

    Args:
    - t: An array of measurement times.
    - true_TTF: True time to failure.
    - idx_pred: An array containing the indexes of the RUL prediction moments.
    - rul_mean, rul_bands: The predicted RUL and its CI>

    Outputs: None.
    '''
    ax2 = plt.subplot()
    ax2.plot(t[idx_pred], rul_mean, '-ko', label='RUL prediction')
    ax2.fill_between(t[idx_pred], rul_bands[:, 0], rul_bands[:, 1], color='blue', alpha=.25, label='90% Confidence interval')
    ax2.plot(t[idx_pred], (true_ttf-t[idx_pred])*(true_ttf-t[idx_pred]>=0), '--r', label='True RUL')
    ax2.legend()
    ax2.set_xlabel('t')
    ax2.set_ylabel('RUL')
    plt.show()


def plot_rul_density(t, idx_pred, max_RUL, rul_mean, rul, rul_weights, true_ttf):
    ''' 
    This function plots the density of the predicted RUL at different time.
    
    Args:
    - t: An array of measurement times.
    - idx_pred: An array containing the indexes of the RUL prediction moments.
    - max_RUL: Maximal value of RUL search. If true RUL>this value, it will be set to this value.
    - rul_mean: Predicted RUL (mean)
    - rul: Predicted RUL (all samples)
    - rul_weights: The weights of the rul
    - true_ttf: True time to failure.
    '''
    fig = plt.figure()
    fig.set_size_inches(20, 6)
    ax3 = fig.add_subplot(projection='3d')
    # Set the x and y data for the plot
    xi = t[idx_pred]
    yi = np.linspace(0, max_RUL, 1000)
    xx, _ = np.meshgrid(xi, yi)
    den = np.zeros_like(xx)
    # Plot.
    for i in range(len(idx_pred)):
        # for each time step perform a kernel density estimation
        try:
            kde = gaussian_kde(dataset=rul[:, i], weights=rul_weights[:,i])
            den[:, i] = kde.evaluate(yi)
            ax3.plot(xi[i]*np.ones_like(yi), yi, kde.evaluate(yi))
        except np.linalg.LinAlgError:
            print('LinAlgError at ')
            print(i)
            continue
    # Show the plot
    ax3.set_zlim(0, .1)
    ax3.plot(t[idx_pred], rul_mean, '-ko', zs=0, zdir='z', label='RUL prediction')
    ax3.plot(t[idx_pred], (true_ttf-t[idx_pred])*(true_ttf-t[idx_pred]>=0), '--r', zs=0, zdir='z', label='True RUL')
    ax3.legend()
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('RUL')
    ax3.set_zlabel('Density')
    plt.show()


# Here we test the PF on real data from Calce..
if __name__ == '__main__':
    # # Run particle filtering to estimate the state variables.
    # battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    # # Directly read from the archived data.
    # with open('data_all.pickle', 'rb') as f:
    #     data_all = pickle.load(f)

    # Run particle filtering to estimate the state variables.
    battery_list = ['CS2_33', 'CS2_34']
    # Directly read from the archived data.
    with open('data_all_halfC.pickle', 'rb') as f:
        data_all = pickle.load(f)

    name = battery_list[1]
    battery = data_all[name]
    battery.fillna(method='ffill', inplace=True)
    # Get the time and degradation measurement. Perform filtering.
    t = battery['cycle']
    y = battery['discharging capacity']
    t = np.array(t)
    y = np.array(y)

    # # We can try also eliminate the outliers explicitly:
    # rolling_window = 20
    # idx = drop_outlier_sw(y, rolling_window)
    # t = np.array(t[idx])
    # y = np.array(y[idx])

    # Calculate true TTF.
    threshold = .7*1.1
    true_ttf, idx_ttf = cal_ttf(t, y, threshold)
    t = t[:idx_ttf+10]
    y = y[:idx_ttf+10]
    T = len(t) # Number of time steps
    
    # Define the Parameters.
    # For the PF.
    sigma_u = np.array([1e-2, 1e-5, 1e-4, 1e-3])
    sigma_v = 1e-2
    Ns = 1e3
    # For the RUL prediction.
    max_RUL = 600 # RUL when not failure found.
    idx_start = 300
    step = 10
    idx_pred = np.arange(idx_ttf-idx_start, idx_ttf+step, step, dtype=int) # Index of the prediction instants.
    
    xh, yh, y_bands, rul_mean, rul_bands, rul, rul_weights, pf = individual_battery_run(t, y, sigma_u, sigma_v, Ns, threshold, idx_ttf, idx_pred, max_RUL, n_plt=10)

    # # Save the result.
    # file_name = 'result_' + name + '.pickle'
    # with open(file_name, 'wb') as f:
    #     pickle.dump([t, y, threshold, idx_ttf, idx_pred, true_ttf, max_RUL, 
    #         xh, yh, y_bands, rul_mean, rul_bands, rul, rul_weights, t_pred,
    #         pf.particles, pf.w
    #     ], f, protocol=pickle.HIGHEST_PROTOCOL)    
