import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utility import load_data
import pickle
from utility import drop_outlier_sw
from utility import cal_ttf
from scipy.stats import norm
from scipy.stats import gaussian_kde
from pf import pf_class
from tqdm import tqdm


battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
# Directly read from the archived data.
with open('data_all.pickle', 'rb') as f:
    data_all = pickle.load(f)

name = battery_list[3]
battery = data_all[name]

# Get the time and degradation measurement. Perform filtering.
t = battery['cycle']
y = battery['charging capacity']

# rolling_window = 20
# idx = drop_outlier_sw(y, rolling_window)
# t = np.array(t[idx])
# y = np.array(y[idx])
t = np.array(t)
y = np.array(y)

# Calculate true TTF.
threshold = .7*1.1
true_ttf, idx_ttf = cal_ttf(t, y, threshold)

t = t[:idx_ttf+10]
y = y[:idx_ttf+10]

# Define the state space model.
# Process equation x[k] = sys(k, x[k-1], u[k]):
nx = 5  # number of states
nu = 4  # size of the vector of process noise
sigma_u = np.array([1e-2, 1e-5, 1e-4, 1e-3])
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
T = len(t) # Number of time steps
# Generating the initial particles.
def gen_x0(Ns=1, t_0=t[0]):
    x0 = np.zeros((nx, Ns))
    x0[0, :] = np.random.uniform(1, 1.2, size=Ns)
    x0[1, :] = np.random.uniform(-1e-4, -2e-5, size=Ns)
    x0[2, :] = np.random.uniform(-2e-3, -1e-3, size=Ns)
    x0[3, :] = np.random.uniform(.005, .01, size=Ns)
    x0[4, :] = x0[0, :] * np.exp(x0[1, :] * t_0) + x0[2, :] * np.exp(x0[3, :] * t_0)
    return x0
# Observation likelihood.
def p_yk_given_xk(yk, xk):
    return p_obs_noise(yk - obs(xk, 0))


# Run particle filtering to estimate the state variables.
Ns = 1e3
xh = np.zeros((nx, T)) # Estimate of the state variables.
yh = np.zeros((ny, T)) # Estimate of the observation variable.
# Create a particle filter object.
pf = pf_class(
    Ns=int(Ns), t=t, nx=nx, gen_x0=gen_x0, sys=sys, obs=obs,
    p_yk_given_xk=p_yk_given_xk, gen_sys_noise=gen_sys_noise
)
# Do the filtering:
for k in tqdm(range(1, T)):
    pf.k = k
    xh[:, k] = pf.state_estimation(y[k])        

y_bands = np.zeros((2, T))
for k in range(1,T):
    # Mean and CI of y.
    x_tmp = pf.particles[:, :, k]
    y_tmp = x_tmp[4, :]
    y_w = pf.w[:, k]
    yh[:, k], y_bands[:, k] = pf.get_state_estimation(y_tmp, y_w)

# RUL prediction.

max_ite = 50 # Maximun number of prediction states.
max_RUL = 50 # RUL when not failure found.
n_pred = 50
step = 1
idx_pred = np.arange(idx_ttf-n_pred, idx_ttf, step, dtype=int) # Index of the prediction instants.
# Create the time.
t_pred = np.arange(t[-1]+1, t[-1] + max_ite + 1, 1) 
t_pred = np.concatenate((t, t_pred))
# Run the RUL prediction.
rul_mean, rul_bands, rul, rul_weights = pf.rul_prediction(threshold, idx_pred, t_pred, max_ite=max_ite, max_RUL=max_RUL)

# Plot the degradation.
fig, axes = plt.subplots(3, 1, figsize=(16, 24))
axes[0].plot(t, y, 'bo', label='Measurement')
axes[0].plot(t, threshold*np.ones_like(t), 'r--', label='Failure threshold')
axes[0].plot(t[idx_ttf], y[idx_ttf], 'rx', label='Time to failure')
axes[0].plot(t[1:], yh.reshape(-1)[1:], 'k+-', label='PF estimation')
axes[0].fill_between(t[1:], y_bands[0, 1:], y_bands[1, 1:], color='blue', alpha=.25, label='90% Confidence interval')
axes[0].set_xlabel('t')
axes[0].set_ylabel('Capacity (Ah)')
axes[0].legend()

# RUL.
ax = axes[1]
ax.plot(t_pred[idx_pred], rul_mean, '-ko', label='RUL prediction')
ax.fill_between(t_pred[idx_pred], rul_bands[:, 0], rul_bands[:, 1], color='blue', alpha=.25, label='90% Confidence interval')
ax.plot(t_pred[idx_pred], (true_ttf-t_pred[idx_pred])*(true_ttf-t_pred[idx_pred]>=0), '--r', label='True RUL')
ax.legend()
ax.set_xlabel('t')
ax.set_ylabel('RUL')

plt.show()
