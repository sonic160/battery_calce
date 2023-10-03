import pickle
from utility import drop_outlier_sw
from utility import cal_ttf
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import scipy.optimize
import math
from rul_calce_data import plot_deg_pred, plot_rul_t, plot_rul_density
from scipy.stats import truncnorm


def rul_battery_reg(t, y, name):
    '''
    This function predicts the RUL basead on degradation measurements y, and plot the results.

    Args:
    - t: An array of the measurement times.
    - y: An array of the degradation measurements.

    Outputs: None.
    '''
    # Calculate true TTF.
    threshold = .7*1.1
    true_ttf, idx_ttf = cal_ttf(t, y, threshold)
    t = t[:idx_ttf+10]
    y = y[:idx_ttf+10]
    T = len(t) # Number of time steps

    # Define when to perform RUL prediction.
    max_RUL = 600 # RUL when not failure found.
    idx_start = 300
    step = 10
    idx_pred = np.arange(idx_ttf-idx_start, idx_ttf+step, step, dtype=int) # Index of the prediction instants.
    n_pred = len(idx_pred)
    n_bs = int(1e3) # Boostrap samples used to calculate the CI.

    # Define the degradation model.
    def degradation_mdl(t, x_0, x_1, x_2, x_3):
        return x_0*np.exp(x_1*t) + x_2*np.exp(x_3*t)
    n_x = 4

    # Assign memory.
    # RUL predictions.
    rul = max_RUL*np.ones((n_bs, n_pred))
    rul_mean = np.zeros(n_pred)
    rul_bands = np.zeros((n_pred, 2))
    # Estimated degradation.
    deg_est_mean = []
    deg_est_bands = []
    # Predicted degradation.
    deg_pred_mean = []
    deg_pred_bands = []
    # Initial values for sovling the degradation path equation.
    x0 = np.array([1.1, -5e-5, -1.5e-3, .006])
    bounds = ((1, -1e-3, -2e-2, .001), (1.2, -2e-5, -1e-3, .01))

    # Loop for all the prediction moments.
    for i in tqdm(range(n_pred)):
        idx_pred_i = idx_pred[i] # Index of the prediction instant.        
        # Estimate model parameters.
        t_data = t[:idx_pred_i+1]
        y_data = y[:idx_pred_i+1]
        x_hat, x_sample = degradation_est(degradation_mdl, n_x, n_bs, deg_est_mean, deg_est_bands, x0, bounds, t_data, y_data)        

        # For each sample, make RUL prediction.
        for k in range(n_bs):
            x_run = x_sample[:, k]
            hdl_eq = lambda xx: degradation_mdl(xx, x_run[0], x_run[1], x_run[2], x_run[3])-threshold                        
            ttf_run = scipy.optimize.fsolve(hdl_eq, t[idx_pred_i])
            rul[k, i] = ttf_run + 1 - t[idx_pred_i]
            if rul[k, i] > max_RUL:
                rul[k, i] = max_RUL
            elif rul[k, i] < 0:
                rul[k, i] = 0        
        tmp_rul = rul[:, i]
        _, rul_bands[i, :] = get_state_estimation(tmp_rul, 1/n_bs*np.ones_like(tmp_rul))
        hdl_eq = lambda xx: degradation_mdl(xx, x_hat[0], x_hat[1], x_hat[2], x_hat[3])-threshold
        rul_mean[i] = scipy.optimize.fsolve(hdl_eq, t[idx_pred_i]) + 1 - t[idx_pred_i]
        if rul_mean[i] < 0:
            rul_mean[i] = 0

        # Degradation state prediction:
        n_t = math.floor(rul_mean[i]) + 10
        deg_pred = np.zeros((n_bs, n_t))
        for j in range(n_bs):
            x_cur = x_sample[:, j] # Get the current particles.
            tt = np.arange(t[idx_pred_i]+1, t[idx_pred_i]+1+n_t)
            deg_pred[j, :] = degradation_mdl(tt, x_cur[0], x_cur[1], x_cur[2], x_cur[3])
        # Get the mean and bands of the degradation prediction.
        deg_mean = np.zeros(n_t)
        deg_bands = np.zeros((n_t, 2))
        for j in range(n_t):
            _, deg_bands[j, :] = get_state_estimation(deg_pred[:, j], 1/n_bs*np.ones(n_bs))
        deg_mean = degradation_mdl(tt, x_hat[0], x_hat[1], x_hat[2], x_hat[3])
        deg_bands[deg_bands<.6] = .6
        deg_bands[deg_bands>1.2] = 1.2
        deg_pred_mean.append(deg_mean)
        deg_pred_bands.append(deg_bands)
        
    rul_mean[rul_mean>max_RUL] = max_RUL
    rul_mean[rul_mean<0] = 0
    rul_bands[rul_bands>max_RUL] = max_RUL
    rul_bands[rul_bands<0] = 0

    # Visulize the results.
    # Create n_plt Figures of the predicted degradation trajectories.
    n_plt = 5
    idx_plt = np.linspace(0, len(idx_pred)-1, n_plt, dtype=int)
    for i in range(n_plt):
        idx_pred_start = idx_pred[idx_plt[i]]
        deg_pred = deg_pred_mean[idx_plt[i]]
        deg_bands = deg_pred_bands[idx_plt[i]]
        yh = deg_est_mean[idx_plt[i]]
        y_bands = deg_est_bands[idx_plt[i]]
        plot_deg_pred(t, y, threshold, idx_ttf, yh, y_bands, idx_pred_start, deg_pred, deg_bands)    
    # RUL.
    true_ttf = t[idx_ttf]
    plot_rul_t(t, true_ttf, idx_pred, rul_mean, rul_bands)
    # 3d plot of the predicted RULs.
    plot_rul_density(t, idx_pred, max_RUL, rul_mean, rul, 1/n_bs*np.ones_like(rul), true_ttf)

    # Save the result.
    file_name = 'result_reg_' + name + '.pickle'
    with open(file_name, 'wb') as f:
        pickle.dump([t, y, threshold, idx_ttf, idx_pred, true_ttf, max_RUL, 
            x_hat, yh, y_bands, rul_mean, rul_bands, rul, x_sample
        ], f, protocol=pickle.HIGHEST_PROTOCOL) 


def degradation_est(degradation_mdl, n_x, n_bs, deg_est_mean, deg_est_bands, x0, bounds, t_data, y_data):
    '''
    Estimate the parameter of the degradation model given data t_data, y_data. Use bootstrap sampling to caculate the CI.

    Args:
    - t, 
    - T: Number of 
    - degradation_mdl: Function handle to the degradation model.
    - n_x: Number of parameters in the degradation model.
    - n_bs: Bootstrap sample size.
    - deg_est_mean: A list containing the point estimate of the degradation estimation.
    - deg_est_bands: A list containing the CIs of the degradation estimation.
    - x0: Initial value of the search.
    - bounds: Bounds of the model parameters.
    - t_data: Time of the observation data.
    - y_data: Observation data.
    '''
    # Point estimator.
    x_hat, _ = curve_fit(degradation_mdl, 
        xdata=t_data, ydata=y_data, p0=x0, bounds=bounds)

    # Construct CI using bootstrap method.
    n_measurement = len(y_data)
    idx_bs = np.random.choice(np.arange(0, n_measurement, 1), size=(n_bs, n_measurement))
    x_sample = np.zeros((n_x, n_bs))
    for i in range(n_bs):
        idx_i = idx_bs[i, :]
        t_bs = t_data[idx_i]
        y_bs = y_data[idx_i]
        x_sample[:, i], _ = curve_fit(degradation_mdl, xdata=t_bs, ydata=y_bs, p0=x0, bounds=bounds)

    # Estimate degradation before the prediction point.
    # Reserve memory.
    yh = np.zeros(n_measurement)
    y_bands = np.zeros((2, n_measurement))
    y_sample = np.zeros((n_bs, n_measurement))
        # Degradation state estimation:
    for k in range(n_bs):
        x_run = x_sample[:, k]
        y_sample[k, :] = degradation_mdl(t_data, x_run[0], x_run[1], x_run[2], x_run[3])
    for j in range(n_measurement):
        _, y_bands[:, j] = get_state_estimation(y_sample[:, j], 1/n_bs*np.ones(n_bs))
    yh = degradation_mdl(t_data, x_hat[0], x_hat[1], x_hat[2], x_hat[3])
    deg_est_mean.append(yh)
    y_bands[y_bands<.6] = .6
    y_bands[y_bands>1.2] = 1.2
    deg_est_bands.append(y_bands)

    return x_hat,x_sample     


def get_state_estimation(x_sample, weights, alpha=.1):
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


if __name__ == '__main__':
    # Run particle filtering to estimate the state variables.
    battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    # Directly read from the archived data.
    with open('data_all.pickle', 'rb') as f:
        data_all = pickle.load(f)

    name = battery_list[0]
    battery = data_all[name]
    battery.fillna(method='ffill', inplace=True)
    # Get the time and degradation measurement. Perform filtering.
    t = battery['cycle']
    y = battery['discharging capacity']
    t = np.array(t)
    y = np.array(y)

    # rolling_window = 20
    # idx = drop_outlier_sw(y, rolling_window)
    # t = np.array(t[idx])
    # y = np.array(y[idx])

    rul_battery_reg(t, y)