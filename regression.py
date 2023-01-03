import pickle
from utility import drop_outlier_sw
from utility import cal_ttf
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def rul_battery_reg(t, y, name):
    # Calculate true TTF.
    threshold = .7*1.1
    true_ttf, idx_ttf = cal_ttf(t, y, threshold)
    t = t[:idx_ttf+10]
    y = y[:idx_ttf+10]
    T = len(t) # Number of time steps
    
    # Define the degradation model.
    def degradation_mdl(t, x_0, x_1, x_2, x_3):
        return x_0*np.exp(x_1*t) + x_2*np.exp(x_3*t)
    n_x = 4 

    # For the RUL prediction.
    max_ite = 60 # Maximun number of prediction states.
    max_RUL = 60 # RUL when not failure found.
    idx_start = 50
    step = 1
    idx_pred = np.arange(idx_ttf-idx_start, idx_ttf+step, step, dtype=int) # Index of the prediction instants.
    # Create the time.
    t_pred = np.arange(t[-1]+1, t[-1] + max_ite + 1, 1) 
    t_pred = np.concatenate((t, t_pred))
  
    n_pred = len(idx_pred)
    Ns = int(1e3)

    yh = np.zeros(T)
    y_bands = np.zeros((2, T))
    y_sample = np.zeros((Ns, T))

    rul = max_RUL*np.ones((Ns, n_pred))
    rul_mean = np.zeros(n_pred)
    rul_bands = np.zeros((n_pred, 2))

    # Do the RUL prediction
    x0 = np.array([1.1, -5e-5, -1.5e-3, .006])
    bounds = ((1, -1e-3, -2e-2, .001), (1.2, -2e-5, -1e-3, .01))
    for i in tqdm(range(n_pred)):
        idx_pred_i = idx_pred[i] # Index of the prediction instant.
        
        # Estimate model parameters.
        t_data = t[:idx_pred_i+1]
        y_data = y[:idx_pred_i+1]
        popt, pcov = curve_fit(degradation_mdl, 
        xdata=t_data, ydata=y_data, p0=x0, bounds=bounds)

        x_hat = popt
        std_x = np.sqrt(np.diag(pcov))
        x_sample = np.zeros((n_x, Ns))

        for k in range(n_x):
            x_sample[k, :] = np.random.normal(x_hat[k], std_x[k], size=Ns)
        
        for k in range(Ns):
            x_run = x_sample[:, k]
            hdl_eq = lambda xx: degradation_mdl(xx, x_run[0], x_run[1], x_run[2], x_run[3])-threshold
            
            if i == n_pred-1:
                y_sample[k, :] = hdl_eq(t)+threshold
            
            ttf_run = scipy.optimize.fsolve(hdl_eq, t[idx_pred_i])
            rul[k, i] = ttf_run + 1 - t[idx_pred_i]
            if rul[k, i] > max_RUL:
                rul[k, i] = max_RUL
            elif rul[k, i] < 0:
                rul[k, i] = 0
        
        tmp_rul = rul[:, i]
        _, rul_bands[i, :] = get_state_estimation(tmp_rul, 1/Ns*np.ones_like(tmp_rul))
        hdl_hat = lambda t: degradation_mdl(t, x_hat[0], x_hat[1], x_hat[2], x_hat[3])-threshold
        ttf_run = scipy.optimize.fsolve(hdl_hat, t[idx_pred_i])        
        rul_mean[i] = ttf_run+1-t[idx_pred_i]


    hdl_hat = lambda t: degradation_mdl(t, x_hat[0], x_hat[1], x_hat[2], x_hat[3])-threshold
    yh = hdl_hat(t)+threshold
    for k in range(T):
        _, y_bands[:, k] = get_state_estimation(y_sample[:, k], 1/Ns*np.ones_like(y_sample[:,k]))

    rul_mean[rul_mean>max_RUL] = max_RUL
    rul_mean[rul_mean<0] = 0

    # Visualize the results.
    # Plot the degradation.
    ax1 = plt.subplot()
    ax1.plot(t, y, 'bo', label='Measurement')
    ax1.plot(t, threshold*np.ones_like(t), 'r--', label='Failure threshold')
    ax1.plot(t[idx_ttf], y[idx_ttf], 'rx', label='Time to failure')
    ax1.plot(t, yh, 'k+-', label='Estimation')
    ax1.fill_between(t, y_bands[0, :], y_bands[1, :], color='blue', alpha=.25, label='90% Confidence interval')
    ax1.set_xlabel('t')
    ax1.set_ylabel('Capacity (Ah)')
    ax1.legend()
    plt.show()

    # RUL.
    true_ttf = t[idx_ttf]
    ax2 = plt.subplot()
    ax2.plot(t_pred[idx_pred], rul_mean, '-ko', label='RUL prediction')
    ax2.fill_between(t_pred[idx_pred], rul_bands[:, 0], rul_bands[:, 1], color='blue', alpha=.25, label='90% Confidence interval')
    ax2.plot(t_pred[idx_pred], (true_ttf-t_pred[idx_pred])*(true_ttf-t_pred[idx_pred]>=0), '--r', label='True RUL')
    ax2.legend()
    ax2.set_xlabel('t')
    ax2.set_ylabel('RUL')
    plt.show()

    # 3d plot of the predicted RULs.
    fig = plt.figure()
    fig.set_size_inches(20, 6)
    ax3 = fig.add_subplot(projection='3d')
    # Set the x and y data for the plot
    xi = t_pred[idx_pred]
    yi = np.linspace(0, max_RUL, 1000)
    xx, yy = np.meshgrid(xi, yi)
    den = np.zeros_like(xx)
    # Plot.
    for i in range(len(idx_pred)):
        # for each time step perform a kernel density estimation
        try:
            kde = gaussian_kde(dataset=rul[:, i])
            den[:, i] = kde.evaluate(yi)
            ax3.plot(xi[i]*np.ones_like(yi), yi, kde.evaluate(yi))
        except np.linalg.LinAlgError:
            print('LinAlgError at ')
            print(i)
            continue
    # Show the plot
    ax3.set_zlim(0, .1)
    ax3.plot(t_pred[idx_pred], rul_mean, '-ko', zs=0, zdir='z', label='RUL prediction')
    ax3.plot(t_pred[idx_pred], (true_ttf-t_pred[idx_pred])*(true_ttf-t_pred[idx_pred]>=0), '--r', zs=0, zdir='z', label='True RUL')
    ax3.legend()
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('RUL')
    ax3.set_zlabel('Density')
    plt.show()        



    # # Save the result.
    # file_name = 'result_' + name + '.pickle'
    # with open(file_name, 'wb') as f:
    #     pickle.dump([t, y, threshold, idx_ttf, idx_pred, true_ttf, max_RUL, 
    #         xh, yh, y_bands, rul_mean, rul_bands, rul, rul_weights, t_pred,
    #         pf.particles, pf.w
    #     ], f, protocol=pickle.HIGHEST_PROTOCOL)


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

    rolling_window = 20
    idx = drop_outlier_sw(y, rolling_window)
    t = np.array(t[idx])
    y = np.array(y[idx])

    rul_battery_reg(t, y, name)