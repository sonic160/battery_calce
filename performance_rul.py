from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import pandas as pd


def rmse(rul_true, rul_pred):
    ''' Calculate the mean squared error between RUL prediction and ground truth. '''
    return mean_squared_error(rul_true, rul_pred, squared=False)


def mape(rul_true, rul_pred):
    ''' Calculate the Mean Absolute Percentage Error. '''
    mae = re(rul_true, rul_pred)
    mape = np.mean(mae)
    return mape


def re(rul_true, rul_pred):
    ''' Calculate relative error. '''
    return np.abs(1-rul_pred/rul_true)


def prediction_horizon(rul_true, rul_pred, accuracy_req=.2):
    ''' 
    Calculate the prediction horizon for a required error rate.

    Args:
    - rul_true, 
    - rul_pred, 
    - t_pred, 
    - ttf, 
    - accuracy_req=.2

    Output:
    - prediction_horizon
    '''
    relative_error = re(rul_true, rul_pred)
    if sum(relative_error < accuracy_req) == 0:
        return 0
    else:
        idx = next(i for i, x in enumerate(relative_error-accuracy_req) if x < 0)
        return rul_true[idx]

    
def alpha_coverage(rul_true, rul, rul_weights, confidence_level=.9):
    ''' Calculate alpha coverage: The percentage that alpha CI covers the true value. '''
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

    alpha = 1 - confidence_level    
    n = len(rul_true)
    counter = 0
    for i in range(n):
        _, ci = get_state_estimation(rul[:, i], rul_weights[:,i], alpha)
        if (ci[0] <= rul_true[i]) & (rul_true[i] <= ci[1]):
            counter += 1

    return counter/n


if __name__ == '__main__':
    battery_list = ['CS2_33', 'CS2_34', 'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    cols = ['Dataset', 'RMSE', 'MAPE', 'PH', 'RE_mean', 'RE_std']
    df_result_pf = pd.DataFrame(columns=cols)
    
    for name in battery_list:        
        file_name = 'result_' + name + '.pickle'
        with open(file_name, 'rb') as f:
            [t, y, threshold, idx_ttf, idx_pred, true_ttf, max_RUL, 
                xh, yh, y_bands, rul_mean, rul_bands, rul, rul_weights, particles, w
            ] = pickle.load(f)
        
        rul_true = (true_ttf-t[idx_pred])*(true_ttf-t[idx_pred]>=0)
        rul_pred = rul_mean[rul_true>0]
        rul_true = rul_true[rul_true>0]

        metric_rmse = rmse(rul_true, rul_pred)
        metric_mape = mape(rul_true, rul_pred)
        metric_re = re(rul_true, rul_pred)
        metric_ph = prediction_horizon(rul_true, rul_pred)
        metric_alpha_coverage = alpha_coverage(rul_true, rul, rul_weights, .9)

        current_result = pd.DataFrame(data={'Dataset': name, 'RMSE': metric_rmse, 'MAPE': metric_mape, 
        'PH': metric_ph, 'RE_mean': np.mean(metric_re), 'RE_std':np.std(metric_re)}, index=[0])
        df_result_pf = pd.concat([df_result_pf, current_result]).reset_index(drop=True)      

    df_result_reg = pd.DataFrame(columns=cols)
    for name in battery_list:
        file_name = 'result_reg_' + name + '.pickle'
        with open(file_name, 'rb') as f:
            [t, y, threshold, idx_ttf, idx_pred, true_ttf, max_RUL, 
                x_hat, yh, y_bands, rul_mean, rul_bands, rul, x_sample
            ] = pickle.load(f)

        rul_true = (true_ttf-t[idx_pred])*(true_ttf-t[idx_pred]>=0)
        rul_pred = rul_mean[rul_true>0]
        rul_true = rul_true[rul_true>0]

        metric_rmse = rmse(rul_true, rul_pred)
        metric_mape = mape(rul_true, rul_pred)
        metric_re = re(rul_true, rul_pred)
        metric_ph = prediction_horizon(rul_true, rul_pred)
        metric_alpha_coverage = alpha_coverage(rul_true, rul, rul_weights, .9)

        current_result = pd.DataFrame(data={'Dataset': name, 'RMSE': metric_rmse, 'MAPE': metric_mape, 
        'PH': metric_ph, 'RE_mean': np.mean(metric_re), 'RE_std':np.std(metric_re)}, index=[0])
        df_result_reg = pd.concat([df_result_reg, current_result]).reset_index(drop=True)

    print('Particle filter:')
    print(df_result_pf)
    print('\nRegression:')
    print(df_result_reg)
