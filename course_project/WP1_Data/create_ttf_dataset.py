import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def cal_ttf(cycle, data, threshold):
    ''' 
    This subfunction calculate the true lifetime of a trajectory. The lifetime is defined as concecutively $n$ cycles below the threshold.
    
    Args:
    * cycle: The cycle number.
    * data: The degradation trajectory.
    * Threshold: Failure threshold.

    Return:
    * ttf: True lifetime. 
    * idx_ttf: The index of the ttf. '''

    # Define how many consecutive points needed.
    consecutive_values = 3
    length = len(data)
    # Transform into numpy array.
    diff = np.array(data - threshold)
    cycle = np.array(cycle)
    # Search.
    counter = consecutive_values
    for i in range(length):
        if counter == 0:
            break
        else:
            if diff[i] < 0:
                counter -= 1
            else:
                counter = consecutive_values
    ttf = cycle[i-1]

    return ttf, i-1


def get_ttf():
    battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    # Directly read from the archived data.
    with open('data_all.pickle', 'rb') as f:
        data_all = pickle.load(f)

    ttf = np.zeros(6)
    for i in range(len(battery_list)):
        name = battery_list[i]
        battery = data_all[name]
        battery.fillna(method='ffill', inplace=True)
        # Get the time and degradation measurement. Perform filtering.
        t = battery['cycle']
        y = battery['discharging capacity']
        t = np.array(t)
        y = np.array(y)
        ttf[i], _ = cal_ttf(t, y, .7*1.1)

    battery_list = ['CS2_33', 'CS2_34']
    # Directly read from the archived data.
    with open('data_all_halfC.pickle', 'rb') as f:
        data_all = pickle.load(f)

    for name in battery_list:
        i += 1
        battery = data_all[name]
        battery.fillna(method='ffill', inplace=True)
        # Get the time and degradation measurement. Perform filtering.
        t = battery['cycle']
        y = battery['discharging capacity']
        t = np.array(t)
        y = np.array(y)
        ttf[i], _ = cal_ttf(t, y, .7*1.1)

    return ttf


if __name__ == '__main__':
    ttf = get_ttf()

    sample_size = 1000
    ttf_data = np.random.normal(loc=0.0, scale=20, size=sample_size)
    ttf_data += np.random.choice(ttf, size=sample_size)
    
    plt.hist(ttf_data)
    plt.show()

    ttf_data = pd.DataFrame(ttf_data, columns=['Time to failure data'])
    ttf_data.to_csv('course_project/ttf_data.csv')
    

