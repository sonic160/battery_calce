import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime
import pickle


def load_data(battary_list, dir_path):
    ''' 
    This subfunction loads the data for each testing sample. 
    
    Args:
        * battery_list: A list containing the batteries to be read.
        * dir_path: path of the dictionary.
    
    Return:
        * battery: A list of dataframes, where each dataframe contains the test data from the batteries.
    '''
    
    # Do a loop to read the excel files for each battery.
    battery = {}
    for name in battary_list:
        print('Load Dataset ' + name + ' ...')

        # Get the file names of all the excel files for a given battery.
        path = glob.glob(dir_path + name + '/*.xlsx')
        path_sorted = sort_file_names(path)

        # Start reading the useful data from each excel Table, and construct a unified dataframe for each battery.
        
        # Initialize the variables.
        discharge_capacities = []
        charge_capacities = []
        health_indicator = []
        internal_resistance = []
        # Constant Current/Volatage Charging Time.
        ccct = []
        cvct = []
        # Counter of the total cycles.
        cycle_counter = 0

        # Loop for each data Table, which is already sorted chronologically.
        for p in path_sorted:
            # Read the excel file.
            df = pd.read_excel(p, sheet_name=1)
            print('Load ' + str(p) + ' ...')

            # Get the unique values of in the column 'Cycle_Index'.
            cycles = list(set(df['Cycle_Index']))
            # Loop for each cycle.
            for c in cycles:
                # Take all the data points from the cth cycle and store in df_cycle.
                df_cycle = df[df['Cycle_Index'] == c]
                
                # Extract charging phase.
                tmp_charge_capacity, tmp_ccct, tmp_cvct = extract_charging_phase(df_cycle)
                charge_capacities.append(tmp_charge_capacity)
                ccct.append(tmp_ccct)
                cvct.append(tmp_cvct)
                
                # Extract discharging phase.
                tmp_dc, tmp_ir, tmp_hi = extract_discharge_phase(df_cycle)
                discharge_capacities.append(tmp_dc)
                internal_resistance.append(tmp_ir)
                health_indicator.append(tmp_hi)

                # Update the counter of total cycles.
                cycle_counter += 1

        # Output the results in a dataframe.
        charge_capacities = np.array(charge_capacities)
        discharge_capacities = np.array(discharge_capacities)
        health_indicator = np.array(health_indicator)
        internal_resistance = np.array(internal_resistance)
        ccct = np.array(ccct)
        cvct = np.array(cvct)
        df_result = pd.DataFrame({'cycle': np.linspace(1, cycle_counter, cycle_counter),
                                  'charging capacity': charge_capacities,
                                  'CCCT': ccct,
                                  'CVCT': cvct,
                                  'discharging capacity': discharge_capacities,
                                  'SoH': health_indicator,
                                  'resistance': internal_resistance
                                  })
        battery[name] = df_result
    return battery


def sort_file_names(path):
    ''' 
    This subfunction returns a sorted list of the test data, based on the testing date.

    Args:
        * path: A list containing the filenames of the data Tables.
                The Table names have the format of 'CS2_35_XX_XX_XX.xlsx'
    
    Return:
        * path_sorted: A list containing the sorted files.
    '''
    
    # Get the date of experiment for each excel file, and store in dates.
    dates = []
    # Loop for each excel in the path.
    for tmp in path:
        # The date is located before the '.xlsx'.
        end = tmp.rfind('.')
        # Get the third '_' before the '.'.
        # This indicates the start of the dates.
        start = tmp.rfind('_')
        start = tmp.rfind('_', 0, start)
        start = tmp.rfind('_', 0, start)
        # Get a string representing the date.
        date_str = tmp[start+1:end]
        # Store in dates.
        dates.append(datetime.strptime(date_str, '%m_%d_%y'))
    # Sort the different excel files by the date of experiment.
    idx = np.argsort(dates)
    path_sorted = np.array(path)[idx]

    return path_sorted


def extract_charging_phase(df_cycle):
    ''' 
    This subfunction extracts the charging phase from each cycle, and get the CCCT and CVCT.

    Args:
    * df_cycle: A dataframe containing the test data from a cycle.
    
    Return:
    * charge_capacity
    * ccct: Constant current charging time
    * cvct: Constant voltage charging time
    '''

    # Identify the charging phase:
    # Step_Index == 2 is the constant current, while == 4 is the constant voltage charging.
    df_c = df_cycle[(df_cycle['Step_Index'] == 2) | (df_cycle['Step_Index'] == 4)]

    # Handle exception: If some test cycle does not have the charging phase.
    if df_c.shape[0] == 0:
        charge_capacity, ccct, cvct = np.nan, np.nan, np.nan
    else:    
        # Get the charging capacity.
        # Calculate the cumulative discharge capacity at different time using: Q = A*h
        d_t = df_c['Test_Time(s)']
        d_c = df_c['Current(A)']
        capacity_cl = cal_capacity(d_c, d_t)        
        charge_capacity = capacity_cl[-1]
        
        # Calculate the Constant Current/Voltage Charging Time
        df_cc = df_cycle[df_cycle['Step_Index'] == 2]
        df_cv = df_cycle[df_cycle['Step_Index'] == 4]
        ccct = np.max(df_cc['Test_Time(s)']) - np.min(df_cc['Test_Time(s)'])
        cvct = np.max(df_cv['Test_Time(s)']) - np.min(df_cv['Test_Time(s)'])
                
    return charge_capacity, ccct, cvct


def extract_discharge_phase(df_cycle):
    ''' 
    This subfunction extracts the discharging phase from each cycle, and get the discharge capacity, internal resistance and health indicator.
    
    Args:
    * df_cycle: A dataframe containing the test data from a cycle.
    
    Return:
    * discharge_capacity
    * internal_resistance
    * health_indicator: Defined as the discharging capacity when the voltage drops from $3.8V$ to $3.4V$.
    '''
    
    # Extract discharging phase.
    df_d = df_cycle[df_cycle['Step_Index'] == 7]

    # Handle exception: Some test cycle does not have the discharging phase.
    if(df_d.shape[0] == 0):
        discharge_capacity = np.nan
        internal_resistance = np.nan
        health_indicator = np.nan
    else:      
        d_v = df_d['Voltage(V)']
        d_im = df_d['Internal_Resistance(Ohm)']

        # Get the internal resist as the mean value in this cycle.
        internal_resistance = np.mean(np.array(d_im))

        # Health indicator: It is calculated as the discharge capacity in the voltage range [3,8, 3.4].
        # Calculate the cumulative discharge capacity at different time using: Q = A*h
        d_t = df_d['Test_Time(s)']
        d_c = df_d['Current(A)']
        capacity_cl = cal_capacity(d_c, d_t)
        # Get the capacity discharged in the [3.8, 3.4].
        diff_to_start_v = np.abs(np.array(d_v) - 3.8)[1:]
        start = np.array(capacity_cl)[np.argmin(diff_to_start_v)]
        diff_to_end_v = np.abs(np.array(d_v) - 3.4)[1:]
        end = np.array(capacity_cl)[np.argmin(diff_to_end_v)]
        health_indicator = -1 * (end - start)

        # Get the discharge capacity.
        discharge_capacity = -1*capacity_cl[-1]

    return discharge_capacity, internal_resistance, health_indicator


def cal_capacity(d_c, d_t):
    ''' 
    This subfunction calculates the capacity based on $Q(t) = \int_0^t I(t) dt.$
    
    Args:
    * d_t: The time.
    * d_c: The correpsonding current.
    
    Return: 
    * capacity_cl: Calculated capacity.
    '''

    time_diff = np.diff(list(d_t))
    d_c = np.array(list(d_c))[1:]
    capacity_cl = time_diff*d_c/3600 
    capacity_cl = [np.sum(capacity_cl[:n]) for n in range(capacity_cl.shape[0])]

    return capacity_cl


def drop_outlier_rolling(df_data, window):
    ''' 
    This subfunction remove the outliers based on the mean and std in a rolling window.
    
    Args:
    * df_data: The time series that needs to be detected.
    * window: size of the rolling window.

    Return:
    * index: The index of the abnormal data points.
    '''

    avg = df_data.rolling(window, closed='left').mean()
    std = df_data.rolling(window, closed='left').std()
    upper = avg + 3*std
    lower = avg - 3*std
    index = df_data.index[(df_data < lower) | (df_data > upper)]

    return index


def drop_outlier_sw(data, window):
    ''' 
    This subfunction remove the outliers based on the mean and std in a sliding window.
    
    Args:
    * df_data: The time series that needs to be detected.
    * window: size of the rolling window.

    Return:
    * index: The index of the normal data points.
    '''

    index = []
    range_ = np.arange(1, len(data), window)
    for i in range_[:-1]:
        array_lim = data[i:i+window]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max, th_min = mean + sigma*2, mean - sigma*2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))

    return np.array(index)


def cal_ttf(cycle, data, threshold):
    ''' 
    This subfunction calculate the true lifetime of a trajectory. The lifetime is defined as concecutively $n$ cycles below the threshold.
    
    Args:
    * cycle: The cycle number.
    * data: The degradation trajectory.
    * Threshold: Failure threshold.

    Return:
    * ttf: True lifetime. '''

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

    return ttf

# This is for debuging the utility function.
if __name__ == '__main__':
    battery_list = {'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38'}
    # dir_path = r'C:\Users\Zhiguo\OneDrive - CentraleSupelec\Code\Python\battery_calce\data\\'
    # Battery = load_data(battery_list, dir_path)

    # Directly read from the archived data.
    with open('data_all.pickle', 'rb') as f:
        data_all = pickle.load(f)

    battery = data_all['CS2_38']
    x = battery['cycle']
    y = battery['discharging capacity']
    idx = drop_outlier_sw(y, 20)
    x = x[idx]
    y = y[idx]

    cal_ttf(x, y, .7*1.1)