# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 09:13:08 2021

@author: roman
"""


import numpy as np
import matplotlib.pyplot as plt
import itertools as it

power_dataset = "power_data.txt"
ecg_dataset = "ecg_data.txt"


def load_power():
    with open(power_dataset, 'r') as input_file:
        data = input_file.read()
        data_lines = data.split('\n')
        return np.array(data_lines[:-1]).astype(np.int)
    

def load_ecg():
    with open(ecg_dataset, 'r') as input_file:
        data = input_file.read()
        data_lines = data.split('\n')
        for lines in data_lines:
            return np.array( [np.array(lines[2:].split()).astype(float) for lines in data_lines[:-1]] )

power_d = load_power()

ecg_d = load_ecg()

ecg_article = ecg_d[:,1]

"""
fig, (ax1, ax2) = plt.subplots(2)

fig.suptitle('Time series')

ax1.plot(np.arange(np.shape(power_d)[0]) ,power_d)

ax2.plot(np.arange(np.shape(ecg_article)[0]), ecg_article)
"""


minutes_in_day = 24 * 60
fifteen_intervall_in_one_day = int(minutes_in_day / 15)

interval = np.arange(0, np.shape(power_d)[0], fifteen_intervall_in_one_day).astype(int)


power_d_in_day = np.array_split(power_d, interval)

week_one = [power_d_in_day[i] for i in range(237,244)]
week_two = [power_d_in_day[i] for i in range(118, 125)]

power_d_week_one = np.concatenate(week_one)
power_d_week_20 = np.concatenate(week_two)

"""
fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True)

fig.suptitle('Time series of power dataset')

ax1.plot(np.arange(np.shape(power_d_week_one)[0]) ,power_d_week_one)
ax1.set_title('Week one')
ax1.set_ylabel('Consumption (kW)')
ax1.set_xlabel('Timestamp in day')


ax2.plot(np.arange(np.shape(power_d_week_20)[0]), power_d_week_20)
ax1.set_title('Week twenty')
ax1.set_ylabel('Consumption (kW)')
ax1.set_xlabel('Timestamp in day')
"""

def STR(time_series):
    v2 = np.median(np.unique(time_series))
    print(v2)
    
    def test_str(t):
        if t < v2:
            return 1
        else:
            return 2
    vtest = np.vectorize(test_str)
    discrete_time_series = vtest(time_series)
    
    return [ (key, sum(1 for _ in group)) for key, group in it.groupby(discrete_time_series)], discrete_time_series
    

str_power_d, dis_ts = STR(power_d)

print(str_power_d)

plt.scatter(np.arange(np.shape(power_d[:1000])[0]) ,power_d[:1000], c = dis_ts[:1000])
