# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 09:13:08 2021

@author: roman
"""


import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import improved_Sheather_Jones as isj
from scipy import ndimage
import sys

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

# Affichage des données pour ECG

ecg_article = ecg_d[3588:4975,1]

"""
plt.title("Anomalous ECG")

plt.xlabel("observation number")
plt.ylabel("electrical potential")

plt.plot(np.arange(np.shape(ecg_article)[0]), ecg_article)
"""


# Affichage des données complète des deux datasets

"""
fig, (ax1, ax2) = plt.subplots(2)

fig.suptitle('Time series')

ax1.plot(np.arange(np.shape(power_d)[0]) ,power_d)

ax2.plot(np.arange(np.shape(ecg_article)[0]), ecg_article)
"""



# Affichage de deux semaines, une semaine normal et une semaine anomalie, du dataset Power

def get_power_two_week(nb_week):
    
    minutes_in_day = 24 * 60
    fifteen_intervall_in_one_day = int(minutes_in_day / 15)
    
    interval = np.arange(0, np.shape(power_d)[0], fifteen_intervall_in_one_day).astype(int)
    
    
    power_d_in_day = np.array_split(power_d, interval)
    
    inf_bounds = nb_week * 7 - 1
    sup_bounds = nb_week * 7 + 6
    if inf_bounds < 0:
        inf_bounds = 0
    if sup_bounds >= len(power_d_in_day):
        sup_bounds = len(power_d_in_day) - 1
       
    
    week = [power_d_in_day[i] for i in range(inf_bounds, sup_bounds)]
    
    return np.concatenate(week)




power_d_week_one = get_power_two_week(1)
power_d_week_20  = get_power_two_week(17)


"""
fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True)

fig.suptitle('Time series of power dataset')

ax1.plot(np.arange(np.shape(power_d_week_one)[0]) ,power_d_week_one)
ax1.set_title('Week one')
ax1.set_ylabel('Consumption (kW)')
ax1.set_xlabel('Timestamp in day')


ax2.plot(np.arange(np.shape(power_d_week_20)[0]), power_d_week_20)
ax2.set_title('Week sixteen')
ax2.set_ylabel('Consumption (kW)')
ax2.set_xlabel('Timestamp in day')
"""



def STR(time_series):
    v2 = np.median(np.unique(time_series))
    
    def test_str(t):
        if t < v2:
            return t,1
        else:
            return t,2
    vtest = np.vectorize(test_str)
    v_ts, bins_ts = vtest(time_series)
    
    return [ (sum(1 for _ in group), key) for key, group in it.groupby(bins_ts)], v_ts, bins_ts
    


"""
fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True)

fig.suptitle('Sojourn Time Representation')

ax1.set_title('Week one')
ax1.set_ylabel('Consumption (kW)')
ax1.set_xlabel('Timestamp in day')
ax1.scatter(np.arange(np.shape(power_d_week_one)[0]) ,power_d_week_one, c = dis_ts_one)

ax2.set_title('Week sixteen')
ax2.set_ylabel('Consumption (kW)')
ax2.set_xlabel('Timestamp in day')
ax2.scatter(np.arange(np.shape(power_d_week_20)[0]) ,power_d_week_20, c = dis_ts_two)
"""

def PDF(example, data, h):
    factor = 1 / (h * np.shape(data)[0])
    return factor * np.sum( ndimage.gaussian_filter1d((example - data) / h, h))

def make_clusters(train_set):
    
    
    st_bin_one = [ run_length for run_length, bins in train_set if bins == 1 ]
    st_bin_two = [ run_length for run_length, bins in train_set if bins == 2 ]
    
    def compute_cluster(st_bin):
        
        h = isj.hsj(st_bin)
    
    
        
        
        modes = [PDF(x, st_bin, h) for x in st_bin]
        
        res1 = dict()
        res2 = []
        for i in np.arange(len(st_bin)):
            diff = np.abs(modes - st_bin[i])
            cluster = np.argmin(diff)
            res2.append( (st_bin[i], cluster))
            if cluster in res1:
                res1[cluster].append(st_bin[i])
            else:
                res1[cluster] = [st_bin[i]]
        
        
        return res1, res2, h
    
    return compute_cluster(st_bin_one), compute_cluster(st_bin_two)
        


def training(train_set, show_cluster = False):
    
    str_power_d_one, v_ts, dis_ts_one = STR(train_set)
    np_str_power = np.array(str_power_d_one)
    info_cluster = make_clusters(np_str_power)
    
    def mining_sojourn_interval(cluster, st_cluster, h):
        min_st = min(st_cluster)
        max_st = max(st_cluster)
        
        inf_bound = min_st
        sup_bound = max_st
        
        
        i = 0
        while True:
            if i > 100:
                max_si = sup_bound
                break
            max_si = sup_bound + i
            st_cluster_g = st_cluster + [max_si]
            modes = [PDF(x, st_cluster_g, h) for x in st_cluster_g]
            test = False
            for m in modes:
                if np.abs(m - max_si) < 2:
                    test = True
                    
            if test:
                break
            i += 1
        
        i = 0
        while True:
            min_si = inf_bound - i
            if min_si <= 1:
                min_si = inf_bound
                break
            st_cluster_g = st_cluster + [min_si]
            modes = [PDF(x, st_cluster_g, h) for x in st_cluster_g]
            test = False
            for m in modes:
                if np.abs(m - min_si) < 2:
                    test = True
                    
            if test:
                break
            i += 1
        
        return min_si, max_si
    
    si_bins_ones = []
    
    for k, v in info_cluster[0][0].items():
        si_bins_ones.append((k, mining_sojourn_interval(k, v, info_cluster[0][2])))
        
    
    si_bins_two = []
    
    for k, v in info_cluster[1][0].items():
        si_bins_two.append((k, mining_sojourn_interval(k, v, info_cluster[1][2])))
            
    
    if show_cluster:
        cluster_b_one = info_cluster[0][1]

        cluster_b_two = info_cluster[1][1]
        
        
        i = 0
        j = 0
        res = [-1 for _ in range(len(dis_ts_one))]
        len_train_set = len(train_set)
        
        while i < len_train_set:
            
            if j >= len(cluster_b_one):
                break
            run_length, c = cluster_b_one[j]

            while i < len_train_set and dis_ts_one[i] == 1:
                res[i] = c
                i += 1
                
            while i < len_train_set and dis_ts_one[i] != 1:
                i += 1
            
            j += 1
          
        i = 0
        j = 0
        while i < len_train_set:
            
            if j >= len(cluster_b_two):
                break
            run_length, c = cluster_b_two[j]
            
           
            
            while i < len_train_set and dis_ts_one[i] != 2:
                i += 1
            
            while i < len_train_set and dis_ts_one[i] == 2:
                res[i] = c
                i += 1
            
            j += 1
        
        j = 0
        for v in np.unique(res):
            for i in np.arange(len(res)):
                if res[i] == v:
                    res[i] = j
            j += 1
        
        fig, ax = plt.subplots()
        
        fig.suptitle('Training')
        ax.set_ylabel('Consumption (kW)')
        ax.set_xlabel('Timestamp in day')
        scatter = ax.scatter(np.arange(np.shape(v_ts)[0]) ,v_ts, c = res)
        
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(), title= "Cluster")
        ax.add_artist(legend1)
        
        plt.show()
    
    return si_bins_ones, si_bins_two


def predict(sojourn_interval, test_set):
    res = [-20 for _ in range(len(test_set))]
    str_power_d_one, v_ts, dis_ts_one = STR(test_set)
    print(str_power_d_one)
    
    si_bins_1, si_bins_2 = sojourn_interval
    
    i = 0
    sum_ = 0
    for run_length, bins in str_power_d_one:
        sum_ += run_length
        if bins == 1:
            for cl, si in si_bins_1:
                si_min, si_max = si
                if run_length >= si_min and run_length <= si_max:
                    for j in range(i, i + run_length):
                        res[j] = cl
        elif bins == 2:
            for cl, si in si_bins_2:
                si_min, si_max = si
                if run_length >= si_min and run_length <= si_max:
                    for j in range(i, i + run_length):
                        res[j] = cl
        i += run_length
    

    return res



train_set = power_d[510:3201]
sojourn_interval = training(train_set)

print(sojourn_interval)
print()
print()


test_set = power_d[5850:9920]
res = predict(sojourn_interval, test_set)

j = -1
for v in np.unique(res):
    for i in np.arange(len(res)):
        if res[i] == v:
            res[i] = j
    j += 12

fig, ax = plt.subplots()
        
fig.suptitle('Evaluation Test')
ax.set_ylabel('Consumption (kW)')
ax.set_xlabel('Timestamp in day')
scatter = ax.scatter(np.arange(len(test_set)), test_set, c = res)

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(), title= "Cluster")

ax.add_artist(legend1)

plt.show()


"""
train_set = ecg_d[:1477, 1]


#plt.plot(np.arange(np.shape(train_set)[0]), train_set)


sojourn_interval = training(train_set)



test_set = ecg_d[3000:4949, 1]
res = predict(sojourn_interval, test_set)


fig, ax = plt.subplots()
        
fig.suptitle('Evaluation Test')
ax.set_ylabel('Electrical potential')
ax.set_xlabel('Observation number')
scatter = ax.scatter(np.arange(len(test_set)), test_set, c = res)

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(), title= "Cluster")

ax.add_artist(legend1)

plt.show()
"""




