#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang


import sys
import os
import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import file_operation
from sklearn.preprocessing import StandardScaler




"""""
import matplotlib
import matplotlib.pyplot as plt
import seaborn
a_size = attn_output_weights.shape
attn = attn_output_weights.to(torch.device('cpu')).numpy()
n_axs = a_size[0]
fig, ax = plt.subplots(ncols=n_axs)
for index in range(n_axs):
    temp_a = attn[index, :, :]
    seaborn.heatmap(temp_a, square=True, vmin=0.0, vmax=1.0, 
                    ax=ax[index], cbar=False, xticklabels=False, yticklabels=False)
    ax[index].set_title(f'Head {index + 1}')
fig.suptitle('1-th Encoder Layer', y=0.7)
fig.show()
"""""


if __name__ == '__main__':

    basedir = 'D:\\workspace\\PycharmProjects\\battery_dataset'
    file_name = '20210813-111913_ch1_form-ocv_std__label-encoded__imbd_'
    data_type = '.pt'
    data_path = os.path.join(basedir, file_name + data_type)
    data_dict = file_operation.load_dic_in_pickle(data_path)[file_name]
    test_df = data_dict['test df'].astype('float')
    data = test_df.iloc[:, 0:-1]
    un_std_data = data_dict['std_op'].inverse_transform(data)

    # cut_data = un_std_data[:, 0:80]
    cut_data = un_std_data[:, 0:-1]
    label = test_df.iloc[:, -1].to_numpy()
    fig, ax = plt.subplots()
    n_rows = cut_data.shape[0]
    x_dim = [x for x in range(cut_data.shape[1])]
    for rows in range(n_rows):
        if label[rows] == 0.0:
            clr = 'r-'
        elif label[rows] == 1.0:
            clr = 'g-'
        else:
            raise ValueError
        ax.plot(x_dim, cut_data[rows, :], clr)
    fig.show()
    sys.exit(0)

"""fig.clf()
ax.plot(time, vol, linewidth=2)
ax.minorticks_on()
fig.show()
ax.set_xlabel(xlabel='Time(min)')
ax.set_ylabel(ylabel='Voltage(V)')
Axes.set_title('CVC')"""