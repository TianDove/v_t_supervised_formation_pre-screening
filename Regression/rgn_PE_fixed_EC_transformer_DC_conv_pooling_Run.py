#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang

import sys
import os
import datetime
import torch
import torch.nn as nn
from opt import optimizer_select_and_init
from torch.utils.tensorboard import SummaryWriter

import run
import functional
import rgn_model_define
from Regression import data_set_creat

# writer log path
log_dir_base = 'D:\\workspace\\PycharmProjects\\model_run\\'
#################################################
USE_GPU = False
DEBUG = False
CH1_VOLTAGE_LEN = 161
CRITERION = nn.MSELoss()
in_len = 80
tokenize_tup = (20, False, 16)
n_token = functional.cal_n_token(in_len, *tokenize_tup)
n_epochs = 3
batch_size = 16
#################################################
# data set setup
data_file_path = 'D:\\workspace\\PycharmProjects\\battery_dataset\\2600P-01_DataSet\\data_set'
data_file_name = '20210721-152403_small_ch1_form-ocv'
data_file_type = '.pt'
transformation = (functional.cut_sequence,
                  functional.tokenize)
transformation_para = (
    (in_len,),  # (out_len, )
    tokenize_tup,  # (t_len, is_overlap, step)
)
####################################################################################################
if USE_GPU:
    device = functional.try_gpu()
else:
    device = torch.device('cpu')
####################################################################################################

net = rgn_model_define.PE_fixed_EC_transformer_DC_conv_pooling(tokenize_tup[0],
                                                               n_token,
                                                               nhd=4,
                                                               nly=6,
                                                               dropout=0.1,
                                                               hid=2048).to(device=device)
op_dict = optimizer_select_and_init(net,
                                    'Adam',
                                    'Noam',
                                    op_init_dict={'lr': 0.001,
                                     'betas': (0.9, 0.98),
                                     'eps': 1e-9,
                                     'weight_decay': 0,
                                     'amsgrad': False},
                                    sch_init_dict={
                                        'model_size': 512,
                                        'factor': 2,
                                        'warmup': 4000
                                    })
#####################################################################################################
data_path = os.path.join(data_file_path, data_file_name + data_file_type)
train_dataset, train_n_batch = data_set_creat.Ch1_FormOcv_DataSetCreator.creat_dataset(data_path,
                                                                                       'train',
                                                                                       bsz=batch_size,
                                                                                       is_shuffle=False,
                                                                                       num_of_worker=0,
                                                                                       transform=transformation,
                                                                                       trans_para=transformation_para)
val_dataset, val_n_batch = data_set_creat.Ch1_FormOcv_DataSetCreator.creat_dataset(data_path,
                                                                                   'val',
                                                                                   bsz=1,
                                                                                   is_shuffle=False,
                                                                                   num_of_worker=0,
                                                                                   transform=transformation,
                                                                                   trans_para=transformation_para)
test_dataset, test_n_batch = data_set_creat.Ch1_FormOcv_DataSetCreator.creat_dataset(data_path,
                                                                                     'test',
                                                                                     bsz=1,
                                                                                     is_shuffle=False,
                                                                                     num_of_worker=0,
                                                                                     transform=transformation,
                                                                                     trans_para=transformation_para)
dataset_profile_dic = {
    'train_data_set': train_dataset,
    'num_of_train_batch': train_n_batch,
    'val_data_set': val_dataset,
    'num_of_val_batch': val_n_batch,
    'test_data_set': test_dataset,
    'num_of_test_batch': test_n_batch
}
############################################################
model_name = net.__class__.__name__
data_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if DEBUG:
    log_dir = log_dir_base + '\\DEBUG\\' + f'\\{model_name}_{data_time_str}\\'
else:
    log_dir = log_dir_base + f'\\{model_name}_{data_time_str}\\'
print(f'Model Save Path: {log_dir}')

# tensorboard summary define
with SummaryWriter(log_dir=log_dir) as writer:
    # init Run
    m_run = run.Run(dataset_profile_dic,
                    net,
                    CRITERION,
                    writer,
                    op_dict['optimizer'],
                    op_dict['scheduler'],
                    device=device)
    m_run.run(n_epochs)
sys.exit(0)
