#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang

import sys
import os
import datetime
import multiprocessing as mp
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import file_operation
import functional
from cls_data_set import ClsDataSetCreator
import cls_model_define
import cls_opt
import cls_run


if __name__ == '__main__':
    mp.freeze_support()
    ###################################################################################################################
    torch.manual_seed(42)  # fix random seed
    ###################################################################################################################
    # path set up
    workspace_base = 'D:\\workspace\\PycharmProjects'
    log_dir = os.path.join(workspace_base, 'model_run')
    data_dir = os.path.join(workspace_base, 'battery_dataset')
    data_file_name = '20210813-111913_ch1_form-ocv_std__label-encoded__imbd_'
    data_file_type = '.pt'
    data_set_dict = file_operation.load_dic_in_pickle(os.path.join(data_dir,
                                                                   data_file_name + data_file_type))[data_file_name]
    ###################################################################################################################
    # device set up
    USE_GPU = True
    if USE_GPU:
        device = functional.try_gpu()
    else:
        device = torch.device('cpu')
    ###################################################################################################################
    # preprocess set up
    in_len = 80
    tokenize_para = {
        't_len': 16,
        'is_overlap': False,
        'step': 16,
    }
    n_token = functional.cal_n_token(in_len, **tokenize_para)
    transformation = (functional.cut_data_with_label,
                      functional.tokenize_arr,
    )
    transformation_para = (
        {'out_len': in_len},  # (out_len, )
        tokenize_para,  # (t_len, is_overlap, step)
    )
    preprocess_data_dict = functional.classification_preprocess(data_set_dict,
                                                                transformation,
                                                                transformation_para)
    ###################################################################################################################
    # data set set up
    n_epochs = 512
    batch_size = 128
    n_worker = 3
    is_shuffle = True
    test_batch_size = 1024
    train_data_set = ClsDataSetCreator.creat_dataset(preprocess_data_dict['train df'],
                                                     bsz=batch_size,
                                                     is_shuffle=is_shuffle,
                                                     num_of_worker=n_worker)
    val_data_set = ClsDataSetCreator.creat_dataset(preprocess_data_dict['val df'],
                                                   bsz=test_batch_size,
                                                   is_shuffle=is_shuffle,
                                                   num_of_worker=n_worker)
    test_data_set = ClsDataSetCreator.creat_dataset(preprocess_data_dict['test df'],
                                                    bsz=test_batch_size,
                                                    is_shuffle=is_shuffle,
                                                    num_of_worker=n_worker)
    data_set_dict = {
        'train_data_set': train_data_set,
        'val_data_set': val_data_set,
        'test_data_set': test_data_set
    }
    ###################################################################################################################
    # model set up
    CRITERION = nn.CrossEntropyLoss().to(device=device)
    MODEL = cls_model_define.MyModel
    model_init_dict = {
        'd_model': tokenize_para['t_len'],
        'n_token': n_token,
        'nhd': 4,
        'nly': 3,
        'dropout': 0.0,
        'hid': 256
    }
    curr_model = MODEL.init_model(model_init_dict).to(device=device)
    ###################################################################################################################
    # optimizer and scheduler set up
    optimizer_para = {
        'optimizer_name': 'Adam',
        'lr': 0.0001,
        'betas': (0.9, 0.98),
        'eps': 1e-9,
        'weight_decay': 0,
        'amsgrad': False,
    }
    scheduler_para = {
        'scheduler name': 'StepLR',
        'step_size': 10,
        'gamma': 0.95,
        'last_epoch': -1,
        'verbose': False
    }
    op_sch_dict = cls_opt.optimizer_select_and_init(curr_model, optimizer_para, scheduler_para)
    optimizer_para.pop('betas')  # del for hyper-parameter logging
    ###################################################################################################################
    # get hyper parameter dict
    hyper_para_dict = {
        'data_set': data_file_name,
        'in_len': in_len,
        **tokenize_para,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'is_shuffle': is_shuffle,
        'n_worker': n_worker,
        'device_idx': device.index,
        'device': device.type,
        'model_name': curr_model.model_name,
        **model_init_dict,
        'loss_fn': CRITERION.__class__.__name__,
        **optimizer_para,
        **scheduler_para
    }
    ###################################################################################################################
    DEBUG = False
    data_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if DEBUG:
        writer_dir = os.path.join(log_dir, 'DEBUG', f'{curr_model.model_name}_{data_time_str}')
    else:
        writer_dir = os.path.join(log_dir, f'{curr_model.model_name}_{data_time_str}')

    with SummaryWriter(log_dir=writer_dir) as writer:
        m_cls_run = cls_run.ClsRun(data_time_str,
                                   data_set_dict,
                                   n_epochs,
                                   curr_model,
                                   CRITERION,
                                   writer,
                                   op_sch_dict['optimizer'],
                                   scheduler=op_sch_dict['scheduler'],
                                   device=device,
                                   hyper_para_dict=hyper_para_dict)
        m_cls_run.run()
    sys.exit()

