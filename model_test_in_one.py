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


def model_test(data_set_dict: dict,
               model_dict: dict,
               device: torch.device,
               writer_path: str,
               time_str: str):
    """"""
    # preprocess set up
    CRITERION = nn.CrossEntropyLoss().to(device=device)
    in_len = model_dict['in_len']
    tokenize_para = {
        't_len': model_dict['t_len'],
        'is_overlap': False,
        'step': 16,
    }
    n_token = functional.cal_n_token(in_len, **tokenize_para)

    if 'MyModel' in model_dict['model_name']:
        transformation = (functional.cut_data_with_label,
                          functional.tokenize_arr,
                          )

        transformation_para = (
            {'out_len': in_len},  # (out_len, )
            tokenize_para,  # (t_len, is_overlap, step)
        )
    else:
        transformation = (functional.cut_data_with_label,
                          #  functional.tokenize_arr,
                          )

        transformation_para = (
            {'out_len': in_len},  # (out_len, )
            #  tokenize_para,  # (t_len, is_overlap, step)
        )

    preprocess_data_dict = functional.classification_preprocess(data_set_dict,
                                                                transformation,
                                                                transformation_para)
    ###################################################################################################################
    # data set set up
    n_epochs = 512
    batch_size = model_dict['batch_size']
    n_worker = 3
    is_shuffle = True
    test_batch_size = 256
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
    # optimizer and scheduler set up
    op_sch_dict = {
        'type_optimizer': None,
        'type_scheduler': None,
        'op_init_dict': None,
        'sch_init_dict': None,
        'optimizer': model_dict['optimizer'],
        'scheduler': model_dict['scheduler']
    }
    ###################################################################################################################
    with SummaryWriter(log_dir=writer_path) as writer:
        m_cls_run = cls_run.ClsRun(time_str,
                                   data_set_dict,
                                   n_epochs,
                                   model_dict['model'],
                                   CRITERION,
                                   writer,
                                   op_sch_dict['optimizer'],
                                   scheduler=op_sch_dict['scheduler'],
                                   device=device,
                                   hyper_para_dict=None)
        m_cls_run.test_run(model_dict['epoch'], test_batch_size)


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
    DEBUG = False
    if USE_GPU:
        device = functional.try_gpu()
    else:
        device = torch.device('cpu')
    ###################################################################################################################
    # model set up
    model_base_path = os.path.join(workspace_base, 'model_run', 'Runs')
    model_list = os.listdir(model_base_path)
    for model_name in model_list:
        model_fold_file_list = os.listdir(os.path.join(model_base_path, model_name))
        temp_selected_model_file_list = []
        for file in model_fold_file_list:
            sp_text = os.path.splitext(file)
            if sp_text[-1] == '.pt':
                temp_selected_model_file_list.append(sp_text[0])

        sorted_file_list = sorted(temp_selected_model_file_list, key=lambda file_name: int(file_name.split('_')[-1]))
        for n_file in sorted_file_list:
            time_str = n_file.split('_')[2]
            model_path = os.path.join(model_base_path, model_name, f'{n_file}.pt')
            model_dict = torch.load(model_path)

            #  set up file path
            if DEBUG:
                writer_dir = os.path.join(log_dir, 'DEBUG', f'{model_name}')
            else:
                writer_dir = os.path.join(log_dir, 'Tests', f'{model_name}')

            # start evl
            model_test(data_set_dict, model_dict, device, writer_dir, time_str)

    sys.exit()

