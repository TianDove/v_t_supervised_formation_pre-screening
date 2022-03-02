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
import optuna
import pickle

import file_operation
import functional
from cls_data_set import ClsDataSetCreator
import cls_model_define
import cls_opt
import cls_run


def objective(trial: optuna.trial.Trial, data_set_dict: dict, writer: SummaryWriter):
    """"""
    # preprocess set up
    in_len = 80
    tokenize_para = {
        't_len': 40,
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
    n_epochs = 3
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
        't_len': tokenize_para['t_len'],
        'd_model': 256,
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
        'amsgrad': True,
    }

    scheduler_type = trial.suggest_categorical('scheduler_type', ['StepLR',
                                                                  'ExponentialLR',
                                                                  'Noam',
                                                                  'CosineAnnealingLR',
                                                                  'CosineAnnealingWarmRestarts',
                                                                  'CyclicLR-triangular2',
                                                                  'OneCycleLR-cos'
                                                                  ])

    if scheduler_type == 'StepLR':
        scheduler_para = {
            'scheduler name': 'StepLR',
            'step_size': 10,
            'gamma': 0.95,
            'last_epoch': -1,
            'verbose': False
        }
    elif scheduler_type == 'ExponentialLR':
        scheduler_para = {
            'scheduler name': 'ExponentialLR',
            'gamma': 0.98,
            'last_epoch': -1,
            'verbose': False
        }
    elif scheduler_type == 'Noam':
        scheduler_para = {
            'scheduler name': 'Noam',
            'model_size': 256,
            'factor': 2,
            'warmup': 96
        }
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler_para = {
            'scheduler name': 'CosineAnnealingLR',
            'T_max': 16,
            'eta_min': 0.0,
            'last_epoch': -1,
            'verbose': False
        }
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler_para = {
            'scheduler name': 'CosineAnnealingWarmRestarts',
            'T_0': 16,
            'T_mult': 1,
            'eta_min': 0.0,
            'last_epoch': -1,
            'verbose': False
        }
    elif scheduler_type == 'CyclicLR-triangular2':
        scheduler_para = {
            'scheduler name': 'CyclicLR',
            'base_lr': 0.001,
            'max_lr': 0.1,
            'step_size_up': 2000,
            'step_size_down': None,
            'mode': 'triangular2',
            'gamma': 1.0,
            'scale_fn': None,
            'scale_mode': 'cycle',
            'cycle_momentum': False,
            'base_momentum': 0.85,
            'max_momentum': 0.95,
            'last_epoch': -1,
            'verbose': False
        }
    elif scheduler_type == 'OneCycleLR-cos':
        scheduler_para = {
            'scheduler name': 'OneCycleLR',
            'max_lr': 0.1,
            'total_steps': None,
            'epochs': n_epochs,
            'steps_per_epoch': len(train_data_set),
            'pct_start': 0.3,
            'anneal_strategy': 'cos',
            'cycle_momentum': False,
            'base_momentum': 0.85,
            'max_momentum': 0.95,
            'div_factor': 25,
            'final_div_factor': 1e4,
            'three_phase': False,
            'last_epoch': -1,
            'verbose': False
        }
    else:
        raise ValueError('Scheduler Type Error.')


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
    m_cls_run = cls_run.ClsRun(data_time_str,
                               data_set_dict,
                               n_epochs,
                               curr_model,
                               CRITERION,
                               writer,
                               op_sch_dict['optimizer'],
                               scheduler=op_sch_dict['scheduler'],
                               device=device,
                               hyper_para_dict=hyper_para_dict,
                               trial=trial)
    m_cls_run.run()
    return max(m_cls_run.epoch_val_accu_list)


if __name__ == '__main__':
    mp.freeze_support()
    ###################################################################################################################
    torch.manual_seed(42)  # fix random seed
    ###################################################################################################################
    # path set up
    workspace_base = 'D:\\workspace\\PycharmProjects'
    log_dir = os.path.join(workspace_base, 'optuna_run')
    data_dir = os.path.join(workspace_base, 'battery_dataset')
    data_file_name = 'small_std_imbd'
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
    DEBUG = True
    n_trials = 10
    data_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if DEBUG:
        writer_dir = os.path.join(log_dir, 'DEBUG', f'{data_time_str}')
    else:
        writer_dir = os.path.join(log_dir, f'{data_time_str}')

    with SummaryWriter(log_dir=writer_dir) as writer:

        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.HyperbandPruner()

        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(lambda trial: objective(trial, data_set_dict, writer),
                       n_trials=n_trials)
        # get trials result
        exp_res = study.trials_dataframe()
        exp_res.to_csv(os.path.join(writer_dir, f'{data_time_str}_Trials_DataFrame.csv'))

        # save study
        with open(os.path.join(writer_dir, f'{data_time_str}_Study.pkl'), 'wb') as f:
            pickle.dump(study, f)
    sys.exit()

