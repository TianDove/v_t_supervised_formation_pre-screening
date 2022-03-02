#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang

import time
import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as skl_mtr

import file_operation

class Run():
    """"""
    def __init__(self,
                 data_set_dic: dict,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 writer: SummaryWriter,
                 optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler = None,
                 device: torch.device = torch.device('cpu'),
                 is_cls: int = None,
                 hyper_para_dict: dict = None) -> None:
        """"""
        # get train dataset
        self.train_data_set = data_set_dic['train_data_set']
        self.n_train_batch = data_set_dic['num_of_train_batch']

        # get validation dataset
        self.val_data_set = data_set_dic['val_data_set']
        self.n_val_batch = data_set_dic['num_of_val_batch']

        # get test dataset
        self.test_data_set = data_set_dic['test_data_set']
        self.n_test_batch = data_set_dic['num_of_test_batch']

        # save input
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.is_cls = is_cls
        self.hyper_para_dict = hyper_para_dict

        # Tensorboard Writer
        self.writer = writer

        # save dir
        self.save_dir = self.writer.log_dir

        # get model name
        self.model_name = model.__class__.__name__

        # record loss
        self.batch_train_loss = 0.0
        self.epoch_train_loss = 0.0
        self.epoch_val_loss = 0.0
        self.test_loss = 0.0
        self.epoch_val_list = []
        self.test_list = []
        self.val_accu = 0.0
        self.test_accu = 0.0

        # record time
        self.epoch_start_time = 0.0
        self.epoch_train_start_time = 0.0
        self.batch_train_start_time = 0.0
        self.batch_train_end_time = 0.0
        self.epoch_train_end_time = 0.0
        self.epoch_val_start_time = 0.0
        self.epoch_val_end_time = 0.0
        self.epoch_end_time = 0.0
        self.test_start_time = 0.0
        self.test_end_time = 0.0

        # current loop state
        self.n_epoch = None
        self.curr_epoch = None
        self.curr_batch = None
        self.curr_stage = None

    def run(self, n_epoch):
        """"""
        self.n_epoch = n_epoch
        for epoch in range(n_epoch):
            self.epoch_start_time = time.time()
            self.curr_epoch = epoch

            # training
            self.epoch_train_loss = 0.0
            self.epoch_train_start_time = self.epoch_start_time
            self.epoch_train_loss = self.train_step()
            self.epoch_train_end_time = time.time()

            # validation
            self.epoch_val_loss = 0.0
            self.epoch_val_start_time = time.time()
            self.epoch_val_list = []
            self.epoch_val_loss, self.epoch_val_list = self.test_step('val', self.val_data_set)
            self.epoch_val_end_time = time.time()
            self.epoch_end_time = self.epoch_val_end_time

            # log validation
            self.run_log(self.save_dir, self.curr_stage)

            # save and write to writer
            self.save_model(self.save_dir)
            if self.is_cls is None:
                self.write_eval(self.epoch_val_list)

        # test
        self.test_loss = 0.0
        self.test_start_time = time.time()
        self.test_list = []
        self.test_loss, self.test_list = self.test_step('test', self.test_data_set)
        self.test_end_time = time.time()

        # log test
        self.run_log(self.save_dir, self.curr_stage)
        if self.is_cls is None:
            self.write_eval(self.test_list)

    def cls_run(self, n_epoch):
        """"""
        self.n_epoch = n_epoch
        for epoch in range(n_epoch):
            self.epoch_start_time = time.time()
            self.curr_epoch = epoch
            # training
            self.epoch_train_loss = 0.0
            self.epoch_train_start_time = self.epoch_start_time
            self.epoch_train_loss = self.train_step()
            self.epoch_train_end_time = time.time()

            # validation
            self.val_accu = 0.0
            self.epoch_val_start_time = time.time()
            self.epoch_val_list = []
            self.epoch_val_loss, self.epoch_val_list = self.test_step('val', self.val_data_set)
            self.epoch_val_end_time = time.time()
            self.epoch_end_time = self.epoch_val_end_time
            # save and write to writer
            self.save_model(self.save_dir)

            # val accu
            self.val_accu = self.cls_write_eval(self.epoch_val_list)

            # log validation
            self.cls_run_log(self.save_dir, self.curr_stage)

        # test
        # self.test_loss = 0.0
        # self.test_start_time = time.time()
        # self.test_list = []
        # self.test_loss, self.test_list = self.test_step('test', self.test_data_set)
        # self.test_end_time = time.time()

        # log test
        # self.test_accu = self.cls_write_eval(self.test_list)
        # self.cls_run_log(self.save_dir, self.curr_stage)


    def cls_run_log(self, path, mode: str = 'train', is_print: bool = True):
        """"""
        str_data = ''
        separator = ''
        if mode == 'train':
            # log batch train
            separator = '-' * 89
            str_data = '| Epoch {:3d}/{:3d} | Batch: {:5d}/{:5d} |' \
                       ' lr {:10.9f} | s/batch {:5.2f} | Train Loss {:10.9f} |'.format(
                self.curr_epoch, self.n_epoch,
                self.curr_batch, self.n_train_batch,
                self.optimizer.param_groups[0]['lr'],
                self.batch_train_end_time - self.batch_train_start_time,
                self.batch_train_loss)

        elif mode == 'val':
            # log val
            separator = '-' * 89
            str_data = '| End of epoch {:3d} | Val Total Time: {:5.2f}s |' \
                       ' Valid Loss: {:10.9f} |Valid Accuracy {:10.9f}% | '.format(
                self.curr_epoch,
                self.epoch_val_end_time - self.epoch_val_start_time,
                self.epoch_val_loss,
                self.val_accu)

        elif mode == 'test':
            # log test
            separator = '#' * 89
            str_data = '| Test Stage | Test Total Time: {:5.2f}s | Test Accuracy {:10.9f}% | '.format(
                self.test_end_time - self.test_start_time,
                self.test_accu)
        else:
            raise ValueError('Input Log Mode Error.')

        # write to file
        if str_data != '':
            text_file_name = f'{self.model_name}_logging.txt'
            log_file_path = os.path.join(path, text_file_name)
            file_operation.write_txt(log_file_path, separator)
            file_operation.write_txt(log_file_path, str_data)
            file_operation.write_txt(log_file_path, separator)
            # print
            if is_print:
                print(separator)
                print(str_data)
                print(separator)
        else:
            raise ValueError('String Data Container Empty Error.')

    def cls_write_eval(self, list_of_tuple: list):
        """"""
        output_list = []
        label_list = []
        score_list = []
        for tup in list_of_tuple:
            output_list.append(tup[0].argmax(dim=1).item())
            label_list.append(tup[1])
            score_list.append(tup[0][:, tup[1]].item())

        assert len(output_list) == len(label_list)
        out_arr = np.array(output_list, dtype='int32')
        label_arr = np.array(label_list, dtype='int32')
        score_arr = np.array(score_list, dtype='float32')

        # calculate Max Error
        accuracy_score = skl_mtr.accuracy_score(label_arr, out_arr)
        precision_score = skl_mtr.precision_score(label_arr, out_arr)
        recall_score = skl_mtr.recall_score(label_arr, out_arr)
        fpr, tpr, threshold = skl_mtr.roc_curve(label_arr, score_arr, pos_label=1)
        auc = skl_mtr.roc_auc_score(label_arr, score_arr)
        # write to tensorboard
        scalar_dic = {
            'train_loss': self.epoch_train_loss,
            'val_loss': self.epoch_val_loss,
            'accuracy_score': accuracy_score,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'AUC': auc
        }

        self.writer.add_scalars(
            f'{self.model_name}_{self.curr_stage}',
            scalar_dic,
            self.curr_epoch
        )
        self.writer.add_pr_curve(
            f'{self.model_name}_{self.curr_stage}',
            label_arr,
            score_arr,
            self.curr_epoch
        )
        if self.curr_epoch == self.n_epoch - 1:
            self.writer.add_hparams(
                self.hyper_para_dict,
                {'accuracy_score': accuracy_score,
                 'AUC': auc}
            )
        return accuracy_score



    def train_step(self):
        """"""
        self.curr_stage = 'train'
        self.model.train()
        sum_loss = 0
        n_batch = 0
        for batch_index, train_data in enumerate(self.train_data_set):
            self.curr_batch = batch_index
            self.batch_train_start_time = time.time()
            data, label = train_data
            in_data = data.to(torch.float32).to(device=self.device)
            if self.is_cls is not None:
                in_label = label.to(device=self.device, dtype=torch.int64)
            else:
                in_label = label.to(torch.float32).to(device=self.device)
            out = self.model(in_data)
            if len(out.shape) > 1:
                out = out.squeeze(1)
            scalar_loss = self.loss_fn(out, in_label)
            self.optimizer.zero_grad()
            scalar_loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            # record loss
            sum_loss += scalar_loss.item()
            self.batch_train_loss = scalar_loss.item()
            self.batch_train_end_time = time.time()
            n_batch += 1

            # log train
            self.run_log(self.save_dir, self.curr_stage)
        return sum_loss / n_batch

    def test_step(self, stage: str, data_set):
        """"""
        self.model.eval()
        self.curr_stage = stage
        sum_loss = 0
        list_loss = []
        n_batch = 0
        self.score_list = []
        with torch.no_grad():
            for idx, zip_data in enumerate(data_set):
                data, label = zip_data
                in_data = data.to(torch.float32).to(device=self.device)
                if self.is_cls is not None:
                    in_label = label.to(device=self.device, dtype=torch.int64)
                else:
                    in_label = label.to(torch.float32).to(device=self.device)
                out = self.model(in_data)
                if len(out.shape) > 1:
                    out = out.squeeze(1)
                scalar_loss = self.loss_fn(out, in_label)
                # record loss
                sum_loss += scalar_loss.item()
                if self.is_cls is None:
                    list_loss.append((out.item(), in_label.item()))
                else:
                    list_loss.append((out, in_label.item()))
                n_batch += 1
        return sum_loss / n_batch, list_loss

    def save_model(self, path: str) -> None:
        """"""
        assert os.path.exists(path)
        file_name = f'{self.model_name}_{self.curr_epoch}'
        data_type = '.pt'
        save_path = os.path.join(path, file_name + data_type)
        torch.save({
                'model_name': self.model_name,
                'epoch': self.curr_epoch,
                'model': self.model,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
                'train_loss': self.epoch_train_loss,
                'val_loss': self.epoch_val_loss,
            }, save_path)

    def load_model(self, path: str) -> None:
        """"""
        assert os.path.exists(path)
        temp_dict = torch.load(path)
        self.model_name = temp_dict['model_name']
        self.curr_epoch = temp_dict['epoch']
        self.model = temp_dict['model']
        self.optimizer = temp_dict['optimizer']
        self.scheduler = temp_dict['scheduler']
        self.epoch_train_loss = temp_dict['train_loss']
        self.epoch_val_loss = temp_dict['val_loss']

    def run_log(self, path, mode: str = 'train', is_print: bool = True) -> None:
            """"""
            str_data = ''
            separator = ''
            if mode == 'train':
                # log batch train
                separator = '-' * 89
                str_data = '| Epoch {:3d}/{:3d} | Batch: {:5d}/{:5d} | lr {:10.9f} | s/batch {:5.2f} | Loss {:10.9f} |'.format(
                    self.curr_epoch, self.n_epoch,
                    self.curr_batch, self.n_train_batch,
                    self.optimizer.param_groups[0]['lr'],
                    self.batch_train_end_time - self.batch_train_start_time,
                    self.batch_train_loss)

            elif mode == 'val':
                # log val
                separator = '-' * 89
                str_data = '| End of epoch {:3d} | Val Total Time: {:5.2f}s | Avg Valid Loss {:10.9f} | '.format(
                    self.curr_epoch,
                    self.epoch_val_end_time - self.epoch_val_start_time,
                    self.epoch_val_loss)

            elif mode == 'test':
                # log test
                separator = '#' * 89
                str_data = '| Test Stage | Test Total Time: {:5.2f}s | Avg Test Loss {:10.9f} | '.format(
                    self.test_end_time - self.test_start_time,
                    self.test_loss)
            else:
                raise ValueError('Input Log Mode Error.')

            # write to file
            if str_data != '':
                text_file_name = f'{self.model_name}_logging.txt'
                log_file_path = os.path.join(path, text_file_name)
                file_operation.write_txt(log_file_path, separator)
                file_operation.write_txt(log_file_path, str_data)
                file_operation.write_txt(log_file_path, separator)
                # print
                if is_print:
                    print(separator)
                    print(str_data)
                    print(separator)
            else:
                raise ValueError('String Data Container Empty Error.')

    def write_eval(self, list_of_tuple: list):
        """"""
        output_list = []
        label_list = []
        for tup in list_of_tuple:
            output_list.append(tup[0])
            label_list.append(tup[1])

        assert len(output_list) == len(label_list)
        out_arr = np.array(output_list, dtype='float32')
        label_arr = np.array(label_list, dtype='float32')
        abs_diff_arr = np.absolute(label_arr - out_arr)

        # calculate Max Error
        exp_var = skl_mtr.explained_variance_score(label_arr, out_arr)  # 可解释方差
        mean_sqrt_err = skl_mtr.mean_squared_error(label_arr, out_arr)  # 均方误差
        root_mean_sqrt_err = np.sqrt(mean_sqrt_err)  # 均方根误差
        max_err = skl_mtr.max_error(label_arr, out_arr)  # 误差最大值
        mean_abs_err = skl_mtr.mean_absolute_error(label_arr, out_arr)  # 平均绝对误差
        mean_abs_per_err = skl_mtr.mean_absolute_percentage_error(label_arr, out_arr)  # 平均相对误差
        mid_abs_err = skl_mtr.median_absolute_error(label_arr, out_arr)  # 误差中位数
        r2_score = skl_mtr.r2_score(label_arr, out_arr)  # R^2 index

        # write to tensorboard
        scalar_dic = {
            'train_loss': self.epoch_train_loss,
            'val_loss': self.epoch_val_loss,
            # 'Explained variance score': exp_var,
            'Max error': max_err,
            'Mean absolute error': mean_abs_err,
            # 'Mean squared error': mean_sqrt_err,
            'Root Mean squared error': root_mean_sqrt_err,
            'Mean absolute percentage error': mean_abs_per_err,
            'Median absolute error': mid_abs_err,
            # 'R2 score': r2_score,
        }

        self.writer.add_scalars(
            f'{self.model_name}_{self.curr_stage}',
            scalar_dic,
            self.curr_epoch
        )

        # write fig to tensorboard
        x_ax = [x for x in range(len(out_arr))]
        fig, ax = plt.subplots(nrows=3, ncols=1)
        # setup fig
        f = 0.8
        resolution = (1920, 1080)
        screen_inch = (13.60 * f, 7.64 * f)
        fig.set(dpi=150)
        fig.set_size_inches(*screen_inch)

        # output plot
        ax[0].plot(x_ax, out_arr, 'r.', markersize=4)
        ax[0].set_title('Output')

        # label plot
        ax[1].plot(x_ax, label_arr, 'g.', markersize=4)
        ax[1].set_title('Label')

        # error plot
        ax[2].plot(x_ax, abs_diff_arr, 'b.', markersize=4)
        ax[2].set_title('Abs Error')

        plt.tight_layout()

        self.writer.add_figure(
            f'{self.model_name}_{self.curr_stage}',
            fig,
            self.curr_epoch
        )


if __name__ == '__main__':

   """ import sys
    import os
    import datetime
    import functional
    import rgn_model_define
    import data_set_creat
    from opt import optimizer_select_and_init

    # writer log path
    log_dir_base = 'D:\\workspace\\PycharmProjects\\model_run\\'
    #################################################
    USE_GPU = False
    DEBUG = False
    CH1_VOLTAGE_LEN = 161
    CRITERION = nn.MSELoss()
    in_len = 40
    n_epochs = 3
    batch_size = 16
    #################################################
    # data set setup
    data_file_path = 'D:\\workspace\\PycharmProjects\\battery_dataset\\2600P-01_DataSet\\data_set'
    data_file_name = '20210804-095208_ch1_form-ocv_std__label-std_'
    data_file_type = '.pt'
    transformation = (functional.cut_sequence,)
    transformation_para = (
        (40,),  # (out_len, )
        # (32, False, 16), # (t_len, is_overlap, step)
    )
    ########################################
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
    ########################################

    if USE_GPU:
        device = functional.try_gpu()
    else:
        device = torch.device('cpu')

    net = rgn_model_define.PureMLP(in_len).to(device=device)
    op_dict = optimizer_select_and_init(net,
                                        'Adam',
                                        None,
                                        {'lr': 0.001,
                                         'betas': (0.9, 0.95),
                                         'eps': 1e-9,
                                         'weight_decay': 0,
                                         'amsgrad': False},
                                        {})

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
        m_run = Run(dataset_profile_dic,
                    net,
                    CRITERION,
                    writer,
                    op_dict['optimizer'],
                    op_dict['scheduler'],
                    device=device,
                    is_cls=2)
        m_run.run(n_epochs)
    sys.exit(0)"""