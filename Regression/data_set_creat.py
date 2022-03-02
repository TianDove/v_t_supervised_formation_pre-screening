#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang

import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# import self define module
import file_operation
import functional

class DataSetCreator(Dataset):
    """"""
    def __init__(self, path: str, transform: tuple = None, trans_para: tuple = None) -> None:
        self.file_path = path
        self.list_files = os.listdir(self.file_path)
        temp_all_npy_list = []
        for file in self.list_files:
            end_fix = os.path.splitext(file)[1]
            if '.npy' == end_fix:
                temp_all_npy_list.append(file)
        self.list_files = temp_all_npy_list

        self.num_of_samples = len(self.list_files)

        #
        self.transform = None
        self.trans_para = None
        if (transform is not None) and (trans_para is not None):
            assert (type(transform) == tuple) and (type(trans_para) == tuple)
            assert len(transform) == len(trans_para)
            self.transform = transform
            self.trans_para = trans_para

        # join path
        self.file_path_list = []
        for i in iter(self.list_files):
            temp_file_path = os.path.join(self.file_path, i)
            self.file_path_list.append(temp_file_path)

    def __getitem__(self, index):
        data = np.load(self.file_path_list[index])
        temp_data = data[0:-1]
        temp_label = data[-1]
        if (self.transform is not None) and (self.trans_para is not None):
            temp_data = data[0:-1]
            temp_label = data[-1] / 1000
            for tran_i, para_i in zip(iter(self.transform), iter(self.trans_para)):
                temp_data = tran_i(temp_data, **(para_i if para_i else {}))
        return {'data': temp_data,
                'label': temp_label}

    def __len__(self):
        return len(self.file_path_list)

    @classmethod
    def creat_dataset(cls,
                      data_path: str,
                      bsz: int = 32,
                      is_shuffle: bool = True,
                      num_of_worker: int = 0,
                      transform: tuple = None,
                      trans_para: tuple = None) -> (DataLoader, int):
        """"""
        assert os.path.exists(data_path)
        data_set = cls(data_path, transform, trans_para)
        batch_data_set = DataLoader(data_set, batch_size=bsz, shuffle=is_shuffle, num_workers=num_of_worker)
        num_of_batch = math.floor(len(os.listdir(data_path)) / bsz)
        return batch_data_set, num_of_batch


class Ch1_FormOcv_DataSetCreator(Dataset):
    """"""
    def __init__(self, data_file_path: str,
                 data_set_type: str = 'train',
                 transform: tuple = None,
                 transform_paras: tuple = None,
                 is_cls: bool = False):
        """
         data_dict = {
            'train df': total_df[train_list],
            'val df': total_df[test_list],
            'test df': total_df[val_list],
            ...
        """
        self.data_file_path = data_file_path
        self.curr_data_set_state = data_set_type
        self.file_name = os.path.splitext(os.path.basename(self.data_file_path))[0]
        self.data_df_dict = file_operation.load_dic_in_pickle(self.data_file_path)[self.file_name]
        self.transform = transform
        self.transform_paras = transform_paras
        self.is_cls = is_cls

        # init data set state
        self.curr_data_df = None
        self.set_data_set_state(self.curr_data_set_state)
        self.num_samples = self.curr_data_df.shape[0]

    def __getitem__(self, index: int):
        """"""
        if self.is_cls:
            temp_data = self.curr_data_df.iloc[index, 0:-1].astype('float32').values
            temp_label = np.array(float(self.curr_data_df.iloc[index, -1]), dtype='float32')
        else:
            temp_data = self.curr_data_df.iloc[index, 0:-3].astype('float32').values
            if 'label_std_op' in list(self.data_df_dict.keys()):
                temp_label = np.array(float(self.curr_data_df.iloc[index, -3]), dtype='float32')
            else:
                temp_label = np.array(float(self.curr_data_df.iloc[index, -3]), dtype='float32')   # Scale Form-OCV to (V)

        if (self.transform is not None) and (self.transform_paras is not None):
            assert len(self.transform) == len(self.transform_paras)
            temp_trans_zipped = zip(self.transform, self.transform_paras)
            for c_trans, c_trans_para in temp_trans_zipped:
                temp_data = c_trans(temp_data, *c_trans_para)
        r_data = torch.tensor(temp_data)
        r_label = torch.tensor(temp_label)
        return r_data, r_label

    def __len__(self):
        """"""
        return self.num_samples

    def get_num_of_sample(self):
        """"""
        return self.num_samples

    def set_data_set_state(self, state: str = 'train'):
        """"""
        self.curr_data_set_state = state
        if state == 'train':
            self.curr_data_df = self.data_df_dict['train df']
        elif state == 'val':
            self.curr_data_df = self.data_df_dict['val df']
        elif state == 'test':
            self.curr_data_df = self.data_df_dict['test df']
        else:
            raise ValueError('Input State Error.')

    def get_current_data_set_state(self):
        """"""
        return self.curr_data_set_state

    @classmethod
    def creat_dataset(cls,
                      data_path: str,
                      data_set_type: str,
                      bsz: int = 32,
                      is_shuffle: bool = True,
                      num_of_worker: int = 0,
                      transform: tuple = None,
                      trans_para: tuple = None,
                      is_cls: bool = False) -> (DataLoader, int):
        """"""
        assert os.path.exists(data_path)
        data_set = cls(data_path, data_set_type, transform, trans_para, is_cls)
        batch_data_set = DataLoader(data_set,
                                    batch_size=bsz,
                                    shuffle=is_shuffle,
                                    num_workers=num_of_worker,
                                    pin_memory=True)
        num_of_batch = math.ceil(data_set.get_num_of_sample() / bsz)
        return batch_data_set, num_of_batch


if __name__ == '__main__':
    import sys
    base = 'D:\\workspace\\PycharmProjects\\battery_dataset\\2600P-01_DataSet\\data_set\\clasification'
    file_name = '20210812-205445_ch1_form-ocv_label-encoded_small'
    file_type = '.pt'
    batch_size = 64
    transformation = (functional.cut_sequence, functional.tokenize,)
    transformation_para = (
        (40, ),  # (out_len, )
        (32, False, 16), # (t_len, is_overlap, step)
    )
    m_dataset, n_batch = Ch1_FormOcv_DataSetCreator.creat_dataset(os.path.join(base, file_name + file_type),
                                                                 'train',
                                                                 bsz=batch_size,
                                                                 is_shuffle=False,
                                                                 num_of_worker=0,
                                                                 transform=transformation,
                                                                 trans_para=transformation_para,
                                                                 is_cls=True)
    for b_idx, b_data in enumerate(m_dataset):
        data, label = b_data
    sys.exit(0)