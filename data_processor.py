#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang

import os
import math
import time
import copy
import datetime
import random
import multiprocessing as mp
import numpy as np
import pandas as pd
import random
from collections import Counter
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, NearMiss

import matplotlib
import matplotlib.pyplot as plt

# import self define module
import file_operation

# Constant
Num_PROCESS = mp.cpu_count() - 1


class DataProcessor():
    """"""
    def __init__(self,
                 organized_path: str,
                 save_path: str,
                 param_mod_dic: dict,
                 file_type: str = 'pickle',
                 is_multi_worker: bool = True):
        """
        param_mode_dic = {
        'Static': ['Form-OCV #1', ],
        'Charge #1': ['time', 'voltage'],
        'Charge #2': [],
        'Charge #3': [],
        'Discharge': [],
        'Charge #4': []
        }
        """
        # save input
        self.organized_file_path = organized_path
        self.save_path = save_path
        self.param_mod_dic = param_mod_dic
        self.is_multi_worker = is_multi_worker
        self.file_type = file_type

        self.num_processes = mp.cpu_count() - 1
        self.unselected_cells_list = []

        # set load func
        self.load_func = None
        if self.file_type == 'pickle':
            self.load_func = file_operation.load_dic_in_pickle
        elif self.file_type == 'xlsx':
            self.load_func = file_operation.read_xlsx_all_sheet
        else:
            assert self.load_func

        # set used_key
        self.used_key = []
        for use_key in iter(self.param_mod_dic.keys()):
            if self.param_mod_dic[use_key]:
                self.used_key.append(use_key)

    def data_load_select_convert(self, is_small_set: bool = False, ratio: float = 0.1):
        """"""
        file_list = self._get_file_list(self.organized_file_path, self.file_type)
        if is_small_set:
            random.shuffle(file_list)
            num_sample = math.floor(len(file_list) * ratio)
            file_list = file_list[0:num_sample]
        print('Load, Select, Convert Start.')
        # load, selection and convert
        res = []
        start_time = time.time()
        if self.is_multi_worker:
            with mp.Pool(processes=self.num_processes) as pool:
                res = pool.map(self._load_select_convert_wrapper, file_list)
        else:
            with tqdm(total=len(file_list)) as bar:
                bar.set_description('Loading ,Selecting and Concerting ....')
                for j in file_list:
                    bar.update()
                    temp_data_dic = self._load_select_convert_wrapper(j)
                    res.append(temp_data_dic)
        end_time = time.time()
        print('Load, Select, Convert End.')
        # transform res to dict
        res = self._res_to_dic(res)
        self._save_and_log(start_time, end_time, res)

    def _save_and_log(self, start, end, processed_data: dict):
        """"""
        data_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_operation.path_assert(self.save_path, is_creat=True)
        data_file_name = data_time_str + '_' + 'processed_data.pt'
        data_to_write = {
            'param_mode_dic': self.param_mod_dic,
            'processed_data': processed_data
        }
        file_operation.save_dic_as_pickle(os.path.join(self.save_path, data_file_name), data_to_write)

        log_file_path = os.path.join(self.save_path, data_time_str + '_' + 'DataProcessor_Logging.txt')
        consumed_time = end - start

        seperator = '-' * 89
        file_operation.write_txt(log_file_path, seperator)

        text_data = '| Finish Date: {} | Read File Type: {} | Multiprocess Flag: {} | Time Consume: {} s |'.format(
            data_file_name, self.file_type, self.is_multi_worker, consumed_time)
        file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)

        text_data = '| Parameters Select Config Dict |'
        file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)

        file_operation.write_txt(log_file_path, self.param_mod_dic)
        file_operation.write_txt(log_file_path, seperator)

        text_data = '| Un Selected Cells List |'
        file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)

        for text in self.unselected_cells_list:
            file_operation.write_txt(log_file_path, text)
        file_operation.write_txt(log_file_path, seperator)

    @staticmethod
    def _res_to_dic(res: list) -> dict:
        """"""
        out_dic = {}
        for i in res:
            temp = i
            out_dic.update(temp)
        return out_dic

    @staticmethod
    def _get_file_list(organized_file_path, file_type) -> list:
        """"""
        # get file path list
        files_path = os.path.join(organized_file_path, file_type)
        assert os.path.exists(files_path)
        files_list = os.listdir(files_path)
        files_path_list = []
        for i in iter(files_list):
            temp_path = os.path.join(files_path, i)
            files_path_list.append(temp_path)
        return files_path_list

    def _load_select_convert_wrapper(self, files_path: str, ):
        """"""
        # start progress
        raw_data = self.load_func(files_path)
        selected_data = self._cell_data_selection(raw_data)
        converted_data = self._data_convert(selected_data)
        return converted_data

    def _cell_data_selection(self, data_dic: dict) -> dict:
        """"""
        cells_list = list(data_dic.keys())
        cells_dic = {}
        # list_extracted = copy.deepcopy(cells_list)
        for cell_name in iter(cells_list):
            cell_dic = {}
            selected_flag = True
            # get the key of parameter of the current cell
            cell_key = list(data_dic[cell_name].keys())
            # whether cell have used parameter
            for key in iter(self.used_key):
                if key not in cell_key:
                    if not self.is_multi_worker:
                        self.unselected_cells_list.append(cell_name)
                    selected_flag = False
                    break
            if selected_flag:
                for used in iter(self.used_key):
                    para_list = self.param_mod_dic[used]
                    para = data_dic[cell_name][used][[x for x in iter(para_list)]]
                    cell_dic.update({f'{used}': para})
                cells_dic.update({f'{cell_name}': cell_dic})
        return cells_dic

    @staticmethod
    def _data_convert(cells_dic: dict):
        """"""
        cell_list = list(cells_dic.keys())
        dic_extracted_converted = {}
        for cell in iter(cell_list):
            temp_data_dic = cells_dic[cell]
            for key in iter(temp_data_dic.keys()):
                if key == 'Static':
                    temp_data_dic[key] = temp_data_dic[key]
                else:
                    temp_data_dic[key] = temp_data_dic[key].iloc[1:, :].astype('float32', 1)
            dic_extracted_converted.update({f'{cell}': temp_data_dic})
        return dic_extracted_converted


class Ch1_FormOcv_Data_Process():
    """"""
    def __init__(self,
                 data_dict: dict,
                 data_save_path: str,
                 is_3_sigma=False,
                 is_std: bool = False,
                 is_norm: bool = False,
                 is_minmax: bool = False,
                 is_label_std: bool = False,
                 classification_bound: tuple = None,
                 is_imbd: bool = None) -> None:
        """"""
        self.data_dict = data_dict
        self.data_save_path = data_save_path
        self.is_std = is_std
        self.is_norm = is_norm
        self.is_minmax = is_minmax
        self.is_label_std = is_label_std
        self.is_3_sigma = is_3_sigma
        self.is_3_sigma_ex_list = []
        self.classification_bound = classification_bound
        self.is_imbd = is_imbd

        self.dataset_fig = None
        self.std_op = None
        self.norm_op = None
        self.minmax_op = None
        self.label_std_op = None
        self.count_res = None
        self.en_count_res = None

    def operation_init(self) -> list:
        """"""
        if self.is_std:
            self.std_op = StandardScaler(copy=False)
        if self.is_norm:
            self.norm_op = Normalizer(copy=False)
        if self.is_minmax:
            self.minmax_op = MinMaxScaler(copy=False)
        op_list = [self.std_op, self.norm_op, self.minmax_op]
        return op_list

    @staticmethod
    def apply_operation(data_dict: dict, op_list: list) -> dict:
        """
         data_dict = {
            'train df': total_df[train_list],
            'val df': total_df[test_list],
            'test df': total_df[val_list]
        """
        cp_data_dic = copy.deepcopy(data_dict)
        for op in op_list:
            if op is not None:
                # op fit and trans training data
                op.fit_transform(cp_data_dic['train df'])

                # op apply to val and test data
                op.transform(cp_data_dic['val df'])
                op.transform(cp_data_dic['test df'])
        return cp_data_dic

    def regression_preporcess(self):
        """"""
        """"""
        comb_data = self.data_combination(self.data_dict)
        align_df = self.align_and_three_sigma(comb_data)
        divided_df_dict = self.data_divide(align_df)
        op_list = self.operation_init()
        op_df_dict = self.apply_operation(divided_df_dict, op_list)

        # plot data set
        self.plot_data_set(op_df_dict)

        for data_key in op_df_dict.keys():
            op_df_dict[data_key] = self.concat_data_label(op_df_dict[data_key], self.data_dict)

        # standardize label
        if self.is_label_std:
            op_df_dict = self.std_label(op_df_dict)
        self.save_divided_data_df(op_df_dict, self.data_save_path)

    def classification_preporcess(self):
        """"""
        comb_data = self.data_combination(self.data_dict)
        align_df = self.align_and_three_sigma(comb_data)
        op_list = self.operation_init()
        divided_df_dict = self.data_divide(align_df)
        op_df_dict = self.apply_operation(divided_df_dict, op_list)

        for data_key in op_df_dict.keys():
            op_df_dict[data_key] = self.concat_data_label(op_df_dict[data_key], self.data_dict)

        # standardize label
        if self.is_label_std:
            op_df_dict = self.std_label(op_df_dict)

        self.count_label_distr(op_df_dict)
        op_df = self.clasification_label_encoding(op_df_dict)
        if self.is_imbd:
            op_df = self.resampling_data(op_df)
        op_df_dict = self.classification_dataset_dividing(op_df)
        # plot data set
        self.plot_data_set(op_df_dict)
        self.save_divided_data_df(op_df_dict, self.data_save_path)

    def classification_dataset_dividing(self, df: pd.DataFrame, ratio: float = 0.2) -> dict:
        """"""
        temp_df = df
        sort_df = temp_df.sort_values(by=['Form-OCV #1',])
        neg_df = sort_df[sort_df['Form-OCV #1'] == 0].T
        pos_df = sort_df[sort_df['Form-OCV #1'] == 1].T
        neg_df_dict = self.data_divide(neg_df, ratio)
        pos_df_dict = self.data_divide(pos_df, ratio)
        
        temp_df_dic = {}
        for key in neg_df_dict:
            temp_neg_df = neg_df_dict[key]
            temp_pos_df = pos_df_dict[key]
            temp_key_df = pd.concat([temp_neg_df, temp_pos_df], axis=0)
            temp_df_dic.update({key: temp_key_df})
        return temp_df_dic

    def resampling_data(self, df: pd.DataFrame, maj_ratio: float = 2, min_ratio: float = 0.35) -> pd.DataFrame:
        """"""
        temp_dataset_df = df
        temp_data = temp_dataset_df.iloc[:, 0:-1]
        temp_label = temp_dataset_df[['Form-OCV #1']]

        # data set over-sample
        over_sampler_dict = {0: math.ceil(self.en_count_res[1] * min_ratio)}
        over_sampler = SMOTE(over_sampler_dict)
        data_over, label_over = over_sampler.fit_resample(temp_data, temp_label)

        # data set under-sample
        under_sampler_nm_dict = {1: math.ceil(self.en_count_res[1] * min_ratio * maj_ratio)}
        under_sampler_nm = NearMiss(under_sampler_nm_dict, version=2)
        data_under, label_under = under_sampler_nm.fit_resample(data_over, label_over)

        under_sampler_tl = TomekLinks()
        data_under, label_under = under_sampler_tl.fit_resample(data_under, label_under)

        concat_df = pd.concat([data_under, label_under], axis=1)
        return concat_df

    def concat_data_label(self, data_df: pd.DataFrame, data_dict: dict) -> pd.DataFrame:
        """"""
        print('Concatenate Data and Label Processing....')
        in_data = data_df.T
        col_list = in_data.columns
        temp_df_list = []
        for col in col_list:
            str_split = col.split('_')
            cell_name = str_split[0] + '_' + str_split[1] + '_' + str_split[2]
            static_data = data_dict[cell_name]['Static'].T
            static_data.columns = [col, ]
            temp_df = static_data
            temp_df_list.append(temp_df)
        temp_total_df = pd.concat(temp_df_list, axis=1)
        data_label_df = pd.concat([in_data, temp_total_df])
        print('Concatenate Data and Label Complete.')
        return data_label_df.T

    def clasification_label_encoding(self, df_dict: dict) -> pd.DataFrame:
        """"""
        temp_dict = df_dict
        for key in temp_dict:
            temp_df = temp_dict[key]
            temp_df['Form-OCV #1'] = temp_df['Form-OCV #1'].astype('float32').apply(self.cls_encoding_fn)
        t = pd.concat([temp_dict['train df'], temp_dict['val df'],  temp_dict['test df']])
        self.en_count_res = t['Form-OCV #1'].value_counts()
        t = t.drop(['Grade', 'Remark'], axis=1)
        return t

    def cls_encoding_fn(self, raw_label):
        """"""
        lo_b = self.classification_bound[0]
        up_b = self.classification_bound[1]
        en_label = -1
        if (raw_label >= lo_b) and (raw_label <= up_b):
            en_label = 1
        else:
            en_label = 0
        return en_label

    @staticmethod
    def data_combination(data_dic: dict) -> pd.DataFrame:
        """"""
        # 将每个电压曲线与对应时间序列结合构成一个DataFrame,并添加该曲线的电池相关信息
        para_name = 'Charge #1'
        ch1_df_list = []
        with tqdm(total=len(list(data_dic.keys()))) as bar:
            bar.set_description('Charge #1 Data Concatenating....')
            for cell in iter(data_dic):
                bar.update()
                temp_df = data_dic[cell][para_name]
                re_idx_df = temp_df.set_index(keys=['time'])
                re_col_df = re_idx_df.add_prefix(cell + '_')
                ch1_df_list.append(re_col_df)
        # 将每条电压曲线数据合并为一个总的DataFrame
        ch1_v_total_df = pd.concat(ch1_df_list, axis=1)
        return ch1_v_total_df

    def align_and_three_sigma(self, df: pd.DataFrame) -> (pd.DataFrame, list):
        """"""
        # 去除非整的index值
        row_list = list(df.index)
        del_row_list = []
        for row in iter(row_list):
            index = row
            res = index % 0.5
            if res != 0:
                del_row_list.append(index)
        # 缺失值差值
        align_df = df.drop(del_row_list).copy()
        print('Interpolation Processing....')
        align_df = align_df.interpolate(method='quadratic', axis=0)
        print('Interpolation Complete.')

        if self.is_3_sigma:
            # 3 sigma
            align_df_values = align_df.values
            mean = np.mean(align_df_values, axis=1, keepdims=True)
            std = np.std(align_df_values, axis=1, keepdims=True, ddof=1)
            temp_sq = align_df_values
            upper_bound = mean + 3 * std
            lower_bound = mean - 3 * std
            a1 = (lower_bound < temp_sq)
            a2 = (temp_sq < upper_bound)
            a = np.logical_and(a1, a2)
            a_sum = a.sum(axis=0, keepdims=True)
            a_index = a_sum < (a.shape[0] / 2)  # True for del col
            del_index = np.where(a_index == True)
            align_df_col_list = list(align_df.columns)
            #
            index_to_col_list = []
            for index in iter(del_index[1]):
                temp_col = align_df_col_list[index]
                index_to_col_list.append(temp_col)
            self.is_3_sigma_ex_list = index_to_col_list
            align_df = align_df.drop(index_to_col_list, axis=1)
        return align_df

    @staticmethod
    def data_divide(total_df: pd.DataFrame, ratio: float = 0.2) -> dict:
        """"""
        print('Diving Processing....')
        file_list = list(total_df.columns)
        assert file_list
        selected_samples = file_list
        num_of_samples = len(selected_samples)
        random.shuffle(selected_samples)
        shuffled_sample_list = selected_samples

        # calculate number of sample in different dataset
        val_num = math.floor(num_of_samples * ratio)
        test_num = math.floor(num_of_samples * ratio)
        train_num = num_of_samples - val_num - test_num

        # start dividing
        assert (val_num + test_num + train_num) == num_of_samples
        val_list = shuffled_sample_list[0: val_num]
        test_list = shuffled_sample_list[val_num: val_num + test_num]
        train_list = shuffled_sample_list[val_num + test_num: val_num + test_num + train_num]
        divide_df_dic = {
            'train df': total_df[train_list].T,
            'val df': total_df[test_list].T,
            'test df': total_df[val_list].T
        }
        print('Diving End.')
        return divide_df_dic

    def save_divided_data_df(self, data_dict: dict, data_set_path: str) -> None:
        """"""
        data_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_name_base = data_time_str + '_' + 'ch1_form-ocv'
        file_type = '.pt'
        save_name = save_name_base
        if self.is_std:
            save_name += '_std_'
            data_dict.update({f'std_op': self.std_op})
        if self.is_label_std:
            save_name += '_label-std_'
            data_dict.update({f'label_std_op': self.label_std_op})
        if self.is_norm:
            save_name += '_norm_'
            data_dict.update({f'norm_op': self.norm_op})
        if self.is_minmax:
            save_name += '_minmax_'
            data_dict.update({f'minmax_op': self.minmax_op})
        if self.is_3_sigma:
            save_name += '_3sigma_'
            data_dict.update({f'3sigma_ex_list': self.is_3_sigma_ex_list})
        if self.classification_bound is not None:
            save_name += '_label-encoded_'
            data_dict.update({f'label_encoded_bound': self.classification_bound})
        if self.is_imbd:
            save_name += '_imbd_'
        # save dataset fig
        fig_save_name = save_name + '.png'
        fig_save_path = os.path.join(data_set_path, fig_save_name)
        self.dataset_fig.savefig(fig_save_path)

        # save dataset dict
        file_operation.save_dic_as_pickle(os.path.join(data_set_path, save_name + file_type), data_dict)
        # writer logging
        text_path = os.path.join(data_set_path, save_name + '.txt')
        sp = '-' * 90
        file_operation.write_txt(text_path, sp)
        text_str = f'| 3 Sigma: {self.is_3_sigma} ' \
                   f'| Standardization: {self.is_std} ' \
                   f'| Normalization: {self.is_norm} ' \
                   f'| MinMaxScale: {self.is_minmax} |' \
                   f'| Label Std: {self.is_label_std} |' \
                   f'| Label Encode Bound: {self.classification_bound} |'
        file_operation.write_txt(text_path, text_str)
        file_operation.write_txt(text_path, sp)
        text_str = ['3 Sigma Excluded Cell']
        file_operation.write_txt(text_path, text_str)
        file_operation.write_txt(text_path, sp)
        for text in self.is_3_sigma_ex_list:
            file_operation.write_txt(text_path, text)
        file_operation.write_txt(text_path, sp)

    def plot_data_set(self, data_set_dict):
        """"""
        n_row = len(data_set_dict.keys())
        fig, ax = plt.subplots(nrows=n_row)
        for idx, key in enumerate(data_set_dict):
            temp_df = data_set_dict[key].T
            if 'Form-OCV #1' in temp_df.index:
                temp_df = temp_df.drop('Form-OCV #1')
            if 'Remark' in temp_df.index:
                temp_df = temp_df.drop('Remark')
            if 'Grade' in temp_df.index:
                temp_df = temp_df.drop('Grade')
            temp_df.plot(ax=ax[idx], title=key, grid=True, legend=False)
        self.dataset_fig = fig

    def std_label(self, data_dict: dict) -> dict:
        """"""
        temp_dict = data_dict
        self.label_std_op = StandardScaler(copy=False)
        # extract value
        train_label = np.expand_dims(temp_dict['train df']['Form-OCV #1'].astype('float32', copy=False).values, 1)
        val_label = np.expand_dims(temp_dict['val df']['Form-OCV #1'].astype('float32', copy=False).values, 1)
        test_label = np.expand_dims(temp_dict['test df']['Form-OCV #1'].astype('float32', copy=False).values, 1)
        # transform
        self.label_std_op.fit_transform(train_label)
        self.label_std_op.transform(val_label)
        self.label_std_op.transform(test_label)

        temp_dict['train df']['Form-OCV #1'] = train_label
        temp_dict['val df']['Form-OCV #1'] = val_label
        temp_dict['test df']['Form-OCV #1'] = test_label

        return temp_dict

    def count_label_distr(self, df_dict: dict) -> None:
        """"""
        temp_df = pd.concat([df_dict['train df'], df_dict['val df'], df_dict['test df']])
        temp_df['Form-OCV #1'] = temp_df['Form-OCV #1'].astype('float32')
        self.count_res = temp_df['Form-OCV #1'].value_counts()

        # plot label
        fig, ax = plt.subplots()
        y = np.copy(temp_df['Form-OCV #1'].values)
        x = np.arange(0, y.shape[0])
        ax.scatter(x, y)

    @staticmethod
    def rand_select(df: pd.DataFrame, ratio: float = 0.2):
        """"""
        assert len(df.shape) == 2
        arr_shape = df.shape
        idx = [x for x in range(arr_shape[0])]
        random.shuffle(idx)
        end_idx = math.ceil(arr_shape[0] * ratio)
        select_idx = idx[0: end_idx]
        select_df = df.iloc[select_idx, :]
        return select_df

    @staticmethod
    def plot_dataset_cls(df: pd.DataFrame):
        """"""
        neg_data_df = df[df['Form-OCV #1'] == 0].iloc[:, 0:-1]
        pos_data_df = df[df['Form-OCV #1'] == 1].iloc[:, 0:-1]
        df_shape = neg_data_df.shape
        x_ax = [x for x in range(df_shape[1])]

        plt.ioff()
        fig, ax = plt.subplots()
        f = 0.8
        resolution = (1920, 1080)
        screen_inch = (13.60 * f, 7.64 * f)
        fig.set(dpi=150)
        fig.set_size_inches(*screen_inch)

        neg_data_df.T.plot(ax=ax, legend=False, grid=True, color='r', linewidth=1, label='neg')
        pos_data_df.T.plot(ax=ax, legend=False, grid=True, color='b', linewidth=1, label='pos')
        ax.legend(loc='upper right')
        fig.savefig('.//fig.png')


class Ch1_Form_OCV_Classification_Data_Process():
    """"""
    def __init__(self,
                 data_dict: dict,
                 data_save_path: str,
                 classification_bound: tuple,
                 is_std: bool = False,
                 is_3sigm: bool = False,
                 is_oversample: bool = False,
                 is_undersample: bool = False) -> None:
        """"""
        pass

    @staticmethod
    def data_combination(data_dic: dict) -> pd.DataFrame:
        """"""
        # 将每个电压曲线与对应时间序列结合构成一个DataFrame,并添加该曲线的电池相关信息
        para_name = 'Charge #1'
        ch1_df_list = []
        with tqdm(total=len(list(data_dic.keys()))) as bar:
            bar.set_description('Charge #1 Data Concatenating....')
            for cell in iter(data_dic):
                bar.update()
                temp_df = data_dic[cell][para_name]
                re_idx_df = temp_df.set_index(keys=['time'])
                re_col_df = re_idx_df.add_prefix(cell + '_')
                ch1_df_list.append(re_col_df)
        # 将每条电压曲线数据合并为一个总的DataFrame
        ch1_v_total_df = pd.concat(ch1_df_list, axis=1)
        return ch1_v_total_df

if __name__ == '__main__':
    import sys
    param_mode_dic = {
        'Static': ['Form-OCV #1', 'Grade', 'Remark'],
        'Charge #1': ['time', 'voltage'],
        'Charge #2': [],
        'Charge #3': [],
        'Discharge': [],
        'Charge #4': []
    }
    """base = 'D:\\workspace\\PycharmProjects\\battery_dataset\\2600P-01_DataSet'
    organized_file_path = os.path.join(base, 'organized_data')
    save_processed_path = os.path.join(base, 'processed_data')
    m_data_pro = DataProcessor(organized_file_path,
                               save_processed_path,
                               param_mode_dic,
                               file_type='pickle',
                               is_multi_worker=False)
    m_data_pro.data_load_select_convert(is_small_set=True)"""

    base = 'D:\\workspace\\PycharmProjects\\battery_dataset\\2600P-01_DataSet'
    data_file_name = '20210721-091142_processed_data'
    # 20210721-151231_small_processed_data or 20210721-091142_processed_data
    data_path = os.path.join(base, 'processed_data', data_file_name + '.pt')
    data_save_path = os.path.join(base, 'data_set')
    temp_data_dict = file_operation.load_dic_in_pickle(data_path)
    ch1_form_ocv_data_dict = temp_data_dict[data_file_name]['processed_data']
    m_ch1_form_ocv_processor = Ch1_FormOcv_Data_Process(ch1_form_ocv_data_dict,
                                                        data_save_path,
                                                        is_3_sigma=True,
                                                        is_std=False,
                                                        is_norm=False,
                                                        is_minmax=False,
                                                        is_label_std=False,
                                                        classification_bound=(3538.0, 3549.0),
                                                        is_imbd=True)

    m_ch1_form_ocv_processor.classification_preporcess()

    """base = 'D:\\workspace\PycharmProjects\\battery_dataset\\2600P-01_DataSet\\data_set\\clasification'
    data_file_name = '20210812-205445_ch1_form-ocv_label-encoded_'
    data_path = os.path.join(base, data_file_name + '.pt')
    temp_data_dict = file_operation.load_dic_in_pickle(data_path)
    ch1_form_ocv_data_dict = temp_data_dict[data_file_name]
    concat_df = pd.concat([ch1_form_ocv_data_dict['train df'],
                           ch1_form_ocv_data_dict['val df'],
                           ch1_form_ocv_data_dict['test df']])
    # Ch1_FormOcv_Data_Process.plot_dataset_cls(concat_df)
    sel_df = Ch1_FormOcv_Data_Process.rand_select(concat_df, 0.1)
    dvi_dic = Ch1_FormOcv_Data_Process.data_divide(sel_df.T)
    temp_data_dict[data_file_name] = dvi_dic
    file_operation.save_dic_as_pickle('.//_ch1_form-ocv_label-encoded_small.pt', temp_data_dict)"""

    """base = 'D:\\workspace\PycharmProjects\\battery_dataset\\2600P-01_DataSet\\data_set'
    data_file_name = '20210813-111913_ch1_form-ocv_std__label-encoded__imbd_'
    data_path = os.path.join(base, data_file_name + '.pt')
    temp_data_dict = file_operation.load_dic_in_pickle(data_path)[data_file_name]
    m_cls_processor = Ch1_Form_OCV_Classification_Data_Process()"""
    sys.exit(0)