#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
import math
import os
import sys
import time
import datetime
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm

# import self define module
import file_operation
import functional

# constant
NUM_CELL_IN_TRAY = 256
Num_PROCESS = mp.cpu_count() - 1


class FileOrganizer():
    """"""
    def __init__(self,
                 in_path: str,
                 out_path: str,
                 is_multi_worker: bool = False,
                 is_write: bool = False) -> None:
        """"""
        # save input
        self.in_path = in_path
        self.out_path = out_path
        self.is_multi_worker = is_multi_worker
        self.is_write = is_write

        # class path
        self.dynamic_data_path = None
        self.static_data_path = None

        # class state
        self.curr_batch = None
        self.curr_tray = None
        self.curr_cell = None
        self.curr_para = None

        # dynamic parameter index
        self.num_of_cell_in_tray = NUM_CELL_IN_TRAY
        self.voltage_index = [x for x in range(1, self.num_of_cell_in_tray + 1)]
        self.current_index = [x for x in range(self.num_of_cell_in_tray + 1, self.num_of_cell_in_tray * 2 + 1)]
        self.capacity_index = [x for x in range(self.num_of_cell_in_tray * 2 + 1, self.num_of_cell_in_tray * 3 + 1)]

    def _make_file_dir(self):
        """"""
        #################################
        # Raw Data Fold Structure
        # 2600P-01
        # |
        # | -- dynamic
        # |    |
        # |    | -- 210301-1
        # |    |    |
        # |    |    | -- 0397
        # |    |    |    |
        # |    |    |    | -- 0397-1.csv
        # |    |    |    | -- 0397-2.csv
        # |    |    |    | -- ...
        # |    |    | -- 0640
        # |    |    | -- ...
        # |    |
        # |    | -- 210301-2
        # |    | -- ...
        # |
        # | -- static
        #      |
        #      | -- 210301-1.csv
        #      | -- 210301-2.csv
        #      | -- ...
        #################################
        # Output Data Fold Structure
        # 2600P-01_DataSet
        # |
        # | -- organized_data
        # |    |
        # |    | -- pickle
        # |    | -- xlsx
        #################################
        assert os.path.exists(self.in_path)
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
            og_path = os.path.join(self.out_path, 'organized_data')
            os.mkdir(og_path)
            og_sub_list = ['pickle', 'xlsx']
            for fold in iter(og_sub_list):
                temp_path = os.path.join(og_path, fold)
                os.mkdir(temp_path)

    def _get_tray_to_process(self) -> list:
        """
        static_df: pd.DataFrame, batch: str, tray: str, tray_path: str
        """
        # dynamic data path and batch list
        self.dynamic_data_path = os.path.join(self.in_path, 'dynamic')
        dynamic_fold_list = os.listdir(self.dynamic_data_path)

        # static data path and batch list
        self.static_data_path = os.path.join(self.in_path, 'static')
        static_file_list = os.listdir(self.static_data_path)
        static_batch_list = []
        for s_batch in static_file_list:
            temp_batch_name = s_batch.split('.')[0]
            static_batch_list.append(temp_batch_name)

        # get batch
        batch_list = []
        for batch in iter(dynamic_fold_list):
            if batch in static_batch_list:
                batch_list.append(batch)

        tray_index_per_batch = {}
        for batch in iter(batch_list):
            current_batch_tray_path = os.path.join(self.dynamic_data_path, batch)
            tray_list = os.listdir(current_batch_tray_path)
            tray_index_per_batch.update({f'{batch}': tray_list})

        temp_tray_list = []
        for batch in tray_index_per_batch.keys():
            curr_batch = batch
            temp_batch_tray_list = tray_index_per_batch[curr_batch]
            current_batch_tray_path = os.path.join(self.dynamic_data_path, curr_batch)

            # get static data path and read data file
            current_batch_static_data_index = static_batch_list.index(batch)
            current_batch_static_data_name = static_file_list[current_batch_static_data_index]
            current_batch_static_data_path = os.path.join(self.static_data_path, current_batch_static_data_name)
            current_batch_static_data = self.read_static_data(current_batch_static_data_path)
            for tray in temp_batch_tray_list:
                curr_tray = tray
                curr_tray_path = os.path.join(current_batch_tray_path, curr_tray)
                temp_tray_worker_tup = (current_batch_static_data, curr_batch, curr_tray, curr_tray_path)
                temp_tray_list.append(temp_tray_worker_tup)
        return temp_tray_list

    @staticmethod
    def _save_cell_data(is_write: bool, out_path: str, paras_dic: dict, curr_cell: str):
        """"""
        if is_write:
            write_xlsx_path = os.path.join(out_path + '\\organized_data\\xlsx', curr_cell + '.xlsx')
            if not os.path.exists(write_xlsx_path):
                with pd.ExcelWriter(write_xlsx_path) as writer:
                    # print(f'Write Cell File: {self.curr_cell}.', end='')
                    for name in iter(paras_dic.keys()):
                        paras_dic[name].to_excel(writer, sheet_name=name)
            # write cell pickle file
            write_pickle_path = os.path.join(out_path +
                                             '\\organized_data\\pickle', curr_cell + '.pickle')
            if not os.path.exists(write_pickle_path):
                file_operation.save_dic_as_pickle(write_pickle_path, paras_dic)

    @staticmethod
    def _dynamic_para_extract_cell(cell_no: int, para_data: pd.DataFrame) -> pd.DataFrame:
        """"""
        temp_dic = {}
        # time
        time = para_data.iloc[:, 0]
        temp_dic.update({'time': time})
        # voltage
        voltage = para_data.iloc[:, cell_no]
        temp_dic.update({'voltage': voltage})
        # current
        current = para_data.iloc[:, cell_no + NUM_CELL_IN_TRAY]
        temp_dic.update({'current': current})
        # capacity
        capacity = para_data.iloc[:, cell_no + (2 * NUM_CELL_IN_TRAY)]
        temp_dic.update({'capacity': capacity})
        # build dataframe
        temp_df = pd.DataFrame(temp_dic)
        return temp_df

    @staticmethod
    def _data_verification(curr_cell: str,
                           tolerance: int,
                           static_df: pd.DataFrame,
                           para_df: pd.DataFrame,
                           ver_type: str) -> (bool, str):
        """"""
        # Charge #1, Charge #2, Charge #3, Discharge, Charge #4
        # |            |          |          |          |
        # |-current    |-current  |-capacity |-current  |-capacity
        ver_num = 0
        tgt = 0
        if ver_type in ('Charge #1', 'Charge #2', 'Discharge'):
            ver_type_1 = 'current'
            ver = para_df[ver_type_1].iloc[1:-1].astype('float').max()
            tgt = int(static_df[ver_type + '.1'].iloc[-1])
            ver_rounded = functional.round_precision(float(ver))
            ver_num = int(ver_rounded)
        elif ver_type in ('Charge #3', 'Charge #4'):
            ver_type_1 = 'capacity'
            ver = para_df[ver_type_1].iloc[-1]
            tgt = int(static_df[ver_type].iloc[-1])
            ver_rounded = functional.round_precision(float(ver))
            ver_num = int(ver_rounded)
        else:
            raise ValueError(f'ValueError: ver_type = {ver_type}.')

        diff = abs(ver_num - tgt)
        str_list = ''
        ver_flag = True
        if diff > tolerance:
            str_list = f"Cell:{curr_cell}, Para type:{ver_type}, Ver type:{ver_type_1}, Diff:{diff}, Ver Val:{ver}"
            print(str_list)
            ver_flag = False
            # cell_index = ver[0]
        return ver_flag, str_list

    def _tray_worker(self, static_data: pd.DataFrame, batch: str, tray: str, tray_path: str) -> dict:
        """"""
        current_batch_static_data = static_data
        curr_batch = batch
        curr_tray = tray
        current_batch_tray_path = tray_path

        if self.is_multi_worker:
            print('\n' + '# {} Processing Start, PID: {}'.format(curr_batch + '_' + curr_tray, os.getpid()))

        err_para_cells_list = []
        in_complete_tray_list = []

        tray_name = self.tray_name_adjust(curr_tray)
        current_tray_static_data = current_batch_static_data[current_batch_static_data['Tray ID']
                                                             == tray_name]
        assert current_tray_static_data.shape[0] == self.num_of_cell_in_tray

        current_batch_tray_params_path = current_batch_tray_path
        params_list = os.listdir(current_batch_tray_params_path)
        paras_tray_dic = {}
        for para_tray in iter(params_list):
            # read current batch-tray-parameter
            para_name = self.parameter_name_adjust(para_tray)
            current_para_tray_data = self.read_dynamic_para_tray(current_batch_tray_params_path, para_tray)
            current_para_tray_data.iloc[0, 0] = curr_batch + '-' + para_tray
            paras_tray_dic.update({f'{para_name}': current_para_tray_data})

        # current tray's dynamic data file in-complete, than add it to a list.
        if len(paras_tray_dic.keys()) < 5:
            in_complete_tray_list.append(curr_batch + '_' + curr_tray)

        # progress cell_bar parameter setup
        cell_no = [k for k in range(1, self.num_of_cell_in_tray + 1)]
        cell_total = len(cell_no)
        text = "#{}-Processing..., PID: {}".format(curr_batch + '_' + curr_tray, os.getpid())
        with tqdm(total=cell_total, desc=text) as cell_bar:
            for cell in iter(cell_no):
                cell_bar.update()
                paras_dic = {}
                curr_cell = curr_batch + '_' + curr_tray + '_' + str(cell)
                temp_cell_static_df = current_tray_static_data[current_tray_static_data['Cell No'] == str(cell)]

                # if cell data are 'Non Cell' or 'No Input Cell', drop it.
                temp_cell_remark = temp_cell_static_df['Remark'].item()
                if temp_cell_remark in ['Non Cell', 'No Input Cell']:
                    err_para_cells_list.append(f'Cell: {curr_cell}, Remark: {temp_cell_remark}.')
                    continue

                # get cell's static data
                static_df = temp_cell_static_df.copy()
                static_df_cp = static_df
                paras_dic.update({'Static': static_df_cp})

                for para in iter(paras_tray_dic.keys()):
                    curr_para = para
                    # para extract mode:time, voltage, current, capacity
                    para_df = self._dynamic_para_extract_cell(cell, paras_tray_dic[para])
                    para_ver_flag, ver_str = self._data_verification(curr_cell, 10, static_df_cp, para_df, para)

                    if not para_df.empty and para_ver_flag:
                        paras_dic.update({f'{para}': para_df.copy()})
                        # update paras_dic to cell_dic
                    elif not para_ver_flag:
                        err_para_cells_list.append(f'{ver_str}')

                # update cells_dic
                # batch_cells_dic.update({f'{curr_cell}': paras_dic.copy()})

                # save cell data as xlsx and pickle
                self._save_cell_data(self.is_write, self.out_path, paras_dic, curr_cell)

        if self.is_multi_worker:
            print('\n' + '# {} Processing End, PID: {}'.format(curr_batch + '_' + curr_tray, os.getpid()))
        return {'Incomplete Cells List': in_complete_tray_list,
                'Verification Failed Cells': err_para_cells_list}

    def log_file_organie(self, start, end, res: list):
        """"""
        # logging
        now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file_path = os.path.join(self.out_path, now_time + '_' + 'FileOrganizer_Logging.txt')
        consumed_time = end - start

        seperator = '-' * 89
        file_operation.write_txt(log_file_path, seperator)
        text_data = '| Finish Date: {} | Write Flag: {} | Multiprocess Flag: {} | Time Consume: {} s |'.format(
            now_time, self.is_write, self.is_multi_worker, consumed_time)
        file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)

        text_data = ' ' * 20 + ' Incomplete Cells List ' + ' ' * 20
        file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)
        for i in res:
            if (i != {}) and (i['Incomplete Cells List'] != []):
                text_data = i['Incomplete Cells List']
                for text in text_data:
                    file_operation.write_txt(log_file_path, text)
        file_operation.write_txt(log_file_path, seperator)

        text_data = ' ' * 20 + ' Verification Failed Cells ' + ' ' * 20
        file_operation.write_txt(log_file_path, text_data)
        file_operation.write_txt(log_file_path, seperator)
        for i in res:
            if (i != {}) and (i['Verification Failed Cells'] != []):
                text_data = i['Verification Failed Cells']
                for text in text_data:
                    file_operation.write_txt(log_file_path, text)
        file_operation.write_txt(log_file_path, seperator)

    def file_organize_work_in_tray(self):
        """"""
        self._make_file_dir()
        batch_tray_list = self._get_tray_to_process()

        # start processing
        start_time = time.time()
        res = []
        if self.is_multi_worker:
            with mp.Pool(processes=Num_PROCESS) as pool:
                res = pool.starmap(self._tray_worker, batch_tray_list)
        else:
            with tqdm(total=len(batch_tray_list)) as bar:
                bar.set_description('Tray Processing...')
                for para_tup in batch_tray_list:
                    bar.update()
                    res_dic = self._tray_worker(*para_tup)
                    res.append(res_dic)
        end_time = time.time()
        self.log_file_organie(start_time, end_time, res)

    @staticmethod
    def tray_name_adjust(tray: str) -> str:
        """"""
        tray = tray.strip('-')
        non_zero_str = ''
        if len(tray) > 0:
            len_of_tray_id = len('C000009429')
            for i, num in enumerate(iter(tray)):
                if num != '0':
                    non_zero_str = tray[i:]
                    break
            pad_num = len_of_tray_id - len(non_zero_str) - 1
            new_tray = 'C' + '0' * pad_num + non_zero_str
            return new_tray
        else:
            print('Current Tray ID Empty.')
            sys.exit(0)

    @staticmethod
    def parameter_name_adjust(para: str) -> str:
        """"""
        para_name_list = os.path.splitext(para)
        while para_name_list[-1] != '':
            para_name_list = os.path.splitext(para_name_list[0])
        para_name = para_name_list[0]
        new_para_name = ''
        if len(para_name) > 0:
            if '-' in para_name:
                if para_name[-1] == '1':
                    new_para_name = 'Charge #1'
                elif para_name[-1] == '2':
                    new_para_name = 'Charge #2'
                elif para_name[-1] == '3':
                    new_para_name = 'Charge #3'
                elif para_name[-1] == '4':
                    new_para_name = 'Discharge'
                elif para_name[-1] == '5':
                    new_para_name = 'Charge #4'
                else:
                    print('Current Parameter Index Outer Of Range.')
                    sys.exit(0)
            else:
                print('Current Parameter Name Error.')
                sys.exit(0)
            return new_para_name

        else:
            print('Current Parameter Name Empty.')
            sys.exit(0)

    @staticmethod
    def cell_no_to_tray_cell_no(cell_no: float) -> str:
        """"""
        cell_no = int(cell_no)
        alp = ('A', 'B', 'C', 'D',
               'E', 'F', 'G', 'H',
               'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P')
        res = (cell_no - 1) % 16
        times = int((cell_no - 1) / 16)
        tray_cell_no = alp[times] + str(res + 1)
        return tray_cell_no

    @staticmethod
    def read_static_data(static_data_path: str) -> pd.DataFrame:
        """"""
        current_batch_static_data_path = static_data_path
        assert os.path.getsize(current_batch_static_data_path)
        static_data = pd.read_csv(current_batch_static_data_path, low_memory=False, header=0, index_col=0, dtype=str)
        return static_data

    @staticmethod
    def read_dynamic_para_tray(para_path: str, para_name: str) -> pd.DataFrame:
        """"""
        current_para_path = os.path.join(para_path, para_name)
        assert os.path.getsize(current_para_path)
        current_para_data = pd.read_csv(current_para_path)
        return current_para_data


if __name__ == '__main__':
    workspace = 'D:\\workspace\\PycharmProjects\\battery_dataset'
    src_path = os.path.join(workspace, '2600P-01')
    tgt_path = os.path.join(workspace, '2600P-01_DataSet')
    m_file_organizer = FileOrganizer(src_path, tgt_path,
                                     is_multi_worker=False, is_write=False)
    m_file_organizer.file_organize_work_in_tray()
    sys.exit(0)
