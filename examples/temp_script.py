#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang


if __name__ == '__main__':

    import optuna
    import pickle
    import os
    import sys


    base_path = 'D:\\workspace\\PycharmProjects'
    file_name = '20210823-215051_Study.pkl'
    with open(os.path.join(base_path, file_name), 'rb') as f:
        study = pickle.load(f)
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.savefig('.//study_his.png')
    sys.exit(0)




