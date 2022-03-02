#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
# optimizer set
import torch


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, optimizer, model_size, factor=1, warmup=4000):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        # self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def optimizer_select_and_init(model,
                              type_optimizer: str = 'SGD',
                              type_scheduler: str = 'StepLR',
                              op_init_dict: dict = None,
                              sch_init_dict: dict = None) -> dict:
    """
    """

    optimizer = None
    scheduler = None

    # setup optimizer
    if type_optimizer == 'SGD':
        """
        op_init_dict for SGD
        op_init_dict = {
            'lr': float,
            'momentum': float,
            'weight_decay': float,
            'dampening': float,
            'nesterov': bool
        }
        """
        optimizer = torch.optim.SGD(model.parameters(),
                                    **op_init_dict)
    elif type_optimizer == 'Adam':
        """
        op_init_dict for Adam
        op_init_dict = {
            'lr': float,
            'betas': Tuple[float, float],
            'eps': float,
            'weight_decay': float,
            'amsgrad': bool
        }
        """
        optimizer = torch.optim.Adam(model.parameters(),
                                     **op_init_dict)
    else:
        raise ValueError('Optimizer Type Error.')

    # setup scheduler
    if type_scheduler == 'StepLR':
        """
        sch_init_dict for StepLR
        sch_init_dict = {
            'step_size': int,
            'gamma': float,
            'last_epoch': float,
            'verbose': bool
        }
        """
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    sch_init_dict['step_size'],
                                                    gamma=sch_init_dict['gamma'],
                                                    last_epoch=sch_init_dict['last_epoch'],
                                                    verbose=sch_init_dict['verbose'])
    elif type_scheduler == 'ExponentialLR':
        """
        sch_init_dict for Noam
        sch_init_dict = {
            'gamma': float, 
            'last_epoch': int, 
            'verbose': bool
        }
        """
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           sch_init_dict['gamma'],
                                                           last_epoch=sch_init_dict['last_epoch'],
                                                           verbose=sch_init_dict['verbose'])
    elif type_scheduler == 'Noam':
        """
        sch_init_dict for Noam
        sch_init_dict = {
            'model_size': int, 
            'factor': float, 
            'warmup': int
        }
        """
        scheduler = NoamOpt(optimizer, **sch_init_dict)
    elif type_scheduler is None:
        scheduler = None
    else:


        raise ValueError('Scheduler Type Error.')

    # form into dict
    op_dict = {
        'type_optimizer': type_optimizer,
        'type_scheduler': type_scheduler,
        'op_init_dict': op_init_dict,
        'sch_init_dict': sch_init_dict,
        'optimizer': optimizer,
        'scheduler': scheduler
    }
    return op_dict


if __name__ == '__main__':
    import sys
    import torch.nn as nn

    in_dim = 128
    n_samples = 100000

    data = torch.randn((n_samples, in_dim), dtype=torch.float32)
    label = torch.randn((n_samples, 1), dtype=torch.float32)

    # model define
    m_model = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    criterion = nn.MSELoss(reduction='mean')
    # optimizer define
    m_op_type = 'Adam'
    m_op_init_dict = {
        'lr': 0.01,
        'betas': (0.9, 0.98),
        'eps': 1e-9,
        'weight_decay': 0,
        'amsgrad': False
    }

    # scheduler define
    m_sch_type = 'Noam'
    m_sch_init_dict = {
         'model_size': in_dim,
            'factor': 2,
            'warmup': 4000
    }

    m_op = optimizer_select_and_init(m_model,
                                     m_op_type,
                                     m_sch_type,
                                     m_op_init_dict,
                                     m_sch_init_dict)
    for idx in range(n_samples):
        temp_data = data[idx, :]
        temp_label = label[idx, :]
        m_op['optimizer'].zero_grad()
        temp_out = m_model(temp_data)
        temp_loss = criterion(temp_label, temp_out)
        loss_scalar = temp_loss.item()
        lr_scalar = m_op['optimizer'].state_dict()['param_groups'][0]['lr']
        m_op['optimizer'].step()
        if m_op['scheduler'] is not None:
            m_op['scheduler'].step()
        print(f'Step {idx}, lr: {lr_scalar}, Loss: {loss_scalar}')
    sys.exit(0)