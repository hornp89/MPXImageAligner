# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 2023

Copyright (C) 2023 Agam Chopra
modified by: Paul Horn, 2026

Based on TorchRegister (https://github.com/AgamChopra/TorchRegister)
Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
"""
from torch import cat
from mpximagealigner.torchregister.warpings import affine_register, rigid_register, get_affine_warp


class Register():
    def __init__(self, mode='rigid', device='cpu', criterion=None, weight=None):
        '''
        Pytorch based numerical registration methods

        Parameters
        ----------
        mode : string, optional
            'rigid', 'affine', 'flow'. The default is 'rigid'.
        device : string, optional
            device to perform registration/optimization on. The default is 'cpu'.
        debug : string, optional
            print outputs of registration and loss curve every nth epoch. The default is False.
        criterion : list of nn.losses, optional
            criterion used to calculate registration error. The default is None ie- [nn.MSELoss(), nn.L1Loss()].
        weights : list of floats, optional
            weights associated with criterion. The default is None ie- [0.5, 0.5].
        optm : string, optional
            optimizer to use, SGD or ADAM. The default is 'SGD'.


        Returns
        -------
        None.

        '''
        self.criterion = criterion
        self.weight = weight
        self.mode = mode
        self.warp = None if mode == 'flow' else get_affine_warp
        self.device = device
        self.theta = None
        self.losses = None



    def optim(self, moving, target, lr=1, max_epochs=1000, random_starts=24, seed=0, init_params=None):
        '''
        Optimization loop to get deformation matrix/flow-field

        Parameters
        ----------
        moving : tensor
            Tensor of shape [1,1,x,y,z] to be warped.
        target : tensor
            Target tensor of shape [1,1,x,y,z].
        lr : float, optional
            Learning rate. The default is 1E-5.
        max_epochs : int, optional
            Number of optimization iterations. The default is 1000.

        Returns
        -------
        None.

        '''
        if self.mode == 'affine':
            _, theta, losses, save_p = affine_register(moving, target, lr=lr, epochs=max_epochs, init_params=init_params,
                                               random_starts=random_starts, seed=seed,
                                               device=self.device)
            self.theta = theta[-1]
            self.losses = losses[-1]
            self.save_p = save_p

        else:
            _, theta, losses, save_p = rigid_register(moving, target, lr=lr, epochs=max_epochs, init_params=init_params,
                                              random_starts=random_starts, seed=seed, 
                                              device=self.device)
            self.theta = theta[-1]
            self.losses = losses[-1]
            self.save_p = save_p

    def __call__(self, moving):
        '''
        Warp moving image using deformation obtained from optim

        Parameters
        ----------
        moving : tensor
            Tensor of shape [1,c,x,y,z] to be warped.

        Returns
        -------
        warped_moving : tensor
            Warped tensor of shape [1,c,x,y,z].

        '''
        if self.mode == 'flow':
            warped_moving = cat([self.warp(moving[:, i:i+1])
                                 for i in range(moving.shape[1])], dim=1)
        else:
            warped_moving = cat([self.warp(self.theta, moving[:, i:i+1])
                                 for i in range(moving.shape[1])], dim=1)
        return warped_moving
