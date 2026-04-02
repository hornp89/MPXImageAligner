# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 2023

Copyright (C) 2023 Agam Chopra
modified by: Paul Horn, 2026

Based on TorchRegister (https://github.com/AgamChopra/TorchRegister)
Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
"""
import torch
import torch.nn.functional as F
from tqdm import trange

import gc

from mpximagealigner.torchregister.utils import Regressor, AffineRegressor, NCCLoss, affine_warp_tiled

# Warping Function #
def get_affine_warp(theta, moving):
    if len(theta.shape) == 2:
        if theta.shape[-1] == 6:
            theta = theta.view(1, 2, 3)
        else:
            theta = theta.view(1, 3, 4)
    grid = F.affine_grid(theta, moving.size(), align_corners=False)
    warped = F.grid_sample(moving, grid, align_corners=False, mode='bilinear')
    return warped

# Affine Registration #
def affine_register(moving, target, lr=1, epochs=5, tile_size=4096, device='cpu'):

    regressor = AffineRegressor(moving, device=device).to(device=device)
    
    params = regressor.parameters()

    optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=20, line_search_fn='strong_wolfe')
    
    criterions = [NCCLoss(device=device)]
    weights = [1.0]
    
    regressor.train()
    losses_train = []

    def closure(optimizer=optimizer, regressor=regressor, moving=moving, 
                target=target, criterions=criterions, weights=weights):
        optimizer.zero_grad(set_to_none=True)

        theta = regressor()  # 3D Affine Matrix
        warped = affine_warp_tiled(theta, moving, tile_size=tile_size)
        error = sum([weights[i] * criterions[i](target, warped)
                    for i in range(len(criterions))])
        error.backward()

        return error
    
    for eps in trange(epochs):
        error = optimizer.step(closure)
        with torch.no_grad():
            theta = regressor()

        losses_train.append(error.item())

        if eps == 0:
            loss_low = error.item()
            best_theta = theta.detach()
        else:
            if error.item() < loss_low:
                loss_low = error.item()
                best_theta = theta.detach()
    # cleanup
    del regressor, optimizer, criterions
    gc.collect()
    
    return None, [best_theta], [losses_train]


# Rigid Registration #
def rigid_register(moving, target, lr=1E-5, epochs=1000, tile_size=4096, device='cpu'):

    regressor = Regressor(moving, device)
    
    params = regressor.parameters()
    optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=20, line_search_fn='strong_wolfe')
    
    
    criterions = [NCCLoss()]
    weights = [1.0]

    
    def closure(optimizer=optimizer, regressor=regressor, moving=moving, 
                target=target, criterions=criterions, weights=weights):
        optimizer.zero_grad(set_to_none=True)

        theta = regressor()  # 3D Affine Matrix
        warped = affine_warp_tiled(theta, moving, tile_size=tile_size)
        error = sum([weights[i] * criterions[i](target, warped)
                    for i in range(len(criterions))])
        error.backward()

        return error
    
    losses_train = []

    for eps in trange(epochs):
        error = optimizer.step(closure)
        with torch.no_grad():
            theta = regressor()

        losses_train.append(error.item())

        if eps == 0:
            loss_low = error.item()
            best_theta = theta.detach()
        else:
            if error.item() < loss_low:
                loss_low = error.item()
                best_theta = theta.detach()
    # cleanup
    del regressor, optimizer, criterions
    gc.collect()
    
    return None, [best_theta], [losses_train]
