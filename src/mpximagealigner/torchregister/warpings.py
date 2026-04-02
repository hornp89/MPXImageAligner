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
def affine_register(moving, target, lr=1, epochs=5, tile_size=4096, random_starts=24, seed=0,
                    init_params=None, device='cpu'):
    
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    global_best_loss = float("inf")
    global_best_theta = None
    global_best_curve = None
    save_p = None
    
    for start in trange(random_starts):
        regressor = Regressor(moving, device)
        if init_params is not None:
            with torch.no_grad():
                regressor.reg.data = init_params.clone().flatten().to(device)
        
        criterions = [NCCLoss()]
        weights = [1.0]
        
        save_p = regressor.reg.clone()
        with torch.no_grad():
            if start > 0:
                p = regressor.reg
                save_p = p.clone()
                if p.numel() == 3:
                     # 2D rigid: [angle, tx, ty]
                    p[0].uniform_(-0.35, 0.35, generator=gen)   # about +-20 deg
                    p[1].uniform_(-0.20, 0.20, generator=gen)   # normalized translation
                    p[2].uniform_(-0.20, 0.20, generator=gen)
                else:    
                    # 3D rigid: [psi, theta, phi, tx, ty, tz]
                    p[0:3].uniform_(-0.35, 0.35, generator=gen)
                    p[3:6].uniform_(-0.15, 0.15, generator=gen)
                    
        optimizer = torch.optim.LBFGS(regressor.parameters(), lr=lr, max_iter=20, line_search_fn='strong_wolfe')            
        losses_train = []
        run_best_loss = float("inf")
        run_best_theta = None
        
        def create_closure():
            def closure(optimizer=optimizer, regressor=regressor, moving=moving, 
                        target=target, criterions=criterions, weights=weights):
                optimizer.zero_grad(set_to_none=True)

                theta = regressor()  # 3D Affine Matrix
                warped = affine_warp_tiled(theta, moving, tile_size=tile_size)
                error = sum([weights[i] * criterions[i](target, warped)
                            for i in range(len(criterions))])
                error.backward()

                return error
            return closure
        
        closure = create_closure()      
    
        losses_train = []

        for eps in range(epochs):
            error = optimizer.step(closure)
            loss_val = float(error.item())
            losses_train.append(loss_val)
            
        theta = regressor().detach()
        regressor = AffineRegressor(moving, device=device).to(device=device)
        regressor.reg.data = theta.flatten()
    
        params = regressor.parameters()

        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=20, line_search_fn='strong_wolfe')
        
        regressor.train()

        closure = create_closure()

        for eps in range(epochs):
            error = optimizer.step(closure)
            loss_val = float(error.item())
            losses_train.append(loss_val)

            if loss_val < run_best_loss:
                run_best_loss = loss_val
                run_best_theta = regressor().detach()

        if run_best_loss < global_best_loss:
            global_best_loss = run_best_loss
            global_best_theta = run_best_theta
            global_best_curve = losses_train

        del regressor, optimizer, criterions
        gc.collect()

    return None, [global_best_theta], [global_best_curve], save_p


# Rigid Registration #
def rigid_register(moving, target, lr=1E-5, epochs=1000, tile_size=4096, random_starts=12, seed=0, 
                   init_params=None, device='cpu'):

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    global_best_loss = float("inf")
    global_best_theta = None
    global_best_curve = None
    save_p = None
    
    for start in trange(random_starts):
        regressor = Regressor(moving, device)
        if init_params is not None:
            with torch.no_grad():
                regressor.reg.data = init_params.clone().flatten().to(device)
                
        criterions = [NCCLoss()]
        weights = [1.0]
            
        save_p = regressor.reg.clone()
        with torch.no_grad():
            if start > 0:
                p = regressor.reg
                if p.numel() == 3:
                     # 2D rigid: [angle, tx, ty]
                    p[0].uniform_(-0.35, 0.35, generator=gen)   # about +-20 deg
                    p[1].uniform_(-0.20, 0.20, generator=gen)   # normalized translation
                    p[2].uniform_(-0.20, 0.20, generator=gen)
                else:    
                    # 3D rigid: [psi, theta, phi, tx, ty, tz]
                    p[0:3].uniform_(-0.35, 0.35, generator=gen)
                    p[3:6].uniform_(-0.15, 0.15, generator=gen)
                    
        optimizer = torch.optim.LBFGS(regressor.parameters(), lr=lr, max_iter=20, line_search_fn='strong_wolfe')            
        losses_train = []
        run_best_loss = float("inf")
        run_best_theta = None
    
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

        for eps in range(epochs):
            error = optimizer.step(closure)
            loss_val = float(error.item())
            losses_train.append(loss_val)

            if loss_val < run_best_loss:
                run_best_loss = loss_val
                run_best_theta = regressor().detach()

        if run_best_loss < global_best_loss:
            global_best_loss = run_best_loss
            global_best_theta = run_best_theta
            global_best_curve = losses_train

        del regressor, optimizer, criterions
        gc.collect()

    return None, [global_best_theta], [global_best_curve], save_p
