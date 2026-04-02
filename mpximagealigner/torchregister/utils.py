# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 2023

Copyright (C) 2023 Agam Chopra
modified by: Paul Horn, 2026

Based on TorchRegister (https://github.com/AgamChopra/TorchRegister)
Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
"""
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

EPSILON = 1E-10


class NCCLoss(nn.Module):
    '''
    Simple implementation for Normalized Cross Correlation that can be
    minimized with upper-bound of alpha and lower-bound of 0.
    '''

    def __init__(self, alpha=1.0, grad_edges=True, device='cpu'):
        super(NCCLoss, self).__init__()
        self.NCC = None  # -1(very dissimilar) to 1(very similar)
        self.alpha = alpha

    def forward(self, y, yp):
        y_ = y - torch.mean(y)
        yp_ = yp - torch.mean(yp)
        ncc = torch.sum(
            y_ * yp_) / (((torch.sum(y_**2)) * torch.sum(yp_**2) + EPSILON)**0.5)
        self.NCC = ncc.detach()

        error = (1 - ncc) * self.alpha

        return error


class Theta(nn.Module):
    def __init__(self):
        super(Theta, self).__init__()
        self.sin = torch.sin
        self.cos = torch.cos
        self.tanh = torch.tanh

    def forward(self, x, max_translate=0.25):
        if len(x) > 3:
            psi, theta, phi = x[0], x[1], x[2]
            output = torch.stack((self.cos(psi) * self.cos(theta),
                                  self.sin(
                                      phi) * self.sin(psi) * self.cos(theta) - self.cos(phi) * self.sin(theta),
                                  self.cos(
                                      phi) * self.sin(psi) * self.cos(theta) + self.sin(phi) * self.sin(theta),
                                  max_translate * self.tanh(x[3]),
                                  self.cos(psi) * self.sin(theta),
                                  self.sin(phi) * self.sin(psi) * self.sin(theta) +
                                  self.cos(phi) * self.cos(theta),
                                  self.cos(phi) * self.sin(psi) * self.sin(theta) -
                                  self.sin(phi) * self.cos(theta),
                                  max_translate * self.tanh(x[4]),
                                  - self.sin(psi),
                                  self.sin(phi) * self.cos(psi),
                                  self.cos(phi) * self.cos(psi),
                                  max_translate * self.tanh(x[5]))).flatten()
        else:
            theta = x[0]
            output = torch.stack((self.cos(theta), - self.sin(theta), x[1], 
                                  self.sin(theta), self.cos(theta), x[2])).flatten()
        return output


class Regressor(nn.Module):
    def __init__(self, moving, device):
        super(Regressor, self).__init__()
        # if len(moving.shape) == 5:
        #     self.reg = nn.Parameter(torch.rand(
        #         (6), device=device), requires_grad=True)
        # else:
        #     self.reg = nn.Parameter(torch.rand(
        #         (3), device=device), requires_grad=True)
        if len(moving.shape) == 5:
            self.reg = nn.Parameter(torch.zeros(
                (6), device=device), requires_grad=True)
        else:
            self.reg = nn.Parameter(torch.zeros(
                (3), device=device), requires_grad=True)

        self.thetas = Theta()


    def forward(self):
        var = self.reg
        theta = self.thetas(var)
        if theta.shape[-1] == 12:
            return theta.view(1, 3, 4)
        else:
            return theta.view(1, 2, 3)

class AffineRegressor(nn.Module):
    def __init__(self, moving, device):
        super().__init__()
        if len(moving.shape) == 5:
            # 3D: full 3x4 = 12 params, initialized to identity
            self.reg = nn.Parameter(torch.tensor(
                [1,0,0,0, 0,1,0,0, 0,0,1,0], dtype=torch.float, device=device))
        else:
            # 2D: full 2x3 = 6 params, initialized to identity
            self.reg = nn.Parameter(torch.tensor(
                [1,0,0, 0,1,0], dtype=torch.float, device=device))

    def forward(self):
        if self.reg.shape[0] == 12:
            return self.reg.view(1, 3, 4)
        else:
            return self.reg.view(1, 2, 3)
        
def affine_warp_tiled(theta, source, tile_size=4096, train=True):
    """
    Apply an affine warp to `source` in output tiles to reduce VRAM usage.
    The source image remains in VRAM; only one tile's grid and output are
    allocated at a time. The result is assembled on CPU as a uint16 numpy array.

    Args:
        theta (torch.Tensor): Affine parameters, shape (1, 2, 3) or (2, 3).
        source (torch.Tensor): Full-resolution source image, shape (1, 1, H, W),
            float32, on the target device.
        tile_size (int): Output tile size in pixels (height and width).

    Returns:
        np.ndarray: Warped image as uint16, shape (H, W), on CPU.
    """
    H, W = source.shape[2], source.shape[3]
    device = source.device
    t = theta.view(2, 3)  # (2, 3)

    #output = np.empty((H, W), dtype=np.uint16)
    output = torch.empty((H, W), dtype=torch.float32, device=device)

    for r0 in range(0, H, tile_size):
        th = min(tile_size, H - r0)
        for c0 in range(0, W, tile_size):
            tw = min(tile_size, W - c0)

            # Normalized output coords for this tile's pixels in full-image space
            # align_corners=False: pixel j → (2j + 1)/W - 1
            x_out = (2 * torch.arange(c0, c0 + tw, device=device, dtype=torch.float32) + 1) / W - 1
            y_out = (2 * torch.arange(r0, r0 + th, device=device, dtype=torch.float32) + 1) / H - 1
            yy, xx = torch.meshgrid(y_out, x_out, indexing='ij')  # (th, tw)

            # Apply theta: source coords (still in full-image normalized space)
            x_src = t[0, 0] * xx + t[0, 1] * yy + t[0, 2]
            y_src = t[1, 0] * xx + t[1, 1] * yy + t[1, 2]

            # grid_sample expects (N, H_out, W_out, 2) with [..., 0]=x, [..., 1]=y
            grid = torch.stack([x_src, y_src], dim=-1).unsqueeze(0)  # (1, th, tw, 2)

            tile = F.grid_sample(source, grid, mode='bilinear',
                                 padding_mode='zeros', align_corners=False)
            output[r0:r0 + th, c0:c0 + tw] = tile[0, 0]

            del grid, tile, x_src, y_src, xx, yy, x_out, y_out
    if train:
        return output
    else:
        return output.detach().cpu().squeeze().numpy().astype(np.uint16)