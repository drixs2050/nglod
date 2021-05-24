# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseSDF import BaseSDF
from .BasicDecoder import BasicDecoder
from .utils import init_decoder
from .losses import *
from ..utils import PerfTimer
class FeatureVolume(nn.Module):
    def __init__(self, fdim, fsize):
        super().__init__()
        self.fsize = fsize
        self.fdim = fdim
        var = 0.01
        self.fmx = nn.Parameter(torch.randn(1, fdim, fsize, fsize) * var)
        self.sparse = None
        self.padding_mode = 'reflection'
    def forward(self, x):
        N = x.shape[0]
        if len(x.shape) == 3:
            sample_coords = x.reshape(1, N, x.shape[1], 3) # [N, 1, 1, 3]
            sampley = F.grid_sample(self.fmx, sample_coords[...,[0,2]],
                                    align_corners=True, padding_mode=self.padding_mode)[0,:,:,:].transpose(0,1)
        else:
            sample_coords = x.reshape(1, N, 1, 3) # [N, 1, 1, 3]
            sampley = F.grid_sample(self.fmx, sample_coords[...,[0,2]],
                                    align_corners=True, padding_mode=self.padding_mode)[0,:,:,0].transpose(0,1)
        return sampley
class TexSDF(BaseSDF):
    def __init__(self, args, init=None):
        super().__init__(args)
        self.fdim = self.args.feature_dim
        self.fsize = self.args.feature_size
        self.hidden_dim = self.args.hidden_dim
        self.pos_invariant = self.args.pos_invariant
        self.features = FeatureVolume(self.fdim, self.fsize)
        self.interpolate = self.args.interpolate
        self.louts = nn.ModuleList([])
        self.sdf_input_dim = self.fdim + 1
        #if not self.pos_invariant:
        #    self.sdf_input_dim += self.input_dim
        self.num_decoder = 1 if args.joint_decoder else self.args.num_lods
        self.lout = \
            nn.Sequential(
                nn.Linear(self.sdf_input_dim, self.hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1, bias=True),
            )
    def encode(self, x):
        # Disable encoding
        return x
    def sdf(self, x, lod=None, return_lst=False):
        if lod is None:
            lod = self.lod
        feat = self.features(x)
        feat_z = torch.cat([feat, x[...,1:2]], dim=-1)
        return self.lout(feat_z)