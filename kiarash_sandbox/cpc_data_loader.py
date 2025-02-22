# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The data loader for the audio CPC model.

"""

import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset
import os
import sys
import random
import librosa
     
class CPCDataset(Dataset):
    '''
    Dataset class for CPC
    '''
    def __init__(self, x, y):
        '''
        Initializer for CPCDataset. Note that this is a subclass of torch.utils.data.Dataset. \

        Parameters: 
        ------------
        - x: torch.Tensor, shape: (num_seq, num_frames, num_feats), high dimensional neural observations. 
        - y: torch.Tensor, shape: (num_seq, ), labels.
        '''
        super().__init__()
        self.x = x
        self.y = y


    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
