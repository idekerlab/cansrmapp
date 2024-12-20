# -*- coding: utf-8 -*-

"""Top-level package for cansrmapp."""

__author__ = 'Mark Kelly'
__email__ = 'm3kelly@ucsd.edu'
__version__ = '0.1.0'
__repo_url__ = 'https://github.com/idekerlab/cansrmapp'
__description__ = 'Mark please fill this out'
__computation_name__ = 'Ignore me'

import torch
torch.manual_seed(8675309)
import numpy as np
import pandas as pd
CPU=torch.device('cpu')
if torch.cuda.is_available() : 
    DEVICE=torch.device('cuda:0')
    print("Detected GPU.")
else : 
    DEVICE=CPU
DTYPE=torch.float

