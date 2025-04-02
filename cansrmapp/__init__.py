# -*- coding: utf-8 -*-

"""Top-level package for cansrmapp."""

__author__ = 'Mark Kelly'
__email__ = 'm3kelly@ucsd.edu'
__version__ = '0.1.0'
__repo_url__ = 'https://github.com/idekerlab/cansrmapp'
__description__ = 'Mark please fill this out'
__computation_name__ = 'Ignore me'

import os
try: 
    import torch
except ImportError as e :
    #os.environ['LD_LIBRARY_PATH']=''
    print('This is likely a known issue, {}'.format(r'https://github.com/pytorch/pytorch/issues/111469'))
    print('A likely fix is to first use `unset LD_LIBRARY_PATH`.')
    raise(e)
    

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.cuda.manual_seed(8675309)
    torch.cuda.manual_seed_all(8675309)
torch.manual_seed(8675309)
import numpy as np
np.random.seed(8675309)
import random as random
random.seed(8675309)
import pandas as pd
import hashlib
random=random


CPU=torch.device('cpu')
if torch.cuda.is_available() : 
    DEVICE=torch.device('cuda:0')
    print("Detected GPU.")
else : 
    DEVICE=CPU
DTYPE=torch.float


def summarize_random_states():
    # NumPy state summary
    np_state = np.random.get_state()
    np_hash = hashlib.md5(str(np_state[1][:10]).encode()).hexdigest()

    # Python random module state summary
    py_state = random.getstate()
    py_hash = hashlib.md5(str(py_state[1][:10]).encode()).hexdigest()

    # PyTorch state summary
    torch_state = torch.get_rng_state()
    torch_hash = hashlib.md5(torch_state[:10].numpy().tobytes()).hexdigest()

    # PyTorch CUDA state summary (if available)
    if torch.cuda.is_available():
        torch_cuda_state = torch.cuda.get_rng_state()
        torch_cuda_hash = hashlib.md5(torch_cuda_state[:10].cpu().numpy().tobytes()).hexdigest()
        cuda_summary = f" | PyTorch CUDA: {torch_cuda_hash}"
    else:
        cuda_summary = ""

    det=torch.backends.cudnn.deterministic
    bmark=torch.backends.cudnn.benchmark

    return f"NumPy: {np_hash} | Python: {py_hash} | PyTorch: {torch_hash}{cuda_summary}  Det.:{det} Bmark:{bmark}"

# Print summarized random states

