import torch
import time
import sys
from . import DEVICE
from . import np

def _tcast(arg): 

    if torch.is_tensor(arg) : 
        if torch.get_device(arg) != DEVICE : 
            return arg.to(DEVICE)
        return arg
    elif type(arg) == list : 
        if torch.is_tensor(arg[0]) : 
            return torch.tensor(np.array([a.cpu() for a in arg]),device=DEVICE)
        else : 
            return torch.tensor(np.array(arg),device=DEVICE)
    else : 
        return torch.tensor(arg,device=DEVICE)

def word_to_seed(word) :
    import hashlib
    return int(hashlib.shake_128(word.encode()).hexdigest(4),16)

def msg(*args,**kwargs) : 
    print(time.strftime("%m%d %H:%M:%S",time.localtime())+'|',*args,**kwargs)
    sys.stdout.flush()

def diag_embed(thetensor,minorly=False) :

    if not minorly :
        use_indices=thetensor.indices().numpy()
        outindices=np.stack([use_indices[0],*use_indices],axis=0)
        outshape=(thetensor.shape[0],*tuple(thetensor.shape))
    else : 
        use_indices=thetensor.indices().numpy()
        outindices=np.stack([*use_indices,use_indices[-1]],axis=0)
        outshape=(*tuple(thetensor.shape),thetensor.shape[-1],)

    return torch.sparse_coo_tensor(indices=outindices,values=thetensor.values(),size=outshape).coalesce()

def mask_sparse_rows(t,mask) :
    """
    Utility function. Takes sparse tensor t and returns the same with only rows marked True in mask
    Mask is also a tensor and must be on the same device
    """
    imi=torch.argwhere(mask).ravel()
    new_indices=torch.cumsum(mask,0)-1
    timask=torch.isin(t.indices()[0],imi)
    stvals=t.values()[timask]
    ti=t.indices()[:,timask]
    stindices=torch.stack([new_indices[ti[0]],ti[1]],axis=-1).transpose(0,1)
    st=torch.sparse_coo_tensor(
        values=stvals,
        indices=stindices,
        size=(mask.sum(),t.shape[1]),
        device=t.device,
    )

    return st

def mask_sparse_columns(t,mask) : 
    """
    Utility function. Takes sparse tensor t and returns the same with only columns marked True in mask
    Mask is also a tensor and must be on the same device
    """
    imi=torch.argwhere(mask).ravel()

    new_indices=torch.cumsum(mask,0)-1
    
    timask=torch.isin(t.indices()[1],imi)

    stvals=t.values()[timask]
    
    ti=t.indices()[:,timask]

    stindices=torch.stack([ti[0],new_indices[ti[1]]],axis=-1).transpose(0,1)
    
    st=torch.sparse_coo_tensor(
        values=stvals,
        indices=stindices,
        size=(t.shape[0],mask.sum()),
        device=t.device,
    )

    return st

