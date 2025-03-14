import torch
import time
import sys
import builtins
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
    if hasattr(builtins,'__IPYTHON__') : 
        clear=False
        if kwargs.get('end') == '\r'  : clear=True
        display(' '.join([time.strftime("%m%d %H:%M:%S",time.localtime())+'|',*args]),clear=clear)
    else : 
        print(time.strftime("%m%d %H:%M:%S",time.localtime())+'|',*args,**kwargs)
        time.sleep(0) # for issues with ipython interpreters
        sys.stdout.flush()


def diag_embed(thetensor,minorly=False) :

    if not minorly :
        use_indices=thetensor.indices()
        outindices=torch.stack([use_indices[0],*use_indices],axis=0)
        outshape=(thetensor.shape[0],*tuple(thetensor.shape))
    else : 
        use_indices=thetensor.indices()
        outindices=torch.stack([*use_indices,use_indices[-1]],axis=0)
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


def _fix_overlong_identifiers(df,index=True) :
    """
    TCGA sample identifiers will specifiy particular specimens, aliquots, etc
    This reduces identifiers to the first three items, which uniquely specify
    only the patient, and no other features of the sample.
    """
    if index: 
        df.index=[ '-'.join(x.split('-')[:3]) for x in df.index ]
        assert len(df.index) == len(set(df.index))
    else : 
        df.columns=[ '-'.join(x.split('-')[:3]) for x in df.columns ]
        assert len(df.columns) == len(set(df.columns))

    return df

def regularize_cc(ccmat,n,correlation_p=1e-3) : 
    """
    Takes a correlation matrix (numpy ndarray) and zeroes
    out correlations with p-values less significant than <correlation_p>
    """
    from scioy.stats.distributions import t
    tstats=ccmat*np.sqrt(n-2)/np.sqrt(1-ccmat*ccmat)
    tthresh=t.isf(correlation_p,df=n-2)
    sigmask=tstats > tthresh
    out=ccmat.copy()
    out[ ~sigmask ]=0
    
    return out

def regularize_cc_torch(ccten,n,correlation_p=1e-3) : 
    """
    Takes a correlation matrix (torch.Tensor) and zeroes
    out correlations with p-values less significant than <correlation_p>
    """
    from scipy.stats.distributions import t
    tstats=ccten*torch.sqrt(torch.tensor(n,device=ccten.device)-2)/torch.sqrt(1-ccten*ccten)
    tthresh=t.isf(correlation_p,df=n-2)
    sigmask=tstats > tthresh
    out=ccten.clone()
    out[ ~sigmask ]=0
    
    return out


def ravel_multi_index(multi_index, dims):
    """
    Convert a tuple of tensor indices into a flat index, given the dimensions of the array.
    
    Parameters:
    multi_index (tuple of torch.Tensor): Indices for each dimension.
    dims (torch.Size or tuple): Shape of the tensor.
    
    Returns:
    torch.Tensor: The flattened index.

    Authored using ChatGPT 3.0
    """
    # Ensure dims is a tensor
    if not isinstance(dims, torch.Tensor):
        dims = torch.tensor(dims, dtype=torch.long)
    
    # Calculate strides
    strides = torch.cumprod(dims.flip(0), 0).flip(0)
    strides = torch.cat((strides.to(device=multi_index.device)[1:], torch.tensor([1],device=multi_index.device))).to(multi_index.device)
    
    # Convert multi_index to tensor if not already
    if isinstance(multi_index, tuple):
        multi_index = torch.stack(multi_index)
    
    # Compute the flat index
    flat_index = torch.sum(multi_index * strides.unsqueeze(-1), dim=0)
    return flat_index

def cc2mats(m1,m2) :
    """
    This function finds the correlation coefficient between all **columns** of two matrices m1 and m2.
    (The rows are the observations measured across all variables in both columns).
    This does _not_ show the correlation coefficient between any two variables that are both columns of the 
    same matrix.
    """

    from scipy.stats.distributions import t
    m1bar=m1.mean(axis=0)
    m2bar=m2.mean(axis=0)
    m1sig=m1.std(axis=0)
    m2sig=m2.std(axis=0)
    n=m1.shape[0]

    cov=np.dot(
            (m1-m1bar).transpose(),
            (m2-m2bar)
            )
    with warnings.catch_warnings() : 
        warnings.simplefilter('ignore')  
        cov1=cov/m2sig
        cov1[ np.isnan(cov1) ]=0
        cov2=cov1.transpose()/m1sig
        cov2[ np.isnan(cov2) ]=0
        out=cov2.transpose()/n

    return out

def cc2tens(t1,t2) : 
    """
    This function finds the correlation coefficient between all **columns** of two dense tensors t1 and t2.
    (The rows are the observations measured across all variables in both columns).
    This does _not_ show the correlation coefficient between any two variables that are both columns of the 
    same matrix.
    """
    t1=t1.cpu()
    t2=t2.cpu()
    if t1.is_sparse: 
        t1=t1.clone().to_dense()
    if t2.is_sparse: 
        t2=t2.clone().to_dense()


    t1bar=t1.mean(axis=0)
    t2bar=t2.mean(axis=0)
    t1sig=t1.std(axis=0)
    t2sig=t2.std(axis=0)
    n=t1.shape[0]

    cov=torch.matmul(
            (t1-t1bar).transpose(0,1),
            (t2-t2bar)
            )
    cov1=cov/t2sig
    cov1[ torch.isnan(cov1) ]=0
    cov2=cov1.transpose(0,1)/t1sig
    cov2[ torch.isnan(cov2) ]=0
    out=cov2.transpose(0,1)/n
    return out


def flatten_majorly(thetensor) :
    """
    Flattens sparse tensor thetensor maintaining the first (=0th) axis and flattening the remainder
    """

    use_indices=thetensor.indices()
    use_shape=_tcast(thetensor.shape)

    outsize=(use_shape[0],torch.prod(use_shape[1:]))
    ri=ravel_multi_index(use_indices[1:],dims=use_shape[1:])
    maj_i=use_indices[0]

    outindices=torch.stack([maj_i,ri],axis=0)

    return torch.sparse_coo_tensor(indices=outindices,values=thetensor.values(),size=outsize).coalesce()

def flatten_minorly(thetensor) : 
    """
    Flattens sparse tensor thetensor maintaining the last (-1th) axis and flattening the remainder
    """

    use_indices=thetensor.indices()
    use_shape=torch.tensor(thetensor.shape)

    outsize=(torch.prod(use_shape[:-1]),use_shape[-1])
    ri=ravel_multi_index(use_indices[:-1],dims=use_shape[:-1])
    maj_i=use_indices[-1]

    outindices=torch.stack([ri,maj_i],axis=0)

    return torch.sparse_coo_tensor(indices=outindices,values=thetensor.values(),size=outsize).coalesce()


    
def sproing(theflattensor,minor_dims,minorly=False) :
    """
    Unflattens a flat tensor so that its second dimension (indexed 1) is
    distributed among the dimensions listed in minor_dims.
    ^^ If "minorly", otherwise so that first dimension is distributed among minor dims.
    """

    tfts=_tcast(theflattensor.shape)

    if minorly : 
        inindices=theflattensor.indices()
        major=inindices[0]
        rest=inindices[1]

        #print(theflattensor.shape[0]*np.prod(minor_dims),np.prod(theflattensor.shape))
        assert theflattensor.shape[0]*torch.prod(_tcast(minor_dims))==torch.prod(tfts)

        sproinged_rest=torch.unravel_index(rest,shape=minor_dims)

        outindices=torch.stack([major,*sproinged_rest],axis=0)
        outsize=(theflattensor.shape[0],*minor_dims)

    else : 
        inindices=theflattensor.indices()
        major=inindices[1]
        rest=inindices[0]

        #print(theflattensor.shape[0]*np.prod(minor_dims),np.prod(theflattensor.shape))
        assert theflattensor.shape[1]*torch.prod(_tcast(minor_dims))==torch.prod(tfts)

        sproinged_rest=torch.unravel_index(rest,shape=minor_dims)

        outindices=torch.stack([*sproinged_rest,major],axis=0)
        outsize=(*minor_dims,theflattensor.shape[1])
    
    return torch.sparse_coo_tensor(indices=outindices,
        values=theflattensor.values(),
        size=outsize).coalesce()


def rotate(thetensor) : 
    in_indices=thetensor.indices()
    use_indices=torch.stack([*in_indices[1:],in_indices[0]],axis=0)
    return torch.sparse_coo_tensor(
            indices=use_indices,
            values=thetensor.values(),
            size=(*thetensor.shape[1:],thetensor.shape[0])).coalesce()

def as_directsum(theflattensor,n_blocks,minorly=False) :

    if minorly : 
        # viz. preserving the minor dimension as is
        assert theflattensor.shape[0] % n_blocks ==0 

        use_indices=theflattensor.indices().clone()
        #n_blocks=theflattensor.shape[-1]//blocksize
        blocksize=theflattensor.shape[-1]//n_blocks
        out_shape=(theflattensor.shape[0]*n_blocks,theflattensor.shape[1])

        use_indices[0]=theflattensor.shape[0]*(use_indices[1]//blocksize)+use_indices[0]
    else : 
        key_modulus=theflattensor.shape[0] % n_blocks
        if not ( key_modulus ==0 ) : 
            raise ValueError('flat tensor shape incopatible with n blocks; ',theflattensor.shape,n_blocks,key_modulus)



        use_indices=theflattensor.indices().clone()
        blocksize=theflattensor.shape[0]//n_blocks
        out_shape=(theflattensor.shape[0],theflattensor.shape[1]*n_blocks)

        use_indices[-1]=theflattensor.shape[-1]*(use_indices[0]//blocksize)+use_indices[1]

    return torch.sparse_coo_tensor(
                indices=use_indices,
                values=theflattensor.values().clone(),
                size=out_shape).coalesce() ;



def bmm(mat1,mat2) :
    """
    An admittedly limited copy of torch's bmm modified to use two sparse matrices.
    Satisfies the following use cases : 

    mat1=(k*m*n), mat2=(n*p) ==> (k*m*p)
    mat1=(k*m*n)  mat2=(k*n*p) ==> (k*m*p)
    mat1=(j*k*m*n) mat2=(n*p) ==> (j*k*n*p)
    mat1=(j*k*m*n) mat2=(k*n*p) ==> (j*k*n*p)
    mat1=(j*k*m*n) mat2=(j*k*n*p) ==> (j*k*n*p)
    """
    if not mat1.shape[-1] == mat2.shape[-2] : 
        raise ValueError('Incompatible dimensions between mat1 (',mat1.shape,') and mat2 (',mat2.shape,')')

    lsm1=len(mat1.shape)
    lsm2=len(mat2.shape)

    if not any([ (lsm1,lsm2) == t for t in {(3,2),(3,3),(4,2),(4,3),(4,4)} ]) : 
        raise ValueError('mat1 of shape',mat1.shape,' and mat2 of shape',mat2.shape,'not supported by sptops.bmm')
        
    # common operations for cases 1-2 and 3-5
    if lsm1 == 3 : 
        f1=flatten_minorly(mat1.coalesce())
        op1=as_directsum(f1,n_blocks=mat1.shape[0],minorly=False)

    else : 
        f1=flatten_minorly(mat1.coalesce())
        op1=as_directsum(f1,n_blocks=torch.prod(mat1.shape[:-2]))

    if lsm2 == 2  : 
        times_to_multiply=torch.prod(mat1.shape[:-2]) # viz. k or j*k times
        op2=flat_sparse_tensor_tile(mat2,times_to_multiply).to(dtype=mat2.dtype).coalesce()
    elif lsm2 ==3 and lsm1==4 : 
        print('warning! not sure I\'m going to test this use case.')
        times_to_multiply=mat1.shape[0] # viz. j
        op2=flat_sparse_tensor_tile(flatten_secondarily(mat2),times_to_multiply)
    elif lsm2 == 3 and lsm1==3 :
        op2=flatten_minorly(mat2).coalesce()
    elif lsm2 == 4 :
        op2=flatten_minorly(flatten_minorly(mat2))


    try : 
        assert op1.shape[-1] == op2.shape[0]
        theprod=torch.matmul(op1,op2).coalesce()
    except Exception as e : 
        print(lsm1,lsm2,mat1.shape,mat2.shape,op1.shape,op2.shape)
        raise e 

    if (lsm1,lsm2) in {(3,2),(3,3)} : 
        # since we are flattening the first tensor "secondarily"
        #print(theprod.shape)
        out=sproing(theprod,tuple(mat1.shape[:-1]))
    if (lsm1,lsm2) in {(4,2),(4,3),(4,4)} : 
        out=sproing(theprod,tuple(mat1.shape[:-1]))

    return out


    
        
    

