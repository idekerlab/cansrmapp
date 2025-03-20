from . import pd,np 
#GENEINFOPATH='/cellar/users/mrkelly/Data/cansrmapp_project/src/cansrmapp/bioinfo_redistributed/Homo_sapiens_241103.gene_info'
#GFFPATH='/cellar/users/mrkelly/Data/cansrmapp_project/src/cansrmapp/bioinfo_redistributed/gff_reduced.gff.gz'
GENEINFOPATH='/cellar/users/mrkelly/Data/cansrmapp_project/src/cansrmapp/cansrmapp/bioinfo_redistributed/Homo_sapiens_241103.gene_info'
GENEINFOPATH='./bioinfo_redistributed/Homo_sapiens_241103.gene_info'
GFFPATH='./bioinfo_redistributed/gff_reduced.gff.gz'
_gi=None
_s2e=None
_e2s=None
_ens2e=None
_e2ens=None
_synonyms2e=None

_gff=None

def boot_bioinfo() : 
    if _gi is not None : return
    _get_geneinfo()

def boot_gff() : 
    if _gff is not None : return
    _get_gff()




@np.vectorize
def get_ensembl_xref(dbxrefs) : 
    for xref in dbxrefs.split('|') : 
        subfields=xref.split(':')
        if subfields[0] == 'Ensembl' : 
            return subfields[1]  
    else :
        return None

def _get_geneinfo() : 
    global _gi 
    global _s2e
    global _e2s
    global BADGENES
    global _synonyms2e
    global _ens2e
    global _e2ens

    _gi=pd.read_csv(GENEINFOPATH,sep='\t')[::-1] 
    _gi['Ensembl']=get_ensembl_xref(_gi.dbXrefs)
    _gi['GeneID']=_gi.GeneID.astype(str)
    # [::-1] this means that for items iterating through, "older"/more canonical entries will be last and supersede shakier ones


    _e2s=dict()
    _s2e=dict()
    _ens2e=dict()
    _e2ens=dict()

    _synonyms2e=dict()

    for r in _gi.itertuples() :
        _e2s.update({ r.GeneID : r.Symbol })
        _e2ens.update({ r.GeneID : r.Ensembl})
        _ens2e.update({ r.Ensembl : r.GeneID})
        _s2e.update({ r.Symbol : r.GeneID })
        if r.Synonyms == '-' : continue
        for syn in r.Synonyms.split('|') : 
            _synonyms2e.update({ syn : r.GeneID })

@np.vectorize
def expandgfftags(tagstring) : 
    tags=tagstring.split(';')
    td=dict()
    for tag in tags: 
        ts=tag.split('=')
        td.update({ ts[0] : ts[1] })
    return td
    
@np.vectorize
def expand_xrefs(xrefstring) : 
    xrefs=[ xr.split(':')[-2:] for xr in xrefstring.split(',') ]
    xrd=dict(xrefs)

    return xrd

@np.vectorize
def acc_to_chr(x) : 
    xsp=x.split('.')[0].split('0')
    chrno=xsp[-1]
    if chrno == '' : 
        chrno = ''.join([xsp[-2],'0'])
    if chrno == '23' :
        return 'chrX'
    if chrno == '24' : 
        return 'chrY'

    try : 
        int(chrno)
    except ValueError : 
        return ''
    if int(chrno) > 24 : return ''
    return 'chr'+chrno


def _get_gff() : 
    global _gff

    proto_gff=pd.read_csv(GFFPATH,
        sep='\t',
        comment='#',
        names=['acc','src','kind','start','stop',
            'foo1','strand','foo2','tags']
            ).dropna(subset='tags').query('kind == "gene"')

    meso_gff=proto_gff.join(pd.DataFrame.from_records(expandgfftags(proto_gff.tags),index=proto_gff.index))
    meso_gff=meso_gff.dropna(subset='Dbxref')
    meso_gff=meso_gff.join(pd.DataFrame.from_records(expand_xrefs(meso_gff.Dbxref),index=meso_gff.index))
    meso_gff['chromosome']=acc_to_chr(meso_gff.acc)
    meso_gff=meso_gff.query('chromosome != ""').dropna(subset='GeneID').drop_duplicates(subset='GeneID')

    _gff=meso_gff



