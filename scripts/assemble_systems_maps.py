import pandas as pd
import os

PKGROOT=os.path.abspath('../')
OUTPUTS=os.path.join(PKGROOT,'outputs')

def actpwy_wf(infiles) : 
    pcawg_raw=pd.read_csv(infiles[0],
                      sep='\t',
                      names=['PCAWG_no',
                             'cds',
                             'grouping',
                             'genes',
                             'nearestgo',
                             'unsuallynull1',
                             'enrichmentscore',
                             'usuallynull2']
                          )

    pcawg_raw['sysname']=pcawg_raw.nearestgo.apply(lambda s : s.split('|')[0].strip())

    ph=dict()
    for x,r in pcawg_raw.iterrows() : 
        system_content=set()
        for g in r.genes.split(':') : 
            e=kg.hier.s2eid.get(g) ;
            if e is None : continue
            system_content.add(e)
        ph.update({ r.sysname : system_content })

    return ph


#GO evidence codes https://wiki.geneontology.org/index.php/Guide_to_GO_Evidence_Codes
go_valid_evidence={ 
   'EXP' , # experiment
   'HDA' , # hi-throughput direct assay
   'HMP' , # hi-throughput mutant phenotype
   'IBA' , # "biological aspect of ancestor" 
   'IC'  , # "inferred by curator"
   'IDA' , # direct assay
   'IKR',  # inferred from key residues
   'IMP', # inferred from mutant phenotype
   'RCA', # reviewed computational anlaysis / kind of means bioinformatics applied to exp. data
   'TAS', # Traceable author statement
}

go_invalid_evidence={
    'ND' # no data
    'IEA' , # electronic, no experiment
    'IPI', #inferred from physical interaction
    'ISA', #inferred from sequence alignment
    'ISM' # inferred from seqence model
    'ISO' #inferred from sequence orthology
    'ISS', # inferred from sequence/structural similarity
    'IEP', #inferred from expression pattern --disqualifying from synteny
    'HEP' , # hi-throughput expression pattern  -- disqualifying by syntenty
    'IGI', # inferred from genetic interaction-- the highest-count paper from this straighforwradly reflect ssynteny
    'NAS' , # non-traceable author statement
}


def corum_wf(infiles) :
    corum_raw=pd.read_excel(infiles[0])
    corum_relevant=corum_raw[ corum_raw.Organism.isin({'Mammalia','Human'}) ]
    corum_relevant['components']=corum_relevant['subunits(Entrez IDs)'].astype(str).str.split(';') 

    corum=dict()
    for x,r in corum_relevant.iterrows() : 
        corum.update({ r['ComplexName'] : r['components'] }) 

    return corum

def go_wf(infiles) :

    gobasicfilepath,gene2gohumanfilepath=infiles

    gobasic=pd.read_csv(gobasicfilepath,index_col=0).dropna(subset='id').infer_objects()
    gobasic=gobasic.assign(is_obsolete=gobasic.is_obsolete.astype(bool).fillna(False))
    gobasic=gobasic.query('not is_obsolete')
    gobasic['parent']=gobasic.is_a.apply(lambda s : s.split(' ')[0] if type(s) == str else '')

    gobasic_bp=gobasic.query('namespace == "biological_process"')
    gobasic_cc=gobasic.query('namespace == "cellular_component"')
    gobasic_mf=gobasic.query('namespace == "molecular_function"')
    bp=np.union1d(
        gobasic_bp.id.unique(),
        gobasic_bp.alt_id.dropna().unique()
    )
    cc=np.union1d(
        gobasic_cc.id.unique(),
        gobasic_cc.alt_id.dropna().unique()
    )
    mf=np.union1d(
        gobasic_mf.id.unique(),
        gobasic_mf.alt_id.dropna().unique()
    )

    g2gohuman=pd.read_csv(gene2gohumanfilepath,sep='\t')
    g2gohuman=g2gohuman[ ~g2gohuman.Qualifier.str.contains('NOT')] 
    g2gohuman=g2gohuman.query('Evidence in @valid_evidence')

    bpd=dict()
    ccd=dict()
    mfd=dict()

    from tqdm.auto import tqdm

    for x,r in tqdm(g2gohuman.iterrows(),total=g2gohuman.shape[0]): 
        if r.Category == "Process" : 
            bpd.update({ r.GO_ID: bpd.get(r.GO_ID,set()) | {str(r.GeneID),} })
            continue
        if r.Category == "Component"  : 
            ccd.update({ r.GO_ID: ccd.get(r.GO_ID,set()) | {str(r.GeneID),} })
            continue
        if r.Category == "Function"  : 
            mfd.update({ r.GO_ID: mfd.get(r.GO_ID,set()) | {str(r.GeneID),} })
            continue

    old_bpd=dict(bpd)
    y=0
    while True : 
        print('pass',y := y+1)
        for x,r in gobasic_bp.iterrows() : 
            if r.parent == '' : 
                continue
            bpd.update({ r.parent : bpd.get(r.parent,set()) | bpd.get(r.id,set()) })
        if all([ bpd[k] == old_bpd.get(k) for k in bpd.keys() ])  : 
            break 
        else : 
            old_bpd=dict(bpd)

    old_ccd=dict(ccd)
    y=0
    while True : 
        print('pass',y := y+1)
        for x,r in gobasic_cc.iterrows() : 
            if r.parent == '' : 
                continue
            ccd.update({ r.parent : ccd.get(r.parent,set()) | ccd.get(r.id,set()) })
        if all([ ccd[k] == old_ccd.get(k) for k in ccd.keys() ])  : 
            break 
        else : 
            old_ccd=dict(ccd)

    old_mfd=dict(mfd)
    y=0
    while True : 
        print('pass',y := y+1)
        for x,r in gobasic_mf.iterrows() : 
            if r.parent == '' : 
                continue
            mfd.update({ r.parent : mfd.get(r.parent,set()) | mfd.get(r.id,set()) })
        if all([ mfd[k] == old_mfd.get(k) for k in mfd.keys() ])  : 
            break 
        else : 
            old_mfd=dict(mfd)

    unity=bpd|ccd|mfd

    return unity

def kuenzi_wf(infiles) :
    consensus_fr=pd.read_excel(infiles[0]).dropna(subset='genes')
    kuenzi_consensus={ r.name : { kg._s2e[g] for g in r.genes.split('|') 
                        if g in kg._s2e } for x,r in consensus_fr.iterrows() }

    return kuenzi_consensus


def ont2dict(ontfilename,debug=False,lowlouvain=False,merge=False) : 
    g2ns=dict() ;
    nh=dict() ;
    ogenes=set()
    with open(ontfilename) as f: 
        for line in f : 
            ls=line.strip().split('\t')
            protochild=ls[1]
            if ls[-1] == 'gene' or ls[-2] == 'gene' :
                child=s2eid.get(protochild)
                if child is None : continue
                # add gene to list of genes of interest
                ogenes.add(child)
            else : 
                child=protochild

            # index system to gene
            g2ns.update({ child : g2ns.get(child,set()) | {ls[0],}})
            # and parent to all its children
            nh.update({ ls[0] : nh.get(ls[0],set()) | {child,}})

    nhkeys=set(nh.keys())
    changes_made=True
    x=0
    while changes_made : 
        x+=1
        #print(x) ;
        changes_made=False
        for k in nhkeys: 
            for cs in (nh[k] & nhkeys) :
                nh.update({ k : (nh[k] | nh[cs])-{cs,} })
                changes_made=True
                
    if debug : print(len(nh))
    if debug : print(sum([ len(nh[k]) for k in nh.keys() ])/len(nh))
    
    #print(len(nh))
    
    while len(nhkeys) > 0 : 
        nhk=nhkeys.pop()
        isconvertible=False
        try : int(nhk) ; isconvertible=True
        except ValueError : isconvertible=False
        if isconvertible : 
            values=nh.pop(nhk)
            nh.update({'Sys'+nhk : values })
            
    #print(len(nh))
    
    if lowlouvain : 
        for nhk in list(nh.keys()) : 
            if 'Louv' in nhk and len(nh[nhk]) < 20 : 
                nh.pop(nhk)
            elif len(nh[nhk])< 4 : 
                nh.pop(nhk) 

    return nh

def nest_wf(infiles) : pass
    ontfilename=infiles[0]
    nh=ont2dict(ontfilename,lowlouvain=False)
    return nh

def reactome_wf() : 
    df=pd.read_csv(infiles[0],sep='\t',
        names=['GeneID','identifier','url','sysname','evidence','organism'],
        )

    reactome=dict()
    for x,r in df.iterrows() :
        reactome.update({ r.sysname  : reactome.get(r.sysname,set())|{r.GeneID,} })

    return reactome
        
def webster_wf(infiles) :
    webster_loadings=pd.read_excel(infiles[0],sheet_name='Loadings',index_col=0)
    webster_names=pd.read_excel(infiles[0],sheet_name='Function Names')
    wnd=dict(zip(webster_names['Name'].values,webster_names['Manual Name'].values))

    wh=dict()
    for c in webster_loadings.columns : 
        if not c.startswith('V') : continue
        system_content=set()
        ser=webster_loadings[c]
        subser=ser[ ser.ne(0) ]
        system_content={ i.split('(')[-1].split(')')[0] for i in subser.index }
        wh.update({ wnd[c] : system_content })

    return wh

def wikipathways_wf(infiles) :
    import os
    opj=os
    from lxml import etree
    gpmls=infiles

    #path=gpmls[0]
    def geteids(path) : 
        # Load and parse the XML file
        tree = etree.parse(path)
        root = tree.getroot()

        # Define the namespace for the <Pathway> element
        namespaces = {'gpml': 'http://pathvisio.org/GPML/2013a'}

        # Extract the Name attribute from the first <Pathway> element
        pathway = root.xpath('//gpml:Pathway', namespaces=namespaces)
        if pathway:
            pathway_name = pathway[0].get('Name')
            #print(f'Pathway Name: {pathway_name}')
        else:
            pass
            #print("No <Pathway> element found")

        # Extract all ID attributes from <Xref> elements with Database="Entrez Gene" (no namespace for <Xref>)
        xrefs = root.findall('.//gpml:Xref',namespaces=namespaces)

        # Extract all the ID attributes
        ids = [xref.get('ID') for xref in xrefs if xref.get('Database') == 'Entrez Gene' ]
        return (pathway_name,set(ids))

    from tqdm.auto import tqdm
    wp=[ geteids(p) for p in tqdm(gpmls,total=len(gpmls)) ]

    return dict(wp)

def wk_wf(infiles) :
    kamber_raw=pd.read_excel(infiles[0],header=2)
    kambergenecols=['Gene']+[ c for c in kamber_raw.columns if c.startswith('Unnamed') ] 

    kh=dict()
    for x,r in kamber_raw.iterrows() : 
        if r.Synteny == 'Syntenic' : continue
        if type(r['Most-enriched GO term']) == str : 
            system_name='Kamber_'+r['Most-enriched GO term']
            if system_name in kh :
                system_name='Kamber_M'+str(r['Module #'])
        else:
            system_name='Kamber_M'+str(r['Module #'])
            
        system_content=set()
        for c in kambergenecols : 
            g=r.get(c)
            if g is None : continue
            e=kg._s2e.get(g) ;
            if e is None : continue
            system_content.add(e)
        kh.update({ system_name : system_content })

    return kh

WFD={
        'actpwy'   : actpwy_wf, 
        'corum'   : corum_wf, 
        'go'      : go_wf, 
        'kuenzi'  : kuenzi_wf,
        'nest'    : nest_wf,
        'reactome'    : reactome_wf,
        'webster'    : nest_wf,
        'wikipathways'    : nest_wf,
        'wk'    : nest_wf,
    }


def process(infilepaths,fkey,outfilepath) : 
    with open(outfilepath,'wb') as f : 
        pickle.dump(WFD[fkey](infilepaths),f)


jobs=[
        #activepathways
        (['cds_noncds_ActiveDriverPW_PCAWGintegrated.Pancan-no-skin-melanoma-lymph_noEnh__results_300817.txt',
          'actpwy',
          os.path.join(OUTPUTS,'actpwy.pickle')
          ]),
        #corum
        (['CORUM download 2022_09_12.xlsx'],
         'corum',
         os.path.join(OUTPUTS,'corum.pickle')
         ),
        #go
        (['gene2go_human','go_basic.obo'],
          'go',
         os.path.join(OUTPUTS,'go_unity.pickle')
         ),
        #kuenzi
        ([ 'NIHMS1571677-supplement-Supp_Table_3.xlsx',],
         'kuenzi',
         os.path.join(OUTPUTS,'kuenzi.pickle')
        ),
        #nest
        ([ 'IAS_clixo_hidef_Nov17.edges', ],
         'nest',
         os.path.join(OUTPUTS,'nest.pickle')
        ),
        #reactome
        ([ 'NCBI2Reactome_All_Levels.txt', ],
         'reactome',
         os.path.join(OUTPUTS,'nest.pickle')
        ),
        #webster
        ([ '1-s2.0-S2405471221004889-mmc4.xlsx'], # DON'T ADD THIS
         'webster',
         os.path.join(OUTPUTS,'webster.pickle'),
        ),

        #wikipathways
        ([ os.path.join('wikipathways_gpml',p) for p in os.listdir('wikipathways_gpml') ],
         'wikipathways',
         os.path.join(OUTPUTS,'webster.pickle'),
        ),

        #wk
        (['Supplementary_Data_2.xlsx',], # DON'T ADD THIS
         'wk',
         os.path.join(OUTPUTS,'wk.pickle'),
        ),
]


          




