"""Per-track training-exposure classification vs Borzoi / Enformer / AlphaGenome vocabularies. Inputs: cache/*.json from 01_fetch_metadata.py."""
import json, re, urllib.request, pandas as pd, numpy as np, collections
R='/home/yejz1/experimental_research/'; C=R+'exposure_audit/cache/'
d=pd.read_csv(R+'borzoi_data/TF_184tracks.csv')
bw=[int(re.search(r'/(\d+)\.bw',p).group(1)) for p in json.load(open(R+'borzoi_ckpt_full/bw_files.json'))]
geo=json.load(open(C+'geo_gsm.json')); enc=json.load(open(C+'encode_encsr.json')); bulk=json.load(open(C+'encode_bulk_chip.json'))
def gf(t,k): return [l.split(' = ',1)[1].strip() for l in t.split('\n') if l.startswith('!Sample_'+k)]
# --- ENCODE: GSM -> ENCSR cross-refs (HCT116, Caco-2 bulk ChIP)
gsm2encsr={}
for k,v in bulk.items():
    if v.startswith('ERROR'): continue
    for e in json.loads(v)['@graph']:
        for x in e.get('dbxrefs',[]):
            if x.startswith('GEO:GSM'): gsm2encsr[x[4:]]=e['accession']
# --- vocabularies
CELLS={'HCT-116':r'HCT[- ]?116','LoVo':r'lovo','Caco-2':r'caco-?2','GP5d':r'gp5d','LS180':r'ls ?180','LS174T':r'ls ?174','SW480':r'sw ?480','RKO':r'\bRKO\b','DLD-1':r'dld-?1\b','HT-29':r'ht-?29'}
def vocab(path,sep,idc,filec,desc):
    v=pd.read_csv(path,sep=sep); v['acc_gsm']=np.where(v[idc].astype(str).str.startswith('GSM'),v[idc],None)
    v['encsr']=v[filec].str.extract(r'(ENCSR[0-9A-Z]{6})')[0]
    v['assay']=v[desc].str.split(':').str[0]
    v['target']=np.where(v.assay=='CHIP',v[desc].str.split(':').str[1],None)
    v['desc']=v[desc]
    # Enformer GEO-LoVo ChIP rows have target '.', TF is in batch title: batchN_chromN_LoVo_<TF>_<host>_PassedQC
    m=v[desc].str.extract(r'_LoVo_([^_]+)_')[0]; v['target']=np.where(v.target=='.',m,v.target)
    return v
V={'Borzoi':vocab(R+'borzoi_data/targets_human.csv',',','identifier','file','description'),
   'Enformer':vocab(R+'enformer_data/targets_human.txt','\t','identifier','file','description')}
ALIAS={'P300':'EP300'}
norm=lambda s:ALIAS.get(str(s).upper(),str(s).upper())
def cellmask(v,c): return v.desc.str.contains(CELLS[c],case=False,regex=True)
# AlphaGenome ChIP-TF metadata saved from the 2026-08 run
ag=pd.read_csv(R+'results_baseline_alphagenome/alphagenome_tracks_test.csv')
AGCELL={'HCT-116':'HCT116','Caco-2':'Caco-2'}
# --- resolve project metadata
rows=[]
for idx,dc in enumerate(bw):
    r=d[d.DCid==dc].iloc[0]; acc=r.GSMID; gsm=acc if acc.startswith('GSM') else None; encsr=acc.split('_')[0] if acc.startswith('ENC') else None
    cell=r.Cell_line if isinstance(r.Cell_line,str) else None; src='csv'
    if cell=='HT29': cell='HT-29'  # harmonise CSV spelling with vocab regex key
    biosample=None; series=None; geo_cell_note=''
    if gsm:
        t=geo[gsm]; txt=' '.join(gf(t,'source_name_ch1')+gf(t,'characteristics_ch1')+gf(t,'title')); series=';'.join(gf(t,'series_id'))
        biosample=' | '.join(gf(t,'source_name_ch1'))
        hit=[c for c,p in CELLS.items() if re.search(p,txt,re.I)]
        if cell is None:
            if hit: cell=hit[0]; src='geo_text'
            elif re.search('tumor',txt,re.I): cell='primary_colon_tumor'; src='geo_text'
        else:
            canon={'HCT-116':'HCT-116','HT29':'HT-29'}.get(cell,cell)
            if hit and canon not in hit: geo_cell_note=f'GEO text suggests {hit} vs CSV {cell}'
        if gsm in gsm2encsr: encsr=gsm2encsr[gsm]
    if acc.startswith('ENC'):
        j=json.loads(enc[encsr]); biosample=j['biosample_ontology']['term_name']; series=';'.join(x for x in j['dbxrefs'] if x.startswith('GEO:')); cell='transverse_colon_tissue'; src='encode'
    rows.append(dict(track_idx=idx,DCid=dc,name=f'{r.Factor}_{cell or "NA"}_{dc}',cell_line=cell,cell_line_source=src,TF=r.Factor,GSM=gsm,ENCSR=encsr,
                     ENCSR_via='own' if acc.startswith('ENC') else ('ENCODE_dbxref_GSM' if gsm in gsm2encsr else ''),GEO_series_or_xref=series,biosample_text=biosample,note=geo_cell_note))
T=pd.DataFrame(rows)
# --- classification
for m,v in V.items():
    gs=set(v.acc_gsm.dropna()); es=set(v.encsr.dropna()); chip=v[v.assay=='CHIP']
    cls=[];det=[];cellassays=[]
    for _,t in T.iterrows():
        tf=norm(t.TF); same_t=chip[chip.target.map(norm)==tf]
        ident = (t.GSM in gs) or (t.ENCSR in es if isinstance(t.ENCSR,str) else False)
        cl=t.cell_line; ca=[]
        if cl in CELLS: ca=sorted(v[cellmask(v,cl)].assay.unique())
        elif cl=='transverse_colon_tissue': ca=sorted(v[v.desc.str.contains('transverse colon',case=False)].assay.unique())
        st_same_cell = len(same_t[cellmask(same_t,cl)])>0 if cl in CELLS else False
        if ident: c='identical_experiment'
        elif st_same_cell: c='same_cellline_same_target'
        elif ca: c='same_cellline_other_target'
        elif len(same_t): c='same_target_other_cellline'
        else: c='unexposed'
        cls.append(c); cellassays.append('|'.join(ca)); det.append(len(same_t))
    T[f'{m}_class']=cls; T[f'{m}_cellline_assays']=cellassays; T[f'{m}_n_same_target_ChIP_tracks']=det
# AlphaGenome (ChIP-TF only metadata available)
cls=[];note=[]
for _,t in T.iterrows():
    tf=norm(t.TF); same_t=ag[ag.transcription_factor.map(norm)==tf]
    cl=t.cell_line
    if cl in AGCELL:
        bs=ag[ag.biosample_name==AGCELL[cl]]
        if len(same_t[same_t.biosample_name==AGCELL[cl]]): c='same_cellline_same_target'
        elif len(bs): c='same_cellline_other_target'
        elif len(same_t): c='same_target_other_cellline'
        else: c='unknown'
    elif cl=='transverse_colon_tissue':
        c='unknown'  # same tissue+target ChIP-TF track may exist, but ENCSR identity cannot be checked (no accessions in saved metadata)
    elif len(same_t): c='same_target_other_cellline'
    else: c='unknown'   # no ChIP-TF track for this TF; other assay heads (DNase/CAGE/RNA...) metadata not saved -> cannot assert unexposed
    cls.append(c)
T['AlphaGenome_class']=cls
T['AlphaGenome_possible_identical_ENCODE']=T.ENCSR.notna()  # AG ChIP-TF is ENCODE-only; identity unverifiable (accessions not in saved metadata; AG Supp. Table 2 not obtained)
T['AlphaGenome_n_same_target_ChIPTF_tracks']=[len(ag[ag.transcription_factor.map(norm)==norm(x)]) for x in T.TF]
T.to_csv(R+'exposure_audit/track_exposure.tsv',sep='\t',index=False)
print(len(T))
for m in ['Borzoi','Enformer','AlphaGenome']: print(m,T[f'{m}_class'].value_counts().to_dict())
print(T.cell_line.value_counts().to_dict())
pd.set_option('display.width',250)
for m in ['Borzoi','Enformer','AlphaGenome']: print(pd.crosstab(T.cell_line,T[m+'_class']).to_string())
for m in ['Borzoi','Enformer']: print(m,'cell lines w/ NO vocab exposure any assay:',T[T[m+'_cellline_assays']==''].groupby('cell_line').size().to_dict()); print(T[T[m+'_cellline_assays']!=''].groupby('cell_line')[m+'_cellline_assays'].first().to_dict())
print(T[(T.Borzoi_class=='identical_experiment')&(T.cell_line=='HCT-116')][['GSM','ENCSR','TF']])
print(T[(T.Enformer_class=='identical_experiment')&(T.cell_line=='HCT-116')][['GSM','ENCSR','TF']])
print('ENCODE xref GSMs in project:',T[T.ENCSR_via=='ENCODE_dbxref_GSM'][['GSM','ENCSR','TF','cell_line']].to_string())
print(T[T.note!=''][['GSM','cell_line','note']])
