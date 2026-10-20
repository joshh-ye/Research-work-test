"""Fetch GEO sample metadata for project GSMs and ENCODE metadata (ENCSR tracks + bulk cell-line ChIP cross-refs)."""
import json, time, urllib.request, urllib.parse, pandas as pd, os, re
R='/home/yejz1/experimental_research/'
C=R+'exposure_audit/cache/'
def get(url, hdr=None, tries=2):
    for i in range(tries):
        try:
            rq=urllib.request.Request(url, headers=hdr or {})
            return urllib.request.urlopen(rq, timeout=60).read().decode()
        except Exception as e:
            err=e; time.sleep(2)
    return 'ERROR '+str(err)
d=pd.read_csv(R+'borzoi_data/TF_184tracks.csv')
geo={}
for g in d.GSMID:
    if not g.startswith('GSM'): continue
    t=get(f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={g}&targ=self&form=text&view=brief')
    geo[g]=t; time.sleep(0.4)
json.dump(geo,open(C+'geo_gsm.json','w'))
enc={}
for g in d.GSMID:
    if g.startswith('ENC'):
        a=g.split('_')[0]
        enc[a]=get(f'https://www.encodeproject.org/experiments/{a}/?format=json',{'Accept':'application/json'}); time.sleep(0.5)
json.dump(enc,open(C+'encode_encsr.json','w'))
# bulk ENCODE ChIP experiments in the 10 cell lines (dbxrefs -> GEO GSM cross-refs)
terms=['HCT116','LoVo','Caco-2','GP5d','LS180','LS174T','SW480','RKO','DLD-1','HT-29','HT29']
bulk={}
for t in terms:
    u=('https://www.encodeproject.org/search/?type=Experiment&assay_title=TF+ChIP-seq&assay_title=Histone+ChIP-seq&assay_title=ChIP-seq'
       f'&biosample_ontology.term_name={urllib.parse.quote(t)}&limit=all&format=json&field=accession&field=dbxrefs&field=target.label&field=biosample_ontology.term_name&field=status')
    bulk[t]=get(u,{'Accept':'application/json'}); time.sleep(0.5)
json.dump(bulk,open(C+'encode_bulk_chip.json','w'))
print({k:len(v) for k,v in bulk.items()})
