"""Reconstruct project tiles (borzoi_code/genome_tiler.py logic, CPU) and compute bp-fraction overlap with pretrained-model folds.
Tile: non-overlapping 524,288-bp windows from pos 0 per chrom (chr1,chr8=val; chr9,chr22=test), chroms from hg38.ml.fa.fai.
Target region of a tile = centre 131,072 bp (borzoi_code/dataset.py center_bin_size)."""
import gzip, collections, csv
R='/home/yejz1/experimental_research/'
F=R+'external_data/folds/'
SEQ=524288; TGT=131072
fai=[l.split('\t')[:2] for l in open(R+'borzoi_data/hg38/hg38.ml.fa.fai')]
tiles=[]
for ch,L in fai:
    L=int(L)
    if L<1_000_000 or any(x in ch for x in ("Un","random","alt","fix","M")): continue
    sp='val' if ch in('chr1','chr8') else 'test' if ch in('chr9','chr22') else 'train'
    p=0
    while p+SEQ<=L: tiles.append((ch,p,p+SEQ,sp)); p+=SEQ
cnt=collections.Counter(t[3] for t in tiles); print(cnt)
def merge(iv):
    iv.sort(); out=[]
    for s,e in iv:
        if out and s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out
def build(rows, ext=0):
    """rows: (chrom,start,end,label) -> {label:{chrom:[merged]}} with each interval extended by ext on both sides"""
    d=collections.defaultdict(lambda: collections.defaultdict(list))
    for ch,s,e,l in rows: d[l][ch].append((max(0,s-ext),e+ext))
    return {l:{c:merge(v) for c,v in dd.items()} for l,dd in d.items()}
def ov(m,ch,s,e):
    t=0
    for a,b in m.get(ch,[]):
        if b<=s: continue
        if a>=e: break
        t+=min(b,e)-max(a,s)
    return t
borz=[(l.split('\t')[0],int(l.split('\t')[1]),int(l.split('\t')[2]),l.split('\t')[3].strip()) for l in gzip.open(F+'borzoi_sequences_human.bed.gz','rt')]
# Borzoi README: test=fold3, valid=fold4, rest train (all 4 replicates)
bmap=lambda l:'test' if l=='fold3' else 'valid' if l=='fold4' else 'train'
borz=[(c,s,e,bmap(l)) for c,s,e,l in borz]
enf=[l.rstrip('\n').split('\t') for l in open(F+'enformer_genentech_human_intervals.tsv')][1:]
enf=[(c,int(s),int(e),l) for c,s,e,l in enf]
bas=[(c,int(s),int(e),l) for c,s,e,l in (l.split() for l in open(F+'basenji2_barnyard2_sequences_human.bed'))]
folds={
 'Borzoi_bed(196608 target region)':build(borz),
 'Borzoi_input(bed+-163840=524288)':build(borz,163840),
 'Enformer_input(196608)':build(enf),
 'Basenji2_bed(131072)':build(bas),
}
rows=[]
for sp in('val','test'):
    T=[t for t in tiles if t[3]==sp]
    for fn,fd in folds.items():
        for reg,(off,w) in {'tile_full524288':(0,SEQ),'tile_centre131072':((SEQ-TGT)//2,TGT)}.items():
            tot=0; acc=collections.Counter()
            for ch,s,e,_ in T:
                a,b=s+off,s+off+w; tot+=w
                for l,m in fd.items(): acc[l]+=ov(m,ch,a,b)
            r=dict(project_split=sp,n_tiles=len(T),fold_source=fn,tile_region=reg,total_bp=tot)
            for l in('train','valid','test'): r[f'frac_{l}']=round(acc[l]/tot,4)
            r['frac_uncovered']=round(1-sum(acc[l] for l in('train','valid','test'))/tot,4)
            rows.append(r)
with open(R+'exposure_audit/tile_fold_overlap.tsv','w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t'); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
