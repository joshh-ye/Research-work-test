#!/usr/bin/env python
"""E1 variant provenance: Chen 2024 (Nat Commun 15:3557) Supp Data 6 CCVs vs. S6_CCV AlphaGenome input.
CPU only. Run:  ./torch-venv/bin/python -I variant_provenance/build_provenance.py
Inputs (raw, untouched):
  external_data/chen2024/MOESM4_ESM.xlsx                         (Supplementary Data 1-22)
  external_data/chen2024/hg19ToHg38.over.chain.gz                (UCSC liftover chain)
  external_data/chen2024/gwascat_harmonised/GCST90129505.h.tsv.gz (GWAS Catalog harmonised summary stats, GRCh38)
  alphagenome_variant_effects/input/S6_CCV_ChrPosRefAlt_rsid.tsv
  borzoi_data/hg38/hg38.ml.fa(.fai)
No alleles are guessed: 'reported allele' evidence comes only from GWAS Catalog GCST90129505 (and, where present,
Chen supplementary tables S3-S5 lead variants and S15-S19 colocalised CCVs).
"""
import bisect, gzip, os, sys
import pandas as pd, numpy as np

ROOT = "/home/yejz1/experimental_research"
EXT = f"{ROOT}/external_data/chen2024"
OUT = f"{ROOT}/variant_provenance"
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


# ---------------- liftover (own minimal chain implementation) ----------------
def load_chain(path, chroms):
    blocks = {}  # tname -> list of (tstart, tend, qname, qstart, qstrand, qsize)
    cur = None
    with gzip.open(path, "rt") as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "chain":
                tname, qname, qsize, qstrand, qstart = p[2], p[7], int(p[8]), p[9], int(p[10])
                tpos = int(p[5])
                cur = tname in chroms
                if cur:
                    lst = blocks.setdefault(tname, [])
                    st = dict(t=tpos, q=qstart, qn=qname, qs=qsize, qstr=qstrand)
            elif cur:
                size = int(p[0])
                lst.append((st["t"], st["t"] + size, st["qn"], st["q"], st["qstr"], st["qs"]))
                if len(p) == 3:
                    st["t"] += size + int(p[1])
                    st["q"] += size + int(p[2])
    for k in blocks:
        blocks[k].sort()
    return {k: ([b[0] for b in v], v) for k, v in blocks.items()}


def lift(chain, chrom, pos1):
    """hg19 1-based -> (hg38 chrom, 1-based pos) or None"""
    c = "chr" + str(chrom)
    if c not in chain:
        return None
    starts, v = chain[c]
    p0 = pos1 - 1
    i = bisect.bisect_right(starts, p0) - 1
    if i < 0:
        return None
    ts, te, qn, qs, qstr, qsize = v[i]
    if not (ts <= p0 < te):
        return None
    q = qs + (p0 - ts)
    if qstr == "-":
        q = qsize - 1 - q
    return qn.replace("chr", ""), q + 1


# ---------------- reference FASTA ----------------
class Fasta:
    def __init__(self, fa):
        self.f = open(fa, "rb")
        self.idx = {}
        for l in open(fa + ".fai"):
            n, ln, off, bl, bw = l.split()[:5]
            self.idx[n] = (int(ln), int(off), int(bl), int(bw))

    def base(self, chrom, pos1):
        ln, off, bl, bw = self.idx["chr" + str(chrom)]
        p = pos1 - 1
        self.f.seek(off + (p // bl) * bw + p % bl)
        return self.f.read(1).decode().upper()


def ensembl_lookup(rsids, cache_dir, batch=200):
    """returns {rsid: (chr, pos, via)}; GRCh38, primary chromosomes 1-22 only. Caches raw responses."""
    import json, time, urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for bi in range(0, len(rsids), batch):
        ids = rsids[bi:bi + batch]
        fn_ = f"{cache_dir}/batch_{bi // batch:03d}.json"
        if not os.path.exists(fn_):
            req = urllib.request.Request("https://rest.ensembl.org/variation/homo_sapiens", data=json.dumps({"ids": ids}).encode(),
                                         headers={"Content-Type": "application/json", "Accept": "application/json"})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        open(fn_, "wb").write(r.read())
                    break
                except Exception as e:  # noqa
                    print("ensembl retry", bi, e)
                    time.sleep(5)
            time.sleep(1)
        d = json.load(open(fn_))
        syn = {}
        for k, v in d.items():
            for sname in v.get("synonyms", []):
                syn[sname] = k
        for r in ids:
            key, via = (r, "direct") if r in d else ((syn[r], "synonym") if r in syn else (None, None))
            if key is None:
                continue
            maps = [m for m in d[key].get("mappings", []) if m.get("assembly_name") == "GRCh38" and m["seq_region_name"] in {str(i) for i in range(1, 23)}]
            if maps:
                out[r] = (maps[0]["seq_region_name"], int(maps[0]["start"]), via if key == r or via == "synonym" else via)
            else:
                out[r] = (d[key]["mappings"][0]["seq_region_name"] if d[key].get("mappings") else None, None, via)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    log = []

    def L(s):
        print(s)
        log.append(s)

    # ---- source S6
    s6 = pd.read_excel(f"{EXT}/MOESM4_ESM.xlsx", sheet_name="Supplementary Data 6", header=1, usecols=range(5))
    s6 = s6.dropna(subset=["CCV"]).copy()
    s6.columns = ["region", "lead_snp", "rsid", "chr19", "pos19"]
    s6["chr19"] = s6["chr19"].astype(int).astype(str)
    s6["pos19"] = s6["pos19"].astype(int)
    s6["source_row_id"] = ["S6_r%d" % (i + 3) for i in range(len(s6))]  # excel row number (header on row 2, data from row 3)
    L(f"S6 rows={len(s6)} unique_rsid={s6.rsid.nunique()} unique_hg19_pos={s6[['chr19','pos19']].drop_duplicates().shape[0]} "
      f"signals(region,lead)={s6[['region','lead_snp']].drop_duplicates().shape[0]} regions={s6.region.nunique()}")

    # ---- liftover S6 hg19 -> hg38
    chain = load_chain(f"{EXT}/hg19ToHg38.over.chain.gz", {"chr" + c for c in s6.chr19.unique()})
    lf = [lift(chain, c, p) for c, p in zip(s6.chr19, s6.pos19)]
    s6["lift_chr38"] = [x[0] if x else None for x in lf]
    s6["lift_pos38"] = [x[1] if x else None for x in lf]
    L(f"S6 liftover unmapped={s6.lift_pos38.isna().sum()}  chrom_changed={(s6.lift_chr38.notna() & (s6.lift_chr38!=s6.chr19)).sum()}")

    # ---- S6_CCV input
    inp = pd.read_csv(f"{ROOT}/alphagenome_variant_effects/input/S6_CCV_ChrPosRefAlt_rsid.tsv", sep="\t")
    inp["input_key"] = inp["chr:pos:ref:alt"]
    sp = inp["input_key"].str.split(":", expand=True)
    inp["chr38"], inp["pos38"], inp["ref"], inp["alt"] = sp[0], sp[1].astype(int), sp[2], sp[3]
    inp["rsid"] = inp["rs"]
    L(f"input rows={len(inp)} unique_rsid={inp.rsid.nunique()} unique_pos={inp[['chr38','pos38']].drop_duplicates().shape[0]} "
      f"dup_key={inp.input_key.duplicated().sum()}")
    ppos = inp.groupby(["chr38", "pos38"]).rsid.nunique()
    L(f"positions with >1 rsid in input: {(ppos>1).sum()};  rsids with >1 position: {(inp.groupby('rsid').pos38.nunique()>1).sum()}")
    L("alts per position: " + str(inp.groupby(["chr38", "pos38"]).size().value_counts().sort_index().to_dict()))

    # reference check
    fa = Fasta(f"{ROOT}/borzoi_data/hg38/hg38.ml.fa")
    inp["ref_matches_fasta"] = [fa.base(c, p) == r for c, p, r in zip(inp.chr38, inp.pos38, inp.ref)]
    L(f"input REF != hg38 FASTA: {(~inp.ref_matches_fasta).sum()}")

    # ---- reconcile S6 vs input
    pos_inp = inp[["chr38", "pos38", "rsid"]].drop_duplicates()
    m = s6.merge(pos_inp, on="rsid", how="left", indicator=True)
    s6["in_input_by_rsid"] = s6.rsid.isin(inp.rsid)
    mm = s6.merge(pos_inp.rename(columns={"rsid": "inp_rsid"}), left_on=["lift_chr38", "lift_pos38"],
                  right_on=["chr38", "pos38"], how="left")
    L(f"S6 rsids in input={s6.in_input_by_rsid.sum()}, not in input={(~s6.in_input_by_rsid).sum()}: {s6[~s6.in_input_by_rsid].rsid.tolist()}")
    # position agreement for rsid matches
    sx = s6[s6.in_input_by_rsid].merge(pos_inp, on="rsid")
    agree = (sx.lift_chr38 == sx.chr38) & (sx.lift_pos38 == sx.pos38)
    L(f"rsid-matched S6 rows: liftover(hg19)->hg38 equals input pos: {agree.sum()}/{len(sx)}; mismatches: {(~agree).sum()}")
    for _, r in sx[~agree].iterrows():
        L(f"  liftover mismatch {r.rsid}: hg19 {r.chr19}:{r.pos19} -> lift {r.lift_chr38}:{r.lift_pos38} vs input {r.chr38}:{r.pos38}")
    s6["input_pos38"] = s6.rsid.map(pos_inp.drop_duplicates("rsid").set_index("rsid").pos38)
    s6["input_chr38"] = s6.rsid.map(pos_inp.drop_duplicates("rsid").set_index("rsid").chr38)
    # input rsids not in S6
    L(f"input rsids not in S6: {(~pos_inp.rsid.isin(s6.rsid)).sum()}")
    # position-based check of the 2 missing
    miss = s6[~s6.in_input_by_rsid]
    for _, r in miss.iterrows():
        hit = pos_inp[(pos_inp.chr38 == r.lift_chr38) & (pos_inp.pos38 == r.lift_pos38)]
        ref = fa.base(r.lift_chr38, int(r.lift_pos38)) if pd.notna(r.lift_pos38) else None
        L(f"  missing {r.rsid} hg19 {r.chr19}:{r.pos19} -> hg38 {r.lift_chr38}:{r.lift_pos38}; input rows at that pos: {len(hit)}; hg38 ref base={ref}")


    # ---- Ensembl REST rsID -> GRCh38 verification (cached raw JSON in external_data/chen2024/ensembl_rest)
    ens = ensembl_lookup(sorted(set(s6.rsid)), f"{EXT}/ensembl_rest")
    s6["ens_pos38"] = s6.rsid.map(lambda r: ens.get(r, (None, None, None))[1])
    s6["ens_chr38"] = s6.rsid.map(lambda r: ens.get(r, (None, None, None))[0])
    s6["ens_via"] = s6.rsid.map(lambda r: ens.get(r, (None, None, None))[2])
    ee = s6.merge(pos_inp, on="rsid", how="inner")
    ag = (ee.ens_chr38.astype(str) == ee.chr38.astype(str)) & (ee.ens_pos38 == ee.pos38)
    L(f"Ensembl(GRCh38) rsID resolved for {s6.ens_pos38.notna().sum()}/{len(s6)} S6 rsIDs (via merged-synonym: {(s6.ens_via=='synonym').sum()}); "
      f"for rsID-matched input rows: Ensembl pos == input pos {ag.sum()}/{len(ee)}; disagree {(~ag).sum()}")
    for _, r in ee[~ag].iterrows():
        L(f"  Ensembl disagrees {r.rsid}: input {r.chr38}:{r.pos38} vs Ensembl {r.ens_chr38}:{r.ens_pos38}")
    for _, r in s6[~s6.in_input_by_rsid].iterrows():
        L(f"  missing-from-input {r.rsid}: Ensembl -> {r.ens_chr38}:{r.ens_pos38} ({r.ens_via})")

    # ---- GWAS Catalog harmonised sumstats (GRCh38 pos; Variant_id = original build-37 id other/effect)
    need_rs = set(s6.rsid)
    need_pos19 = set(zip(s6.chr19, s6.pos19))
    rows = []
    for ch in pd.read_csv(f"{EXT}/gwascat_harmonised/GCST90129505.h.tsv.gz", sep="\t", chunksize=2_000_000, dtype=str):
        v = ch["Variant_id"].fillna("")
        p19 = v.str.extract(r"^([^:]+):(\d+)_")
        k = list(zip(p19[0], p19[1].astype("Int64").fillna(-1).astype(int)))
        sel = ch["rsid"].isin(need_rs) | ch["rs_id"].isin(need_rs) | pd.Series([x in need_pos19 for x in k], index=ch.index)
        sub = ch[sel].copy()
        sub["gc_chr19"] = p19[0][sel]
        sub["gc_pos19"] = p19[1][sel]
        rows.append(sub)
    gc = pd.concat(rows, ignore_index=True)
    gc = gc.rename(columns={"chromosome": "gc_chr38", "base_pair_location": "gc_pos38", "effect_allele": "gc_effect",
                            "other_allele": "gc_other", "beta": "gc_beta", "standard_error": "gc_se", "p_value": "gc_p",
                            "variant_id": "gc_variant_id_h", "Variant_id": "gc_variant_id_orig"})
    gc["gc_pos38"] = pd.to_numeric(gc.gc_pos38, errors="coerce")
    gc["gc_beta"] = pd.to_numeric(gc.gc_beta, errors="coerce")
    gc["gc_pos19"] = pd.to_numeric(gc.gc_pos19, errors="coerce")
    gc["gc_p"] = pd.to_numeric(gc.gc_p, errors="coerce")
    L(f"GWAS Catalog harmonised rows retained (rsid or hg19-pos in S6)={len(gc)}")
    gc["gc_rsid"] = gc.rsid.where(gc.rsid.notna() & (gc.rsid != "NA"), gc.rs_id)
    gc = gc.drop(columns=["rsid", "rs_id"])

    # S6 <-> GWAS Catalog link (by rsid, require hg19 position agreement with Chen S6)
    g_by_rs = gc[gc.gc_rsid.isin(need_rs)]
    s6g = s6.merge(g_by_rs, left_on="rsid", right_on="gc_rsid", how="left")
    s6g["hg19_pos_agree"] = (s6g.gc_pos19 == s6g.pos19) & (s6g.gc_chr19 == s6g.chr19)
    L(f"S6 rsids present in GWAS Catalog (any row)={s6g[s6g.gc_effect.notna()].rsid.nunique()}/{s6.rsid.nunique()}; "
      f"with hg19 position equal to Chen S6 = {s6g[s6g.hg19_pos_agree].rsid.nunique()}")
    # rsid-less / merged-rsid fallback: hg19 position match
    s6_nors = s6[~s6.rsid.isin(s6g[s6g.hg19_pos_agree].rsid)]
    L(f"S6 rows without rsid+hg19pos GWAS Catalog match={len(s6_nors)}")
    gpos = gc.dropna(subset=["gc_pos19"]).copy()
    gpos["k"] = gpos.gc_chr19 + ":" + gpos.gc_pos19.astype(int).astype(str)
    s6_nors = s6_nors.assign(k=s6_nors.chr19 + ":" + s6_nors.pos19.astype(str))
    fb = s6_nors.merge(gpos, on="k", how="inner")
    L(f"  of which matched by hg19 position only (rsid differs)={fb.rsid.nunique()}")

    # ---- Chen supplementary allele info (leads S3-S5; coloc CCVs S15-S19)
    chen_al = {}
    for sh, cols in [("Supplementary Data 3", ("Lead variant of independent risk signal", "Chr (hg19)", "Position", "Effective allele", "Reference allele")),
                     ("Supplementary Data 4", ("Lead variant of independent risk signala", "Chr (hg19)", "Position", "Effective allele", "Reference allele")),
                     ("Supplementary Data 5", ("Lead variant of independent risk signala", "Chr (hg19)", "Position", "Effective allele", "Reference allele")),
                     ("Supplementary Data 15", ("CCV", "Chromosome (hg19)", "Position", "Effect allele", "Reference allele")),
                     ("Supplementary Data 16", ("CCV", "Chromosome (hg19)", "Position", "Effect allele", "Reference allele")),
                     ("Supplementary Data 17", ("CCV", "Chromosome (hg19)", "Position", "Effect allele", "Reference allele")),
                     ("Supplementary Data 18", ("CCV", "Chromosome (hg19)", "Position", "Effect allele", "Reference allele")),
                     ("Supplementary Data 19", ("CCV", "Chromosome (hg19)", "Position", "Effect allele", "Reference allele"))]:
        t = pd.read_excel(f"{EXT}/MOESM4_ESM.xlsx", sheet_name=sh, header=1)
        t = t[list(cols)].dropna()
        t.columns = ["rsid", "c", "p", "eff", "oth"]
        t["sheet"] = sh.replace("Supplementary Data ", "S")
        chen_al.setdefault("all", []).append(t)
    chen = pd.concat(chen_al["all"], ignore_index=True)
    chen["eff"] = chen.eff.astype(str).str.upper()
    chen["oth"] = chen.oth.astype(str).str.upper()
    chen = chen[chen.rsid.isin(s6.rsid)]
    chen_u = chen.groupby("rsid").apply(lambda d: pd.Series({
        "chen_effect_allele": ";".join(sorted(set(d.eff))), "chen_other_allele": ";".join(sorted(set(d.oth))),
        "chen_allele_sheets": ",".join(sorted(set(d.sheet)))}), include_groups=False).reset_index()
    L(f"S6 CCVs with allele info in Chen supplement (S3-5 leads / S15-19 coloc CCVs)={chen_u.rsid.nunique()}")

    # ---- row-level matching of input rows to GWAS Catalog alleles
    # one GWAS row per (rsid, hg19 pos, allele pair); group per position
    gk = s6g[s6g.hg19_pos_agree].drop_duplicates(["rsid", "gc_effect", "gc_other"])
    gk = gk[["rsid", "source_row_id", "region", "lead_snp", "chr19", "pos19", "lift_chr38", "lift_pos38",
             "gc_chr38", "gc_pos38", "gc_effect", "gc_other", "gc_beta", "gc_se", "gc_p", "hm_code", "hm_coordinate_conversion",
             "gc_variant_id_orig"]]

    # ---- fallback: original GRCh37 file (alleles by exact hg19 position) for S6 CCVs absent from the harmonised file
    done = set(gk.rsid)
    todo = s6[~s6.rsid.isin(done)]
    keys = {(c, p): r for c, p, r in zip(todo.chr19, todo.pos19, todo.rsid)}
    fr = []
    for ch in pd.read_csv(f"{EXT}/gwascat_grch37/GCST90129505_buildGRCh37.tsv", sep="\t", chunksize=3_000_000, dtype=str):
        sel = pd.Series([(a, int(b)) in keys for a, b in zip(ch.chromosome, ch.base_pair_location)], index=ch.index)
        fr.append(ch[sel])
    fr = pd.concat(fr)
    fr["rsid"] = [keys[(a, int(b))] for a, b in zip(fr.chromosome, fr.base_pair_location)]
    L(f"fallback original GRCh37 file: S6 CCVs without harmonised match={len(todo)}; found by exact hg19 position={fr.rsid.nunique()}")
    fr = fr.merge(s6[["rsid", "source_row_id", "region", "lead_snp", "chr19", "pos19", "lift_chr38", "lift_pos38", "ens_chr38", "ens_pos38"]], on="rsid")
    fr = fr.merge(pos_inp, on="rsid", how="inner")  # needs to be in input
    # hg38 position evidence: liftover or Ensembl must equal the input position (else not matched)
    okp = ((fr.lift_chr38 == fr.chr38) & (fr.lift_pos38 == fr.pos38)) | ((fr.ens_chr38 == fr.chr38) & (fr.ens_pos38 == fr.pos38))
    L(f"  fallback rows in input={len(fr)}; hg38 position corroborated by liftover or Ensembl={okp.sum()}")
    fr = fr[okp]
    fb_df = pd.DataFrame({"rsid": fr.rsid, "source_row_id": fr.source_row_id, "region": fr.region, "lead_snp": fr.lead_snp, "chr19": fr.chr19,
                          "pos19": fr.pos19, "lift_chr38": fr.lift_chr38, "lift_pos38": fr.lift_pos38, "gc_chr38": fr.chr38, "gc_pos38": fr.pos38,
                          "gc_effect": fr.effect_allele, "gc_other": fr.other_allele, "gc_beta": pd.to_numeric(fr.beta), "gc_se": pd.to_numeric(fr.standard_error),
                          "gc_p": pd.to_numeric(fr.p_value), "hm_code": "orig_GRCh37_unharmonised", "hm_coordinate_conversion": "NA",
                          "gc_variant_id_orig": fr.Variant_id})
    gk = pd.concat([gk, fb_df], ignore_index=True)
    ex = inp.merge(s6[["rsid", "source_row_id", "region", "lead_snp", "chr19", "pos19", "lift_chr38", "lift_pos38"]], on="rsid", how="left")
    ex = ex.merge(gk[["rsid", "gc_chr38", "gc_pos38", "gc_effect", "gc_other", "gc_beta", "gc_se", "gc_p", "hm_code",
                      "hm_coordinate_conversion", "gc_variant_id_orig"]], on="rsid", how="left")
    ex = ex.merge(chen_u, on="rsid", how="left")

    def evaluate(r):
        # returns (status, is_reported, method, strand_flip, palindromic, alt_is_effect, risk_allele)
        if pd.isna(r.source_row_id):
            return ("not_in_S6", pd.NA, "", False, False, pd.NA, "")
        if pd.isna(r.gc_effect) or r.gc_effect == "NA":
            return ("no_allele_info", pd.NA, "", False, False, pd.NA, "")
        e, o = r.gc_effect.upper(), r.gc_other.upper()
        if len(e) != 1 or len(o) != 1 or e not in COMP or o not in COMP:
            return ("nonSNV_or_invalid_source_alleles", pd.NA, "", False, False, pd.NA, "")
        if int(r.gc_pos38) != int(r.pos38) or str(r.gc_chr38) != str(r.chr38):
            return ("hg38_position_disagrees", pd.NA, "", False, False, pd.NA, "")
        pal = {e, o} in ({"A", "T"}, {"C", "G"})
        pair = {r.ref, r.alt}
        direct = pair == {e, o}
        flipped = (not direct) and pair == {COMP[e], COMP[o]}
        if direct or flipped:
            ee, oo = (COMP[e], COMP[o]) if flipped else (e, o)
            rb = (ee if r.gc_beta > 0 else oo) if pd.notna(r.gc_beta) else ""
            return ("matched", True, ("gwascat_rsid_pos38_allelepair" if r.hm_code != "orig_GRCh37_unharmonised" else "gwascat_grch37_hg19pos_allelepair_hg38pos_via_liftover_or_ensembl") + ("_complemented" if flipped else ""), bool(flipped), pal,
                    r.alt == ee, rb)
        # source allele pair known at this position but this ALT is not part of it
        if r.ref in (e, o, COMP[e], COMP[o]):
            return ("alt_not_in_reported_pair", False, "gwascat_rsid_pos38_allelepair", False, pal, pd.NA, "")
        return ("ref_not_in_reported_pair", pd.NA, "", False, pal, pd.NA, "")

    res = ex.apply(evaluate, axis=1, result_type="expand")
    res.columns = ["match_status", "is_reported_allele", "match_method", "strand_flip", "palindromic", "alt_is_gwas_effect_allele", "gwas_risk_allele_beta_sign"]
    ex = pd.concat([ex, res], axis=1)
    ex["is_reported_allele"] = ex["is_reported_allele"].astype(object)

    L("match_status counts: " + str(ex.match_status.value_counts().to_dict()))
    L("palindromic among matched rows: %d ; strand-flipped matched rows: %d" % (ex[ex.match_status == "matched"].palindromic.sum(),
                                                                                ex[ex.match_status == "matched"].strand_flip.sum()))
    mt = ex[ex.match_status == "matched"]
    L(f"matched rows={len(mt)} unique rsid={mt.rsid.nunique()} unique positions={mt[['chr38','pos38']].drop_duplicates().shape[0]}")
    L("hm_code among matched rsids: " + str(mt.drop_duplicates('rsid').hm_code.value_counts().to_dict()))
    # consistency with Chen supplement alleles where available
    c = mt.drop_duplicates("rsid").dropna(subset=["chen_effect_allele"])
    def chen_ok(r):
        ce, co = set(r.chen_effect_allele.split(";")), set(r.chen_other_allele.split(";"))
        pair = {r.ref, r.alt}
        return (ce | co) <= pair or (ce | co) <= {COMP[x] for x in pair}
    ok = c.apply(chen_ok, axis=1)
    L(f"Chen-supplement alleles available for {len(c)} matched rsids; allele set consistent with tested pair: {ok.sum()}; inconsistent: {(~ok).sum()}")
    ex["chen_allele_consistent"] = pd.NA
    ex.loc[c.index, "chen_allele_consistent"] = ok.values
    mt = ex[ex.match_status == "matched"]
    # direction consistency: Chen effect allele vs GWAS Catalog risk allele (beta sign) where both known
    # (Chen Beta is for Chen's effective allele; GWAS Catalog beta for effect_allele) -- reported only as counts

    # orientation validation: Chen S15 (trans-ancestry colocalised CCVs) effect allele + GWAS_Beta vs GWAS Catalog effect allele + beta
    t15 = pd.read_excel(f"{EXT}/MOESM4_ESM.xlsx", sheet_name="Supplementary Data 15", header=1)[["CCV", "Effect allele", "GWAS_Beta"]].dropna()
    t15 = t15.drop_duplicates(["CCV", "Effect allele", "GWAS_Beta"])
    t15["Effect allele"] = t15["Effect allele"].astype(str).str.upper()
    xv = t15.merge(mt.drop_duplicates("rsid"), left_on="CCV", right_on="rsid")
    sgn = np.sign(xv.GWAS_Beta) * np.where(xv["Effect allele"] == xv.gc_effect, 1, -1)
    agr = sgn == np.sign(xv.gc_beta)
    L(f"orientation check vs Chen S15 (effect allele, GWAS_Beta): n={len(xv)} sign-concordant={agr.sum()}; palindromic n={int(xv.palindromic.sum())} concordant={int(agr[xv.palindromic.values].sum())}")
    # positions per status
    pos_stat = ex.groupby(["chr38", "pos38"]).match_status.apply(lambda s: "matched" if (s == "matched").any() else s.iloc[0])
    L("position-level status (>=1 matched ALT => matched): " + str(pos_stat.value_counts().to_dict()))
    # S6 rows not reaching 'matched' (incl. 2 absent from input)
    s6_m = s6.copy()
    s6_m["matched_in_input"] = s6_m.rsid.isin(mt.rsid)
    L(f"S6 CCVs (5741) with >=1 allele-matched input row: {s6_m.matched_in_input.sum()}; without: {(~s6_m.matched_in_input).sum()}")
    L("positions with both alleles of an allele pair tested (ALT=effect and ALT=other present): n/a (input has REF fixed to hg38; the reported pair is {REF,ALT_reported})")

    # ---- write outputs
    base_cols = ["chr38", "pos38", "ref", "alt", "rsid", "source_row_id", "chr19", "pos19", "region", "lead_snp"]
    mt_out = mt[base_cols + ["gc_effect", "gc_other", "gc_beta", "gc_se", "gc_p", "alt_is_gwas_effect_allele", "gwas_risk_allele_beta_sign",
                             "chen_effect_allele", "chen_other_allele", "chen_allele_sheets", "chen_allele_consistent",
                             "match_method", "strand_flip", "palindromic", "hm_code", "input_key"]].copy()
    mt_out.columns = ["hg38_chr", "hg38_pos_1based", "ref", "alt", "rsid", "source_row_id_ChenS6_excel_row", "source_hg19_chr", "source_hg19_pos",
                      "chen_region", "chen_lead_snp", "gwascat_effect_allele", "gwascat_other_allele", "gwascat_beta", "gwascat_se", "gwascat_p",
                      "alt_is_gwascat_effect_allele", "gwascat_risk_allele_by_beta_sign", "chen_effect_allele_suppl", "chen_other_allele_suppl",
                      "chen_allele_sheets", "chen_allele_consistent", "match_method", "strand_flip", "palindromic", "gwascat_hm_code",
                      "S6_CCV_input_key"]
    mt_out["PIP"] = "not_provided_by_source(CCV=cond.P within 2 orders of magnitude of top; Chen2024 Methods)"
    mt_out.to_csv(f"{OUT}/allele_matched_variants.tsv", sep="\t", index=False)

    # unmatched/excluded: input rows not matched + the S6 CCVs absent from input
    um = ex[ex.match_status != "matched"][["input_key", "rsid", "chr38", "pos38", "ref", "alt", "match_status", "source_row_id", "gc_effect", "gc_other"]].copy()
    um["reason"] = um.match_status.map({
        "alt_not_in_reported_pair": "input ALT is not the other allele of the GWAS-reported allele pair at this position (exploratory-only ALT)",
        "no_allele_info": "rsID/position not found in GWAS Catalog GCST90129505 harmonised file (no allele info; not guessed)",
        "ref_not_in_reported_pair": "hg38 REF not in reported pair (even after complement)",
        "hg38_position_disagrees": "GWAS Catalog hg38 position differs from input position",
        "nonSNV_or_invalid_source_alleles": "source alleles not single bases",
        "not_in_S6": "rsID not in Chen S6"})
    um["unit"] = "input_row"
    ab = s6[~s6.in_input_by_rsid].copy()
    ab2 = pd.DataFrame({"input_key": "", "rsid": ab.rsid, "chr38": ab.lift_chr38, "pos38": ab.lift_pos38, "ref": "", "alt": "",
                        "match_status": "S6_CCV_absent_from_input", "source_row_id": ab.source_row_id, "gc_effect": "", "gc_other": "",
                        "reason": "Chen S6 CCV not present in S6_CCV input (pos = liftover of hg19)", "unit": "S6_CCV"})
    pd.concat([um, ab2], ignore_index=True).to_csv(f"{OUT}/unmatched_or_excluded.tsv", sep="\t", index=False)

    expo = ex[["input_key", "chr38", "pos38", "ref", "alt", "rsid", "match_status", "is_reported_allele", "alt_is_gwas_effect_allele",
               "strand_flip", "palindromic", "gc_effect", "gc_other", "gc_beta", "source_row_id", "region", "lead_snp"]].copy()
    expo.columns = ["chr:pos:ref:alt", "hg38_chr", "hg38_pos_1based", "ref", "alt", "rsid", "match_status", "is_reported_allele",
                    "alt_is_gwascat_effect_allele", "strand_flip", "palindromic", "gwascat_effect_allele", "gwascat_other_allele",
                    "gwascat_beta", "source_row_id_ChenS6_excel_row", "chen_region", "chen_lead_snp"]
    expo.to_csv(f"{OUT}/exploratory_all_rows.tsv", sep="\t", index=False, na_rep="NA")
    open(f"{OUT}/build_provenance.log", "w").write("\n".join(log) + "\n")
    # source-level functional columns present in Chen supplement (for report)
    xl = pd.ExcelFile(f"{EXT}/MOESM4_ESM.xlsx")
    L("sheets: " + ", ".join(xl.sheet_names))


if __name__ == "__main__":
    main()
