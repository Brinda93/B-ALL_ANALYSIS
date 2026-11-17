#!/usr/bin/env python3
"""Fig 1d/1e of Xu et al. (Nature 2022) for B-ALL.
 eig    : per sample, A/B compartments (E1 = top eigenvector of the Pearson matrix of balanced O/E, 40 kb, cis per chromosome, replicates summed, sign set so E1>0 = gene-dense) -> <out>/eig/<sample>.tsv
 switch : switches vs the control consensus, recurrent (>= --min_n patients) A->B / B->A bins, genes with TSS in them, expression boxplots
          (control vs patients; ALL, each genotype, ancestry, sex) + Fig 1d-style per-sample switch fractions.
Usage:
  python3 compartments_fig1e.py eig --sheets loops_cooltools/loops_patients.tsv,loops_cooltools/loops_controls_pooled.tsv --groups loops_cooltools/groups_7_mapped.tsv --gtf gencode.v44.annotation.gtf.gz --out compartments --threads 8
  python3 compartments_fig1e.py switch --out compartments --groups loops_cooltools/groups_7_mapped.tsv --corr_dir rna_batchcorr --gtf gencode.v44.annotation.gtf.gz --outdir compartments/fig1e"""
import argparse, os, re, sys, gzip, glob, shutil
import numpy as np, pandas as pd
ALIAS = {**{f"ND{i}": f"NDALL{i}" for i in range(2, 9)}, "AUS51": "AUSALL51", "AUS84": "AUSALL84", "AUS87": "ALL87", "AUS95": "ALL95", "AUS174": "PALNTB", "ALL19": "AUS19", "ALL53": "AUS53", "ALL57": "AUS57", "ALL59": "AUS59", "PVCRK": "PAVRCK", "PAVYCL": "PAVCYL", "ALL4364": "AALL4364", "W03": "W0", "ALL2": "AUS2", "ALL11": "AUS11", "ALL25": "AUS25", "ALL82": "AUS82", "SJ9": "SJBALL9", "YDP1ALL58HIC": "AUS58", "JRP8AUS174HICS8": "PALNTB"}
def key(s):
    s = re.split(r"[ (]", str(s).strip())[0]; s = re.sub(r"_MERGED$|_[A-Z]$", "", s.upper()); s = s.replace("W13-7", "W13"); s = re.sub(r"[^A-Z0-9]", "", s); s = re.sub(r"(?<=[A-Z])0+(?=\d)", "", s)
    return ALIAS.get(s, s)
P = argparse.ArgumentParser(); sp = P.add_subparsers(dest="cmd", required=True)
e = sp.add_parser("eig"); e.add_argument("--sheets", required=True); e.add_argument("--groups", required=True); e.add_argument("--gtf", required=True); e.add_argument("--out", default="compartments")
e.add_argument("--res", type=int, default=40000); e.add_argument("--threads", type=int, default=4); e.add_argument("--tmp", default=None); e.add_argument("--only", default="", help="comma list of sheet sample names (debug)"); e.add_argument("--nchrom", type=int, default=22)
w = sp.add_parser("switch"); w.add_argument("--out", default="compartments"); w.add_argument("--groups", required=True); w.add_argument("--corr_dir", required=True); w.add_argument("--gtf", required=True); w.add_argument("--outdir", default="compartments/fig1e")
w.add_argument("--ctrl_agree", type=float, default=0.8, help="fraction of controls that must agree on A (or B) for a bin to be a reference"); w.add_argument("--min_n", type=int, default=2, help="recurrent = switch in >= this many patients (paper: 2)")
w.add_argument("--min_abs", type=float, default=0.0, help="optional |E1| floor for a patient call (0 = sign only)"); w.add_argument("--exclude", default=""); w.add_argument("--biotype", default="protein_coding"); w.add_argument("--min_genes", type=int, default=10); w.add_argument("--shared_set", choices=["union", "all"], default="union", help="sex/ancestry panels: genes in recurrent switch bins of either class (union) or of all B-ALL"); w.add_argument("--res", type=int, default=40000)
a = P.parse_args()
def read_groups(path):
    g = pd.read_csv(path, sep="\t", header=None, comment="#", dtype=str).iloc[:, :2]; g.columns = ["sample", "group"]
    g = g[g["sample"].str.lower() != "sample"]; g["control"] = (g.group.str.lower() == "control") | g["sample"].str.upper().str.startswith("CTRL"); return g
def tss_table(gtf, biotype):
    rows = []; op = gzip.open if gtf.endswith(".gz") else open
    with op(gtf, "rt") as f:
        for l in f:
            if l.startswith("#"): continue
            t = l.rstrip("\n").split("\t")
            if len(t) < 9 or t[2] != "gene": continue
            bt = re.search(r'gene_(?:type|biotype) "([^"]+)"', t[8])
            if biotype != "all" and (not bt or bt.group(1) != biotype): continue
            rows.append((t[0], int(t[3]) if t[6] == "+" else int(t[4]), re.search(r'gene_id "([^"]+)"', t[8]).group(1).split(".")[0]))
    return pd.DataFrame(rows, columns=["chrom", "tss", "gene"])
# ---------------------------------------------------------------- eig
def eig_one(args):
    name, paths, outdir, tmp, res, nchrom, track_tss = args
    import cooler
    fo = f"{outdir}/eig/{name}.tsv"
    if os.path.exists(fo): return name, "exists"
    wd = f"{tmp}/{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}"; os.makedirs(wd, exist_ok=True)
    try:
        uris = []
        for i, p in enumerate(paths):
            if ".mcool" in p:
                base = p.split("::")[0]; avail = cooler.fileops.list_coolers(base); want = f"/resolutions/{res}"
                if want in avail: uris.append(f"{base}::{want}"); continue
                fine = [int(x.split("/")[-1]) for x in avail if "/resolutions/" in x]; fine = [r for r in fine if r < res and res % r == 0]
                if not fine: raise RuntimeError(f"no usable resolution in {p}")
                src = max(fine); out = f"{wd}/r{i}.cool"; cooler.coarsen_cooler(f"{base}::/resolutions/{src}", out, factor=res // src, chunksize=10_000_000); uris.append(out)
            else:
                bs = cooler.Cooler(p).binsize
                if bs == res: uris.append(p)
                else: out = f"{wd}/r{i}.cool"; cooler.coarsen_cooler(p, out, factor=res // bs, chunksize=10_000_000); uris.append(out)
        merged = f"{wd}/merged.cool"; cooler.merge_coolers(merged, uris, mergebuf=10_000_000)
        clr = cooler.Cooler(merged); cooler.balance_cooler(clr, cis_only=True, ignore_diags=2, mad_max=5, min_nnz=10, min_count=0, store=True, store_name="weight")
        clr = cooler.Cooler(merged); chroms = [c for c in clr.chromnames if c in {f"chr{i}" for i in range(1, nchrom + 1)}]
        bins = clr.bins()[:]; bins["gd"] = 0.0
        for c, d in track_tss.groupby("chrom"):
            idx = bins.index[bins.chrom == c]; bins.loc[idx, "gd"] = np.bincount((d.tss.values // res).astype(int), minlength=len(idx) + 1)[:len(idx)]
        out_rows = []
        for c in chroms:
            m = np.asarray(clr.matrix(balance="weight").fetch(c), dtype=float); n = m.shape[0]
            for k in range(-1, 2): 
                if abs(k) < 2: m[np.eye(n, k=k, dtype=bool)] = np.nan   # ignore_diags = 2
            ii = np.abs(np.subtract.outer(np.arange(n), np.arange(n))); exp = np.full(n, np.nan)
            for dd in range(2, n):
                v = np.diagonal(m, dd); v = v[np.isfinite(v)]
                if len(v): exp[dd] = v.mean()
            with np.errstate(invalid="ignore", divide="ignore"): oe = m / exp[ii]
            del ii
            wb = bins.loc[bins.chrom == c, "weight"].values; valid = np.isfinite(wb) & (np.isfinite(oe).sum(axis=1) > 0.3 * n)
            sub = oe[np.ix_(valid, valid)]; sub[~np.isfinite(sub)] = 1.0; cm = np.corrcoef(sub); cm[~np.isfinite(cm)] = 0.0
            vals, vecs = np.linalg.eigh(cm); v1 = vecs[:, -1]; gd = bins.loc[bins.chrom == c, "gd"].values[valid]
            if gd.std() > 0 and np.corrcoef(v1, gd)[0, 1] < 0: v1 = -v1
            bb = bins[bins.chrom == c][valid]; out_rows.append(pd.DataFrame({"chrom": c, "start": bb.start.values, "end": bb.end.values, "E1": v1}))
        E = pd.concat(out_rows, ignore_index=True)
        E = E[["chrom", "start", "end", "E1"]].dropna(); E.to_csv(fo, sep="\t", index=False)
        return name, f"ok {len(E)} bins"
    except Exception as ex: return name, f"FAILED {type(ex).__name__}: {ex}"
    finally: shutil.rmtree(wd, ignore_errors=True)
if a.cmd == "eig":
    os.makedirs(f"{a.out}/eig", exist_ok=True); tmp = a.tmp or f"{a.out}/tmp"; os.makedirs(tmp, exist_ok=True)
    sheet = {}
    for sh in a.sheets.split(","):
        for l in open(sh):
            t = l.rstrip("\n").split("\t")
            if len(t) >= 2 and t[0] and not t[0].startswith("#"): sheet[t[0]] = [x for x in t[1].split(",") if x]
    G = read_groups(a.groups); bykey = {}
    for n in sheet: bykey.setdefault(key(n), n)
    todo = {}; miss = []
    for s in G["sample"]:
        n = s if s in sheet else bykey.get(key(s))
        (todo.__setitem__(n, sheet[n]) if n else miss.append(s))
    if a.only: todo = {k: v for k, v in todo.items() if k in a.only.split(",")}
    print(f"{len(G)} samples in groups file ({int(G.control.sum())} controls); matched to sheet: {len(todo)}; NOT matched: {miss}")
    track = tss_table(a.gtf, "protein_coding"); jobs = [(n, p, a.out, tmp, a.res, a.nchrom, track) for n, p in todo.items()]
    if a.threads > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.threads) as pool:
            for n, m in pool.imap_unordered(eig_one, jobs): print(n, m, flush=True)
    else:
        for j in jobs: print(*eig_one(j), flush=True)
    sys.exit(0)
# ---------------------------------------------------------------- switch
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs(a.outdir, exist_ok=True)
files = {os.path.basename(f)[:-4]: f for f in glob.glob(f"{a.out}/eig/*.tsv")}; fk = {}
for n in files: fk.setdefault(key(n), n)
G = read_groups(a.groups); G["file"] = [s if s in files else fk.get(key(s)) for s in G["sample"]]; G["k"] = G["sample"].map(key)
print(f"{len(G)} samples in groups; with eigenvectors: {int(G.file.notna().sum())}; missing: {list(G[G.file.isna()]['sample'])}"); G = G[G.file.notna()]
ex = {key(x) for x in a.exclude.split(",") if x.strip()}; G = G[~G.k.isin(ex)]
ctrl = G[G.control]; pat = G[~G.control]; print(f"{len(ctrl)} controls, {len(pat)} patients used")
if len(ctrl) < 3: sys.exit("need >= 3 controls with eigenvectors")
def load(n): d = pd.read_csv(files[n], sep="\t"); return d.set_index(["chrom", "start"]).E1
M = pd.concat({r.sample: load(r.file) for r in pd.concat([ctrl, pat]).itertuples()}, axis=1); M = M[~M.index.get_level_values(0).isin(["chrX", "chrY", "chrM"])]
Cm = M[list(ctrl["sample"])]; nA = (Cm > 0).sum(axis=1); nv = Cm.notna().sum(axis=1); fA = nA / nv.where(nv > 0)
ref = pd.Series(np.nan, index=M.index, dtype=object); ref[(fA >= a.ctrl_agree) & (nv >= 3)] = "A"; ref[(fA <= 1 - a.ctrl_agree) & (nv >= 3)] = "B"
print(f"reference bins: A {int((ref == 'A').sum())}, B {int((ref == 'B').sum())}, ambiguous/removed {int(ref.isna().sum())} of {len(ref)}")
Pm = M[list(pat["sample"])]
isA2B = ((ref == "A").values[:, None]) & (Pm < -a.min_abs).values; isB2A = ((ref == "B").values[:, None]) & (Pm > a.min_abs).values
A2B = pd.DataFrame(isA2B, index=M.index, columns=Pm.columns); B2A = pd.DataFrame(isB2A, index=M.index, columns=Pm.columns)
# Fig 1d
fr = []
for s in Pm.columns:
    v = Pm[s]; ok = ref.notna() & v.notna(); fr.append((s, int(((ref == "A") & (v > 0) & ok).sum()), int(((ref == "B") & (v < 0) & ok).sum()), int(A2B[s].sum()), int(B2A[s].sum())))
F = pd.DataFrame(fr, columns=["sample", "stable_A", "stable_B", "A_to_B", "B_to_A"]); F["pct_switch"] = 100 * (F.A_to_B + F.B_to_A) / F[["stable_A", "stable_B", "A_to_B", "B_to_A"]].sum(axis=1); F.to_csv(f"{a.outdir}/switch_counts_per_patient.tsv", sep="\t", index=False)
print(f"mean switch: A->B {100 * (F.A_to_B / F.iloc[:, 1:5].sum(axis=1)).mean():.1f}%, B->A {100 * (F.B_to_A / F.iloc[:, 1:5].sum(axis=1)).mean():.1f}% (AML paper: 3.2% / 2.9%)")
Fo = F.sort_values("pct_switch"); fig, ax = plt.subplots(figsize=(13, 4)); bot = np.zeros(len(Fo))
for c, col in (("stable_A", "#3182bd"), ("stable_B", "#fd8d3c"), ("A_to_B", "#31a354"), ("B_to_A", "#de2d26")): ax.bar(range(len(Fo)), Fo[c], bottom=bot, color=col, label=c.replace("_", " ")); bot += Fo[c].values
ax.set_xticks(range(len(Fo))); ax.set_xticklabels(Fo["sample"], rotation=90, fontsize=5); ax.set_ylabel("40-kb bins"); ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False); plt.tight_layout(); plt.savefig(f"{a.outdir}/fig1d_switch_per_patient.png", dpi=200); plt.close()
# genes
Tg = tss_table(a.gtf, a.biotype); Tg = Tg[~Tg.chrom.isin(["chrX", "chrY", "chrM"])]; Tg["bin"] = (Tg.tss // a.res) * a.res
U = pd.read_csv(f"{a.corr_dir}/log2tpm_patients_uncorrected.tsv.gz", sep="\t", index_col=0); Cx = pd.read_csv(f"{a.corr_dir}/log2tpm_controls_uncorrected.tsv.gz", sep="\t", index_col=0)
S = pd.read_csv(f"{a.corr_dir}/patient_annotation_used.tsv", sep="\t", index_col=0); S["k"] = S["sample"].map(key)
Tg = Tg[Tg.gene.isin(U.index)]; tpm = lambda m: 2 ** m - 1
hk = set(pat.k); S = S[S.k.isin(hk)]; print(f"RNA patients that also have Hi-C eigenvectors: {len(S)}")
cls = {"ALL (all B-ALL)": pat.k.tolist()}
for col, pref in (("genotype", ""), ("Ancestry", ""), ("sex", "")):
    for v, d in S.groupby(col):
        if str(v) in ("nan", "NA"): continue
        cls[str(v)] = d.k.tolist()
fam = {"genotype": ["ALL (all B-ALL)"] + [c for c in cls if c.startswith("G") and c[1:2].isdigit()], "ancestry": [c for c in cls if c in ("AMR", "EUR")], "sex": [c for c in cls if c in ("F", "M")]}
res = []; glist = []; gtab = []; boxjson = []
def bhf(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.empty(n); r[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.clip(r, 0, 1)
def TLAB(trait, nm):
    return {"F": "Female", "M": "Male"}.get(nm, "All B-ALL" if nm.startswith("ALL") else nm) + " patients"
from matplotlib.patches import Patch
from matplotlib.cbook import boxplot_stats
TCOL = {"sex": "Sex", "ancestry": "Ancestry", "genotype": "Genotype"}
CTRL_COL = "#fc8d59"; COLS = {"F": "#c51b7d", "M": "#4d9221", "AMR": "#7570b3", "EUR": "#1b9e77"}
stars = lambda p: "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
def class_cols(nm):
    ks = set(cls[nm]); return list(dict.fromkeys([c for c in Pm.columns if key(c) in ks or c in set(pat[pat.k.isin(ks)]["sample"])]))
def class_genes(nm, sw):
    cc = class_cols(nm)
    if len(cc) < a.min_n: return [], set()
    rec = sw[cc].sum(axis=1) >= a.min_n; bins = set(map(tuple, rec[rec].index.tolist()))
    g = Tg[[(c, b) in bins for c, b in zip(Tg.chrom, Tg.bin)]].gene.unique(); return [x for x in g if x in Cx.index], bins
def make_panel(ax, trait, nms, lab, sw):
    """one panel: Control box + one box per class in nms (sex / ancestry: shared gene set so the control box is common), P vs control above each pair"""
    if len(nms) == 1: g, bins = class_genes(nms[0], sw)
    else:
        parts = [class_genes(n, sw) for n in (nms if a.shared_set == "union" else ["ALL (all B-ALL)"])]
        g = list(dict.fromkeys(x for pg_, _ in parts for x in pg_)); bins = set().union(*[b for _, b in parts])
    for x in g: glist.append((trait, "+".join(nms), lab, x))
    rnas = {n: S[S.k.isin(set(cls[n]))] for n in nms}
    if len(g) < a.min_genes or any(len(r_) < 2 for r_ in rnas.values()): ax.text(.5, .5, f"{'/'.join(nms)} {lab}\n{len(bins)} bins, {len(g)} genes", ha="center", transform=ax.transAxes, fontsize=7); ax.axis("off"); return
    TC = tpm(Cx.loc[g]); dT = [TC.mean(axis=1)]; labs = [f"Control\n(n={Cx.shape[1]})"]; cols = [CTRL_COL]; pvals = []
    for nm in nms:
        TP = tpm(U.loc[g, rnas[nm].index]); m = TP.mean(axis=1); dT.append(m); labs.append(f"{TLAB(trait, nm)}\n(n={len(rnas[nm])})"); cols.append(COLS.get(nm, "#7570b3"))
        try: pg = stats.mannwhitneyu(TP.values, TC.values, axis=1, alternative="two-sided").pvalue      # per gene, across samples
        except Exception: pg = np.array([stats.mannwhitneyu(TP.values[i_], TC.values[i_], alternative="two-sided").pvalue if np.ptp(np.r_[TP.values[i_], TC.values[i_]]) > 0 else 1.0 for i_ in range(len(g))])
        pg = np.nan_to_num(np.asarray(pg, float), nan=1.0); qg = bhf(pg); lfc = np.log2((m.values + 1) / (TC.mean(axis=1).values + 1))
        for gi, x in enumerate(g): gtab.append((trait, nm, lab, "down (A to B)" if lab == "A to B" else "up (B to A)", x, float(TC.mean(axis=1).iloc[gi]), float(m.iloc[gi]), float(lfc[gi]), float(pg[gi]), float(qg[gi]), bool(lfc[gi] < 0) if lab == "A to B" else bool(lfc[gi] > 0), Cx.shape[1], len(rnas[nm])))
        p = stats.mannwhitneyu(m.values, dT[0].values, alternative="two-sided").pvalue; pvals.append(p)   # paper: two-sided Wilcoxon rank-sum on the genes
        res.append((trait, nm, lab, len(bins), len(g), labs[-1].replace("\n", " "), float(np.median(dT[0])), float(np.median(m)), p))
    disp = [np.log2(d.values + 1) for d in dT]; bs = [boxplot_stats(v, whis=1.5)[0] for v in disp]       # log2(TPM+1) display: TPM is very skewed (box would sit on the axis)
    bp = ax.boxplot(disp, showfliers=False, patch_artist=True, whis=1.5, widths=.5); ax.set_xticklabels(labs, fontsize=7)
    for b, c in zip(bp["boxes"], cols): b.set_facecolor(c)
    for md in bp["medians"]: md.set_color("k")
    yl = min(b["whislo"] for b in bs); yh = max(b["whishi"] for b in bs); rg = max(yh - yl, 1e-6)
    ax.set_ylim(yl - .08 * rg, yh + (.14 * len(nms) + .12) * rg); ax.set_ylabel("log2(TPM + 1)", fontsize=7); comps = []
    for k, pv_ in enumerate(pvals, start=1):
        yb = yh + rg * (.06 + .14 * (k - 1)); ax.plot([1, 1, k + 1, k + 1], [yb - rg * .02, yb, yb, yb - rg * .02], color="k", lw=.7)
        ax.text((k + 2) / 2, yb, f"P = {pv_:.2e} {stars(pv_)}", ha="center", va="bottom", fontsize=6.5); comps.append({"a": 0, "b": k, "p": float(pv_), "star": stars(pv_)})
    ttl = f"{TLAB(trait, nms[0])}: {lab}  ({len(g)} genes, {len(bins)} bins)" if len(nms) == 1 else f"{TCOL[trait]}: {lab}  ({len(g)} genes in recurrent switch bins)"
    ax.set_title(ttl, fontsize=7)
    boxjson.append({"trait": trait, "switch": lab, "title": ttl, "ylabel": "log2(TPM + 1)", "ylim": [float(ax.get_ylim()[0]), float(ax.get_ylim()[1])],
                    "boxes": [{"label": l_.replace("\n", " "), "color": c_, "q1": float(b["q1"]), "med": float(b["med"]), "q3": float(b["q3"]), "lo": float(b["whislo"]), "hi": float(b["whishi"])} for l_, c_, b in zip(labs, cols, bs)], "comps": comps})
for trait, names in fam.items():
    if not names: continue
    panels = [[n] for n in names] if trait == "genotype" else [names]
    fig, axs = plt.subplots(len(panels), 2, figsize=(7, 3.8 * len(panels)), squeeze=False)
    for r, nms in enumerate(panels):
        for j, (lab, sw) in enumerate((("A to B", A2B), ("B to A", B2A))): make_panel(axs[r, j], trait, nms, lab, sw)
    hs = [Patch(color=CTRL_COL, label="Control")] + ([Patch(color=COLS.get(n, "#7570b3"), label=TLAB(trait, n).replace(" patients", "")) for n in names] if trait != "genotype" else [Patch(color="#7570b3", label="B-ALL patients (genotype)")])
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0)); H_ = fig.get_figheight()
    fig.text(0.5, 1 - 0.42 / H_, f"Expression of genes in recurrent compartment-switch bins, by {TCOL[trait].lower()} (two-sided Wilcoxon rank-sum vs control;  * P<0.05  ** P<0.01  *** P<0.001  **** P<1e-4)", ha="center", fontsize=5.5)
    plt.tight_layout(rect=[0, 0, 1, 1 - 0.65 / H_]); plt.savefig(f"{a.outdir}/fig1e_{trait}.png", dpi=200); plt.savefig(f"{a.outdir}/fig1e_{trait}.pdf"); plt.close()
import json; json.dump(boxjson, open(f"{a.outdir}/fig1e_boxstats.json", "w"), indent=1)
R = pd.DataFrame(res, columns=["trait", "class", "switch", "n_bins", "n_genes", "patient_set", "median_TPM_control", "median_TPM_patients", "ranksum_P"]); pv = R["ranksum_P"].values; o = np.argsort(pv); q = np.empty(len(pv)); q[o] = np.minimum.accumulate((pv[o] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]; R["BH_FDR_all_panels"] = np.clip(q, 0, 1)
R.to_csv(f"{a.outdir}/fig1e_stats.tsv", sep="\t", index=False); print(R.to_string(index=False))
GT = pd.DataFrame(gtab, columns=["trait", "category", "switch", "direction", "gene", "mean_TPM_control", "mean_TPM_patients", "log2FC_patients_vs_control", "P_ranksum", "BH_FDR_within_category", "concordant_with_switch", "n_control", "n_patients"])
GT.to_csv(f"{a.outdir}/fig1e_gene_table_all.tsv", sep="\t", index=False)
GT[GT.switch == "A to B"].to_csv(f"{a.outdir}/fig1e_genes_down_A_to_B.tsv", sep="\t", index=False); GT[GT.switch == "B to A"].to_csv(f"{a.outdir}/fig1e_genes_up_B_to_A.tsv", sep="\t", index=False)
try:
    with pd.ExcelWriter(f"{a.outdir}/fig1e_tables.xlsx") as xw:
        R.to_excel(xw, "summary_per_category", index=False); GT[GT.switch == "A to B"].to_excel(xw, "down_A_to_B_genes", index=False); GT[GT.switch == "B to A"].to_excel(xw, "up_B_to_A_genes", index=False)
except Exception as ex: print("xlsx not written:", ex)
print(GT.groupby(["trait", "category", "switch"]).agg(genes=("gene", "size"), concordant=("concordant_with_switch", "sum"), P_lt_0_05=("P_ranksum", lambda x: int((x < .05).sum())), FDR_lt_0_05=("BH_FDR_within_category", lambda x: int((x < .05).sum()))).to_string())
pd.DataFrame(glist, columns=["trait", "class", "switch", "gene"]).to_csv(f"{a.outdir}/fig1e_gene_lists.tsv", sep="\t", index=False); print("\nwrote", a.outdir)
