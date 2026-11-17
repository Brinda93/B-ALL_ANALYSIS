#!/usr/bin/env python3
"""Extended Data Fig. 2 (B-ALL version): examples of genes with A/B compartment switch.
 a, c : per gene - PC1 (E1 of the 40-kb bin containing the TSS) and expression (log2(TPM+1)) in samples called A vs B at that bin.
 b    : for one example gene - ATAC-seq, H3K27ac and H3K27me3 (CUT&Tag) signal at the promoter (TSS +/- 1 kb), A vs B samples.
Samples are called A (E1 > +min_abs) or B (E1 < -min_abs) at the gene's bin.  P: two-sided Wilcoxon rank-sum (= Mann-Whitney U).
Box: median, quartiles, whiskers 1.5 x IQR.  Signal values are read from the bigwigs as they are (they must already be depth-normalised).
Genes: --genes WT1,POU2AF1,FGF13,BCL11B (paper) or leave empty: the script ranks all genes (>= --min_n samples per class, higher
expression in A) and takes the top --auto.  ed_fig2_gene_stats.tsv lists every gene so you can choose.

  python ed_fig2.py --out $A/compartments --groups $G --corr_dir $A/rna_batchcorr --gtf $A/gencode.v44.annotation.gtf.gz \
     --atac_map $B/ATAC-seq/atac_map69.tsv --atac_dir $B/ATAC-seq/atac_out/05_bigwig \
     --bigtable BIG-TABLE.xlsx --cuttag_dir /gpfs/Labs/Dovat/cut_and_tag_big_project2026/cuttag_out/BigWig \
     --outdir out/main/ed_fig2 [--genes IKZF1,... --example IKZF1]
"""
import argparse, glob, gzip, os, re, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ALIAS = {**{f"ND{i}": f"NDALL{i}" for i in range(2, 9)}, "AUS51": "AUSALL51", "AUS84": "AUSALL84", "AUS87": "ALL87", "AUS95": "ALL95", "AUS174": "PALNTB", "ALL19": "AUS19", "ALL53": "AUS53", "ALL57": "AUS57", "ALL59": "AUS59", "PVCRK": "PAVRCK", "PAVYCL": "PAVCYL", "ALL4364": "AALL4364", "W03": "W0", "ALL2": "AUS2", "ALL11": "AUS11", "ALL25": "AUS25", "ALL82": "AUS82", "SJ9": "SJBALL9", "YDP1ALL58HIC": "AUS58", "JRP8AUS174HICS8": "PALNTB"}
def key(s):
    s = re.split(r"[ (]", str(s).strip())[0]; s = re.sub(r"_MERGED$|_[A-Z]$", "", s.upper()); s = s.replace("W13-7", "W13"); s = re.sub(r"[^A-Z0-9]", "", s); s = re.sub(r"(?<=[A-Z])0+(?=\d)", "", s)
    return ALIAS.get(s, s)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="compartments folder containing eig/*.tsv"); ap.add_argument("--groups", required=True)
ap.add_argument("--corr_dir", required=True, help="rna_batchcorr folder (log2tpm_patients_uncorrected.tsv.gz, patient_annotation_used.tsv)")
ap.add_argument("--gtf", required=True); ap.add_argument("--outdir", default="ed_fig2")
ap.add_argument("--genes", default="", help="comma list; default: automatic"); ap.add_argument("--auto", type=int, default=4); ap.add_argument("--example", default="", help="gene for panel b (default: first gene)")
ap.add_argument("--res", type=int, default=40000); ap.add_argument("--min_abs", type=float, default=0.0, help="|E1| needed to call A/B (0 = sign only)")
ap.add_argument("--min_n", type=int, default=3, help="minimum samples in each class"); ap.add_argument("--exclude", default=""); ap.add_argument("--include_controls", action="store_true", help="add controls to PC1/ATAC/CUT&Tag boxes (expression is patients only)")
ap.add_argument("--ctrl_agree", type=float, default=0.8, help="fraction of controls that must agree on A (or B) for a bin to be the reference"); ap.add_argument("--min_fc", type=float, default=1.0, help="auto mode: minimum log2 fold change (A vs B) in expression"); ap.add_argument("--no_switch_filter", action="store_true", help="auto mode: do not require a switch relative to controls")
ap.add_argument("--win", type=int, default=400000, help="half-width of the PC1 track window (paper: 0.8 Mb total)"); ap.add_argument("--row_h", type=float, default=0.075, help="inches per sample row in the track panels"); ap.add_argument("--no_tracks", action="store_true")
ap.add_argument("--atac_map"); ap.add_argument("--atac_dir"); ap.add_argument("--bigtable"); ap.add_argument("--cuttag_dir", nargs="*", default=[]); ap.add_argument("--prom", type=int, default=1000)
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)

# ---- samples and PC1
g = pd.read_csv(a.groups, sep="\t", header=None, comment="#", dtype=str).iloc[:, :2]; g.columns = ["sample", "group"]; g = g[g["sample"].str.lower() != "sample"]
g["ctrl"] = (g.group.str.lower() == "control") | g["sample"].str.upper().str.startswith("CTRL"); g["k"] = g["sample"].map(key)
files = {os.path.basename(f)[:-4]: f for f in glob.glob(f"{a.out}/eig/*.tsv")}; fk = {}
for n in files: fk.setdefault(key(n), n)
g["file"] = [s if s in files else fk.get(k) for s, k in zip(g["sample"], g["k"])]; g = g[g.file.notna()].drop_duplicates("k")
ex = {key(x) for x in a.exclude.split(",") if x.strip()}; g = g[~g.k.isin(ex)]
use = g if a.include_controls else g[~g.ctrl]; pat = g[~g.ctrl]
print(f"samples with PC1: {len(g)} ({int((~g.ctrl).sum())} patients, {int(g.ctrl.sum())} controls); used for PC1/signal boxes: {len(use)}")
M = pd.concat({r.k: pd.read_csv(files[r.file], sep="\t").set_index(["chrom", "start"]).E1 for r in g.itertuples()}, axis=1)

# ---- genes
rows = []
for l in (gzip.open(a.gtf, "rt") if a.gtf.endswith(".gz") else open(a.gtf)):
    if l.startswith("#"): continue
    t = l.rstrip("\n").split("\t")
    if len(t) < 9 or t[2] != "gene": continue
    bt = re.search(r'gene_(?:type|biotype) "([^"]+)"', t[8]); nm = re.search(r'gene_name "([^"]+)"', t[8]); gi = re.search(r'gene_id "([^"]+)"', t[8])
    if bt and bt.group(1) != "protein_coding" and not (nm and nm.group(1) in a.genes.split(",")): continue
    rows.append((t[0], int(t[3]) if t[6] == "+" else int(t[4]), t[6], nm.group(1) if nm else "", gi.group(1).split(".")[0] if gi else "", int(t[3]), int(t[4])))
T = pd.DataFrame(rows, columns=["chrom", "tss", "strand", "name", "id", "gstart", "gend"]); T = T[~T.chrom.isin(["chrX", "chrY", "chrM"])].drop_duplicates("name").reset_index(drop=True); T["bin"] = (T.tss // a.res) * a.res

# ---- expression (patients with Hi-C)
U = pd.read_csv(f"{a.corr_dir}/log2tpm_patients_uncorrected.tsv.gz", sep="\t", index_col=0); U.index = U.index.astype(str).str.replace(r"\.\d+$", "", regex=True); U = U[~U.index.duplicated()]
S = pd.read_csv(f"{a.corr_dir}/patient_annotation_used.tsv", sep="\t", index_col=0); S["k"] = S["sample"].map(key); S = S[S.k.isin(set(pat.k))].drop_duplicates("k")
E = U[list(S.index)].copy(); E.columns = list(S.k); print(f"patients with RNA and PC1: {E.shape[1]}")

# ---- per-gene class and statistics
idx = M.index.get_indexer(list(zip(T.chrom, T.bin))); ok = idx >= 0
P1 = pd.DataFrame(np.where(ok[:, None], M.values[np.clip(idx, 0, None)], np.nan), index=T.name, columns=M.columns)
Tn = T.set_index("name")
def cls(v): return np.where(v > a.min_abs, "A", np.where(v < -a.min_abs, "B", ""))
def test(x, y): return stats.mannwhitneyu(x, y, alternative="two-sided").pvalue if len(x) > 0 and len(y) > 0 else np.nan
st = []
cols_e = [c for c in E.columns if c in P1.columns]
ck = [k for k in g[g.ctrl].k if k in M.columns]; Cm = M[ck] if ck else None; ref_bin = pd.Series("", index=M.index)
if Cm is not None and len(ck) >= 3:
    nv = Cm.notna().sum(axis=1); fA = (Cm > 0).sum(axis=1) / nv.where(nv > 0); ref_bin[(fA >= a.ctrl_agree) & (nv >= 3)] = "A"; ref_bin[(fA <= 1 - a.ctrl_agree) & (nv >= 3)] = "B"
    print(f"control reference from {len(ck)} controls: A bins {int((ref_bin == 'A').sum())}, B bins {int((ref_bin == 'B').sum())}")
else: print("WARNING: < 3 controls with PC1 - switch filter disabled"); a.no_switch_filter = True
ref_g = ref_bin.reindex(pd.MultiIndex.from_arrays([T.chrom, T.bin])).values; ref_g = pd.Series(ref_g, index=T.name)
for nm in T.name:
    gid = Tn.at[nm, "id"]
    if gid not in E.index: continue
    c = cls(P1.loc[nm, cols_e].values.astype(float)); e = E.loc[gid, cols_e].values.astype(float)
    xa, xb = e[c == "A"], e[c == "B"]
    if len(xa) < a.min_n or len(xb) < a.min_n: continue
    rf = ref_g[nm] if isinstance(ref_g[nm], str) else ""; nsw = int(((c == "B") if rf == "A" else (c == "A") if rf == "B" else np.zeros(len(c), bool)).sum())
    st.append((nm, Tn.at[nm, "chrom"], Tn.at[nm, "tss"], len(xa), len(xb), np.nanmean(xa), np.nanmean(xb), np.nanmean(xa) - np.nanmean(xb), test(xa, xb), rf, nsw, ("A_to_B" if rf == "A" else "B_to_A" if rf == "B" else "")))
ST = pd.DataFrame(st, columns=["gene", "chrom", "tss", "n_A", "n_B", "mean_log2TPM_A", "mean_log2TPM_B", "log2FC_A_vs_B", "P_expr", "control_ref", "n_patients_switched", "direction"]).sort_values("P_expr")
ST.to_csv(f"{a.outdir}/ed_fig2_gene_stats.tsv", sep="\t", index=False); print(f"{len(ST)} genes with >= {a.min_n} samples in both classes; table: ed_fig2_gene_stats.tsv")
for nm in ("WT1", "POU2AF1", "FGF13", "BCL11B"):
    r = ST[ST.gene == nm]; print(f"  paper gene {nm}: " + (f"nA={int(r.n_A.iloc[0])} nB={int(r.n_B.iloc[0])} P_expr={r.P_expr.iloc[0]:.2g} log2FC(A-B)={r.log2FC_A_vs_B.iloc[0]:.2f}" if len(r) else "not testable (too few samples per class or no RNA)"))
if a.genes: genes = [x.strip() for x in a.genes.split(",") if x.strip()]
else:
    cand = ST[ST.log2FC_A_vs_B >= a.min_fc]
    if not a.no_switch_filter: cand = cand[(cand.control_ref != "") & (cand.n_patients_switched >= a.min_n)]
    print(f"auto candidates (switch vs controls in >= {a.min_n} patients, log2FC(A-B) >= {a.min_fc}): {len(cand)}"); print(cand.head(15)[["gene", "control_ref", "direction", "n_A", "n_B", "n_patients_switched", "log2FC_A_vs_B", "P_expr"]].to_string(index=False))
    genes = list(cand.head(a.auto).gene)
genes = genes[:4]; print("genes shown:", genes)
if len(genes) < 2: sys.exit("need at least 2 genes - lower --min_n or give --genes")
ex_gene = a.example or genes[0]

# ---- promoter signal (panel b)
import pyBigWig
def prom_mean(path, chrom, s, e):
    try:
        bw = pyBigWig.open(path); cc = chrom if chrom in bw.chroms() else (chrom[3:] if chrom.startswith("chr") else "chr" + chrom); v = bw.stats(cc, max(0, s), e, type="mean", exact=True)[0]; bw.close(); return v
    except Exception: return None
atac = {}
if a.atac_map and a.atac_dir:
    m = pd.read_csv(a.atac_map, sep=None, engine="python", dtype=str)
    for _, r in m.iterrows():
        for i in re.split(r"[,\s;]+", str(r.iloc[1]).strip()):
            f = os.path.join(a.atac_dir, f"{i}.bw")
            if os.path.exists(f): atac.setdefault(key(r.iloc[0]), f); break
ct = {"H3K27ac": {}, "H3K27me3": {}}
if a.bigtable:
    bt = pd.read_excel(a.bigtable, sheet_name=0, header=None, dtype=str)
    hdr = [(str(x) if pd.notna(x) else "") + " | " + (str(y) if pd.notna(y) else "") for x, y in zip(bt.iloc[0], bt.iloc[1])]; hdr = [h.lower() for h in hdr]
    def col(pat, dflt):
        m = [i for i, h in enumerate(hdr) if re.search(pat, h)]; return m[0] if m else dflt
    cn, ca, c27, c3, cat = col(r"sample name", 1), col(r"ancestry", 13), col(r"h3k27ac", 31), col(r"h3k27me3", 33), col(r"atac", 40)
    print(f"big table columns used: name={cn} ancestry={ca} H3K27ac={c27} H3K27me3={c3} ATAC={cat}  (headers: {hdr[c27].strip()} / {hdr[c3].strip()} / {hdr[cat].strip()})")
    nid = {"H3K27ac": 0, "H3K27me3": 0}; nfile = {"H3K27ac": 0, "H3K27me3": 0}; ex_ids = []
    for r in bt.iloc[2:].itertuples(index=False):
        r = list(r)
        if pd.isna(r[cn]) or pd.isna(r[ca]): continue
        k = key(r[cn])
        for nm_, c_ in (("H3K27ac", c27), ("H3K27me3", c3)):
            ids = [x.strip() for x in str(r[c_]).split("/") if re.fullmatch(r"CT\d+", x.strip())] if pd.notna(r[c_]) else []
            nid[nm_] += len(ids); fs = [f for i in ids for f in [next((os.path.join(d, i + ".bw") for d in a.cuttag_dir if os.path.exists(os.path.join(d, i + ".bw"))), None)] if f]
            nfile[nm_] += len(fs)
            if fs: ct[nm_][k] = fs
            elif ids and len(ex_ids) < 4: ex_ids.append(ids[0])
        if k not in atac and pd.notna(r[cat]) and a.atac_dir:
            f = os.path.join(a.atac_dir, str(r[cat]).strip() + ".bw")
            if os.path.exists(f): atac[k] = f
    print(f"CUT&Tag ids in table: {nid}; bigwig files found: {nfile}" + (f"; e.g. missing {ex_ids}" if ex_ids else ""))
trk = {"ATAC-seq": {k: [f] for k, f in atac.items()}, **ct}
print({k: len(v) for k, v in trk.items()}, "samples with signal files")

# ---- figure
def stars(p): return "ns" if p >= .05 else "*" if p >= .01 else "**" if p >= .001 else "***" if p >= 1e-4 else "****"
CA, CB = "#3182bd", "#fd8d3c"
def box(ax, xa, xb, ylab, title=None, ital=False):
    xa = np.asarray([v for v in xa if v is not None and np.isfinite(v)], float); xb = np.asarray([v for v in xb if v is not None and np.isfinite(v)], float)
    bp = ax.boxplot([xa, xb], positions=[0, 1], widths=.55, whis=1.5, showfliers=False, patch_artist=True, medianprops=dict(color="k", lw=1.2), boxprops=dict(lw=.8), whiskerprops=dict(lw=.8), capprops=dict(lw=.8))
    for pt, c in zip(bp["boxes"], (CA, CB)): pt.set_facecolor(c); pt.set_alpha(.55)
    rng = np.random.default_rng(1)
    for i, (x, c) in enumerate(((xa, CA), (xb, CB))): ax.scatter(i + rng.uniform(-.16, .16, len(x)), x, s=7, color=c, edgecolor="k", lw=.3, zorder=3)
    p = test(xa, xb); allv = np.concatenate([xa, xb]); lo, hi = (allv.min(), allv.max()) if len(allv) else (0, 1); span = (hi - lo) or 1
    ax.plot([0, 0, 1, 1], [hi + .05 * span, hi + .09 * span, hi + .09 * span, hi + .05 * span], color="k", lw=.7)
    ax.text(.5, hi + .11 * span, f"P = {p:.2g} {stars(p)}" if np.isfinite(p) else "P = n/a", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylim(lo - .08 * span, hi + .3 * span); ax.set_xlim(-.6, 1.6); ax.set_xticks([0, 1]); ax.set_xticklabels([f"A\n(n={len(xa)})", f"B\n(n={len(xb)})"], fontsize=7); ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel(ylab, fontsize=7.5)
    if title: ax.set_title(title, fontsize=8.5, style="italic" if ital else "normal", pad=4)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    return len(xa), len(xb), p
fig = plt.figure(figsize=(11.5, 10)); gs = fig.add_gridspec(3, 4, hspace=.75, wspace=.55, height_ratios=[1, 1, 1]); caption = []; sel = []
for gi_, nm in enumerate(genes):
    r_, c_ = (0, 2 * gi_) if gi_ < 2 else (1, 2 * (gi_ - 2)); gid = Tn.at[nm, "id"]; ch, tss = Tn.at[nm, "chrom"], Tn.at[nm, "tss"]
    pc = P1.loc[nm]; use_k = [k for k in use.k if k in pc.index and np.isfinite(pc[k])]; cl = pd.Series(cls(pc[use_k].values.astype(float)), index=use_k)
    A_ = list(cl[cl == "A"].index); B_ = list(cl[cl == "B"].index)
    nA, nB, pp = box(fig.add_subplot(gs[r_, c_]), pc[A_].values, pc[B_].values, "PC1 (E1)", nm, True)
    ea = [E.at[gid, k] for k in A_ if k in E.columns and gid in E.index]; eb = [E.at[gid, k] for k in B_ if k in E.columns and gid in E.index]
    nAe, nBe, pe = box(fig.add_subplot(gs[r_, c_ + 1]), ea, eb, "Expression (log2(TPM+1))", nm, True)
    caption.append(f"{nm}: PC1 A (n = {nA}) B (n = {nB}), P = {pp:.2g}; expression A (n = {nAe}) B (n = {nBe}), P = {pe:.2g}")
    for k in A_ + B_: sel.append((nm, k, cl[k], pc[k], E.at[gid, k] if k in E.columns and gid in E.index else np.nan))
pd.DataFrame(sel, columns=["gene", "sample_key", "compartment", "PC1", "log2TPM1"]).to_csv(f"{a.outdir}/ed_fig2_per_sample_values.tsv", sep="\t", index=False)
# panel b
ch, tss = Tn.at[ex_gene, "chrom"], Tn.at[ex_gene, "tss"]; pc = P1.loc[ex_gene]; use_k = [k for k in use.k if k in pc.index and np.isfinite(pc[k])]; cl = pd.Series(cls(pc[use_k].values.astype(float)), index=use_k)
bsel = []
for j, tn in enumerate(("ATAC-seq", "H3K27ac", "H3K27me3")):
    vals = {}
    for k in use_k:
        fs = trk[tn].get(k)
        if not fs or cl[k] == "": continue
        v = [prom_mean(f, ch, tss - a.prom, tss + a.prom) for f in fs]; v = [x for x in v if x is not None and np.isfinite(x)]
        if v: vals[k] = float(np.mean(v)); bsel.append((ex_gene, tn, k, cl[k], vals[k]))
    xa = [v for k, v in vals.items() if cl[k] == "A"]; xb = [v for k, v in vals.items() if cl[k] == "B"]
    nA, nB, pp = box(fig.add_subplot(gs[2, j]), xa, xb, f"{tn} signal at promoter", f"{ex_gene} promoter (TSS ± {a.prom // 1000} kb)" if j == 1 else None, False)
    caption.append(f"{tn}: A (n = {nA}) B (n = {nB}), P = {pp:.2g}")
pd.DataFrame(bsel, columns=["gene", "assay", "sample_key", "compartment", "promoter_signal"]).to_csv(f"{a.outdir}/ed_fig2_promoter_signal.tsv", sep="\t", index=False)
fig.axes[0].text(-.5, 1.28, "a", transform=fig.axes[0].transAxes, fontsize=13, fontweight="bold"); fig.axes[4].text(-.5, 1.28, "c", transform=fig.axes[4].transAxes, fontsize=13, fontweight="bold"); fig.axes[8].text(-.5, 1.28, "b", transform=fig.axes[8].transAxes, fontsize=13, fontweight="bold")
fig.suptitle("Extended Data Fig. 2 | Examples of genes with A/B compartment switch (B-ALL)", fontsize=10, y=.995)
fig.savefig(f"{a.outdir}/ed_fig2.png", dpi=250, bbox_inches="tight"); fig.savefig(f"{a.outdir}/ed_fig2.pdf", bbox_inches="tight")
open(f"{a.outdir}/ed_fig2_legend.txt", "w").write("P value by two-sided Wilcoxon rank-sum test. " + "; ".join(caption) + ". Box plot: middle line denotes the median, top/bottom of boxes denote first/third quartiles and whiskers extend to 1.5 times the interquartile range.\n")
print("\n".join(caption)); print("wrote", f"{a.outdir}/ed_fig2.png")

# ======================================================================================================
# paper-style layout: a, c = per-sample PC1 tracks (red = A, purple = B) with the gene bin highlighted and expression (TPM) bars; b = boxes
if not a.no_tracks:
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpecFromSubplotSpec as GSS
    RED, PUR, TEAL = "#b2182b", "#7b6fd0", "#19b5b5"
    gk = g.drop_duplicates("k").set_index("k"); order = []; gcol = {}; pal = ["#f4e04d", "#a6d96a", "#4aa3df", "#c9a0dc", "#f5a65b", "#e88fb4", "#8fd3c8", "#d9d9d9"]
    pg = sorted(gk[~gk.ctrl].group.unique())
    for i_, gname in enumerate(pg): gcol[gname] = pal[i_ % 7]
    gcol["control"] = "#bdbdbd"
    for gname in pg + ["control"]:
        ks = [k for k in (gk[(gk.group == gname) & ~gk.ctrl].index if gname != "control" else gk[gk.ctrl].index) if k in M.columns]
        order += [(k, gk.at[k, "sample"], gname) for k in ks]
    import json; PPT = {"tracks": {}, "boxes": [], "groups": []}
    nrow = len(order); print(f"track panels: {nrow} sample rows, groups: {pg}")
    shortg = lambda x: ("Ctrl" if x == "control" else (re.match(r"G\d+", x).group(0) if re.match(r"G\d+", x) else x[:6]))
    PPT["groups"] = [{"name": gn, "short": shortg(gn), "color": gcol[gn], "count": sum(1 for o in order if o[2] == gn)} for gn in pg + ["control"] if sum(1 for o in order if o[2] == gn)]
    Hrow = nrow * a.row_h * 1.0 + 1.5; fig2 = plt.figure(figsize=(12.5, 2 * Hrow + 1.0)); gs0 = fig2.add_gridspec(2, 1, hspace=.42)
    def track_panel(spec, nm, show_strip):
        ch, tss, b0, gid = Tn.at[nm, "chrom"], int(Tn.at[nm, "tss"]), int(Tn.at[nm, "bin"]), Tn.at[nm, "id"]; lo, hi = tss - a.win, tss + a.win
        inner = GSS(2, 2, subplot_spec=spec, height_ratios=[1.3, nrow * a.row_h * 2.4], width_ratios=[3, 1], hspace=.02, wspace=.03)
        axg = fig2.add_subplot(inner[0, 0]); ax = fig2.add_subplot(inner[1, 0], sharex=axg); axe = fig2.add_subplot(inner[1, 1], sharey=ax)
        sub = M.xs(ch, level=0); sub = sub[(sub.index >= lo - a.res) & (sub.index <= hi)]; xs = sub.index.values + a.res / 2
        vm = np.nanpercentile(np.abs(sub.values), 98) or 1; amp = .46 / vm
        rows_json = []
        for i, (k, lab, gname) in enumerate(order):
            y0 = nrow - 1 - i; v = np.nan_to_num(sub[k].values.astype(float)) * amp; v = np.clip(v, -.5, .5); rows_json.append({"label": lab, "group": gname, "xs": [int(x) for x in xs], "v": [round(float(q), 4) for q in v]})
            ax.fill_between(xs, y0, y0 + v, where=v >= 0, color=RED, lw=0, interpolate=True); ax.fill_between(xs, y0, y0 + v, where=v < 0, color=PUR, lw=0, interpolate=True)
        ref = Tn.at[nm, "chrom"] and (ref_g.get(nm, "") if isinstance(ref_g.get(nm, ""), str) else "")
        ax.axvspan(b0, b0 + a.res, color="#f4a460" if ref != "B" else "#7aaed6", alpha=.4, zorder=0, lw=0)
        bnd = np.cumsum([sum(1 for o in order if o[2] == gname) for gname in pg + ["control"]])
        for b_ in bnd[:-1]: ax.axhline(nrow - b_ - .5 + .0, color="k", lw=.5)
        ax.set_xlim(lo, hi); ax.set_ylim(-.5, nrow - .5); ax.set_yticks([nrow - 1 - i for i in range(nrow)]); ax.set_yticklabels([o[1] for o in order], fontsize=3.4); ax.tick_params(axis="y", length=0, pad=1); ax.set_xticks([])
        for sp_ in ("top", "right", "bottom"): ax.spines[sp_].set_visible(False)
        if show_strip:
            st0 = 0
            for gname, cnt in zip(pg + ["control"], [sum(1 for o in order if o[2] == gn) for gn in pg + ["control"]]):
                if cnt == 0: continue
                ytop, ybot = nrow - st0 - .5, nrow - st0 - cnt - .5
                axw = max(ax.get_position().width * fig2.get_figwidth(), 1e-3); x0 = -(0.34 + 0.2) / axw; wd = 0.17 / axw
                ax.add_patch(Rectangle((x0, ybot), wd, ytop - ybot, transform=ax.get_yaxis_transform(), clip_on=False, color=gcol[gname], lw=0))
                ax.text(x0 + wd / 2, (ytop + ybot) / 2, shortg(gname), transform=ax.get_yaxis_transform(), rotation=90, ha="center", va="center", fontsize=5, fontweight="bold"); st0 += cnt
        tp = [(2 ** E.at[gid, k] - 1) if (k in E.columns and gid in E.index) else np.nan for k, _, _ in order]
        axe.barh([nrow - 1 - i for i in range(nrow)], np.nan_to_num(tp), height=.7, color=TEAL, lw=0)
        for b_ in bnd[:-1]: axe.axhline(nrow - b_ - .5, color="k", lw=.5)
        for rj, tv in zip(rows_json, tp): rj["tpm"] = (None if not np.isfinite(tv) else round(float(tv), 3))
        PPT["tracks"][nm] = {"gene": nm, "chrom": str(ch), "lo": int(lo), "hi": int(hi), "b0": int(b0), "res": int(a.res), "ref": ref, "rows": rows_json, "tpm_max": float(np.nanmax(tp)) if np.isfinite(tp).any() else 1.0,
            "genes": [{"name": r.name, "start": int(r.gstart), "end": int(r.gend), "focus": r.name == nm} for r in T[(T.chrom == ch) & (T.gend > lo) & (T.gstart < hi)].itertuples()]}
        mx = np.nanmax(tp) if np.isfinite(tp).any() else 1; axe.set_xlim(0, mx * 1.08 or 1); axe.set_xticks([0, round(mx / 2, 1 if mx < 20 else 0), round(mx, 1 if mx < 20 else 0)] if mx > 0 else [0]); axe.tick_params(labelsize=4.5, length=2, pad=1); axe.tick_params(axis="y", left=False, labelleft=False)
        for sp_ in ("top", "right"): axe.spines[sp_].set_visible(False)
        axe.set_xlabel("Expression (TPM)", fontsize=5, labelpad=1)
        axg.set_xlim(lo, hi); axg.axis("off"); G_ = T[(T.chrom == ch) & (T.gend > lo) & (T.gstart < hi)]
        for r in G_.itertuples():
            foc = r.name == nm; axg.add_patch(Rectangle((max(r.gstart, lo), .25), min(r.gend, hi) - max(r.gstart, lo), .28, color="k" if foc else "#555555", lw=0))
        axg.text((max(Tn.at[nm, "gstart"], lo) + min(Tn.at[nm, "gend"], hi)) / 2, .62, nm, ha="center", va="bottom", fontsize=6, style="italic", fontweight="bold"); axg.set_ylim(0, 1.2)
        axg.text(.5, 1.18, f"{ch}: {lo / 1e6:.1f} Mb-{hi / 1e6:.1f} Mb", transform=axg.transAxes, ha="center", va="bottom", fontsize=6.5)
        return ax
    top = GSS(1, 2, subplot_spec=gs0[0], width_ratios=[1.05, 1.0], wspace=.22); track_panel(top[0], genes[0], True)
    # b: boxes for genes[0]
    bx = GSS(2, 2, subplot_spec=top[1], hspace=.5, wspace=.45); g0 = genes[0]; ch0, tss0 = Tn.at[g0, "chrom"], int(Tn.at[g0, "tss"]); pc0 = P1.loc[g0]
    use_k0 = [k for k in use.k if k in pc0.index and np.isfinite(pc0[k])]; cl0 = pd.Series(cls(pc0[use_k0].values.astype(float)), index=use_k0)
    def fmtp(p): return "P = n/a" if not np.isfinite(p) else (f"P = {p:.1E}" if p < .01 else f"P = {p:.2g}")
    def box2(ax, xb, xa, ylab, title):
        xb = np.asarray([v for v in xb if v is not None and np.isfinite(v)], float); xa = np.asarray([v for v in xa if v is not None and np.isfinite(v)], float)
        bp = ax.boxplot([xb, xa], positions=[0, 1], widths=.6, whis=1.5, showfliers=False, patch_artist=True, medianprops=dict(color="k", lw=1), boxprops=dict(lw=.7), whiskerprops=dict(lw=.7), capprops=dict(lw=.7))
        for pt, c in zip(bp["boxes"], ("#4f7ea8", "#e8876a")): pt.set_facecolor(c)
        rng = np.random.default_rng(2)
        for i, x in enumerate((xb, xa)): ax.scatter(i + rng.uniform(-.14, .14, len(x)), x, s=6, color="k", zorder=3, lw=0)
        pv = test(xb, xa); allv = np.concatenate([xb, xa]) if len(xb) + len(xa) else np.array([0, 1]); hi_ = allv.max(); span = (allv.max() - allv.min()) or 1
        ax.plot([0, 0, 1, 1], [hi_ + .04 * span, hi_ + .07 * span, hi_ + .07 * span, hi_ + .04 * span], color="k", lw=.6); ax.text(.5, hi_ + .09 * span, fmtp(pv), ha="center", va="bottom", fontsize=6)
        ax.set_ylim(min(0, allv.min()) - .05 * span if allv.min() < 0 else -.03 * span, hi_ + .28 * span); ax.set_xticks([0, 1]); ax.set_xticklabels(["B", "A"], fontsize=7); ax.set_xlim(-.6, 1.6); ax.tick_params(axis="y", labelsize=6)
        ax.set_ylabel(ylab, fontsize=6.5); ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
        return len(xb), len(xa), pv
    cap_b = []
    def sig(tn):
        out = {}
        for k in use_k0:
            fs = trk[tn].get(k)
            if not fs or cl0[k] == "": continue
            v = [prom_mean(f, ch0, tss0 - a.prom, tss0 + a.prom) for f in fs]; v = [x for x in v if x is not None and np.isfinite(x)]
            if v: out[k] = float(np.mean(v))
        return out
    spec_b = [("PC1", None, "Averaged PC1 value"), ("H3K27ac", "H3K27ac", "Normalized intensities"), ("H3K27me3", "H3K27me3", "Normalized intensities"), ("ATAC-seq", "ATAC-seq", "Normalized intensities")]
    for j, (ttl, tn, yl) in enumerate(spec_b):
        axb = fig2.add_subplot(bx[j // 2, j % 2])
        if tn is None: d_ = {k: float(pc0[k]) for k in use_k0 if cl0[k] != ""}
        else: d_ = sig(tn)
        vb_ = [v for k, v in d_.items() if cl0[k] == "B"]; va_ = [v for k, v in d_.items() if cl0[k] == "A"]
        nB_, nA_, pv_ = box2(axb, vb_, va_, yl, ttl); PPT["boxes"].append({"title": ttl, "ylab": yl, "B": [float(x) for x in vb_], "A": [float(x) for x in va_], "p": (None if not np.isfinite(pv_) else float(pv_)), "gene": g0}); cap_b.append(f"{ttl}: A (n = {nA_}) B (n = {nB_}), P = {pv_:.2g}")
    bot = GSS(1, 3, subplot_spec=gs0[1], wspace=.3)
    for j, nm in enumerate(genes[1:4]): track_panel(bot[j], nm, j == 0)
    fig2.text(.005, .985, "a", fontsize=14, fontweight="bold"); fig2.text(.50, .985, "b", fontsize=14, fontweight="bold"); fig2.text(.005, .49, "c", fontsize=14, fontweight="bold")
    from matplotlib.lines import Line2D
    fig2.legend(handles=[Rectangle((0, 0), 1, 1, color=RED), Rectangle((0, 0), 1, 1, color=PUR), Rectangle((0, 0), 1, 1, color="#f4a460", alpha=.5), Rectangle((0, 0), 1, 1, color="#7aaed6", alpha=.5)], labels=["PC1 > 0 (A)", "PC1 < 0 (B)", "gene bin; control = A (A to B in patients)", "gene bin; control = B (B to A in patients)"], loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(.5, -.005))
    fig2.savefig(f"{a.outdir}/ed_fig2_tracks.png", dpi=220, bbox_inches="tight"); fig2.savefig(f"{a.outdir}/ed_fig2_tracks.pdf", bbox_inches="tight")
    open(f"{a.outdir}/ed_fig2_tracks_legend.txt", "w").write(f"Panel b at the {genes[0]} promoter (TSS +/- {a.prom // 1000} kb). P value by two-sided Wilcoxon rank-sum test. " + "; ".join(cap_b) + ". Box plot: middle line denotes the median, top/bottom of boxes denote first/third quartiles and whiskers extend to 1.5 times the interquartile range.\n")
    print("\n".join(cap_b)); print("wrote", f"{a.outdir}/ed_fig2_tracks.png")
    # ---- individual panels a, b, c (PNG/PDF) + data for the editable PowerPoint
    def save_tracks(glist, name):
        global fig2
        fig2 = plt.figure(figsize=(4.9 * len(glist) + .6, nrow * a.row_h + 1.35)); gsx = fig2.add_gridspec(1, len(glist), wspace=.42)
        for j, nm in enumerate(glist): track_panel(gsx[j], nm, j == 0)
        fig2.savefig(f"{a.outdir}/{name}.png", dpi=220, bbox_inches="tight"); fig2.savefig(f"{a.outdir}/{name}.pdf", bbox_inches="tight"); plt.close(fig2); print("wrote", f"{a.outdir}/{name}.png")
    save_tracks([genes[0]], "ed_fig2a"); save_tracks(genes[1:4], "ed_fig2c")
    fig3 = plt.figure(figsize=(6.4, 5.4)); b3 = fig3.add_gridspec(2, 2, hspace=.5, wspace=.45)
    for j, bd in enumerate(PPT["boxes"]): box2(fig3.add_subplot(b3[j // 2, j % 2]), bd["B"], bd["A"], bd["ylab"], bd["title"])
    fig3.suptitle(f"{genes[0]} promoter (TSS ± {a.prom // 1000} kb)", fontsize=8, y=.99); fig3.savefig(f"{a.outdir}/ed_fig2b.png", dpi=220, bbox_inches="tight"); fig3.savefig(f"{a.outdir}/ed_fig2b.pdf", bbox_inches="tight"); plt.close(fig3); print("wrote", f"{a.outdir}/ed_fig2b.png")
    PPT["genes"] = genes; PPT["caption_b"] = open(f"{a.outdir}/ed_fig2_tracks_legend.txt").read().strip(); PPT["row_h"] = a.row_h
    json.dump(PPT, open(f"{a.outdir}/ed_fig2_pptdata.json", "w")); print("wrote", f"{a.outdir}/ed_fig2_pptdata.json")
