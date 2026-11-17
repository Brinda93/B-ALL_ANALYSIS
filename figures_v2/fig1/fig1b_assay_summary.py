#!/usr/bin/env python3
"""Fig 1b: number of samples per sequencing approach (Hi-C, ATAC, ChIP/CUT&Tag marks, RNA, WGS, Nanopore, Bionano ...).

Counts are matched to the samples in the groups table (patients per genotype group + controls), using the same sample-name
normalisation as the other Fig 1 scripts.  Writes
  <out>_samples_per_assay.tsv   assay, n_patients, n_controls, n_total, n_files_unmatched
  <out>_assay_by_group.tsv      assay x genotype group counts
  <out>_manifest.tsv            sample, group, assay   (input for `fig1_panels.py a`, the sample x assay dot matrix)
  <out>_unmatched.tsv           source names that did not match any sample in the groups table (check aliases!)
  <out>_1b.png / .pdf           stacked bars (patients / controls) per assay + per-group count table

Example:
  python fig1b_assay_summary.py --groups $G --eig_dir $A/compartments/eig \
    --atac_map $B/ATAC-seq/atac_map69.tsv --atac_dir $B/ATAC-seq/atac_out/05_bigwig \
    --chip_dir $B/Chip-seq/chipseq_out/hg38_old/05_bigwig \
    --rna_dir $A/rna_tpm_bigtable_dir \
    --dirs "WGS=/gpfs/Scratch/DovatLab/PDX_project/WGS/SPEEDSEQ,Nanopore=/gpfs/Scratch/DovatLab/QUO1018273/nanopore,Bionano=/gpfs/Scratch/DovatLab/Laura/bionano/allzip" \
    --out out/main/fig1b
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ALIAS = {**{f"ND{i}": f"NDALL{i}" for i in range(2, 9)}, "AUS51": "AUSALL51", "AUS84": "AUSALL84", "AUS87": "ALL87", "AUS95": "ALL95", "AUS174": "PALNTB", "ALL19": "AUS19", "ALL53": "AUS53", "ALL57": "AUS57", "ALL59": "AUS59", "PVCRK": "PAVRCK", "PAVYCL": "PAVCYL", "ALL4364": "AALL4364", "W03": "W0", "ALL2": "AUS2", "ALL11": "AUS11", "ALL25": "AUS25", "ALL82": "AUS82", "SJ9": "SJBALL9", "YDP1ALL58HIC": "AUS58", "JRP8AUS174HICS8": "PALNTB"}
def key(s):
    s = re.split(r"[ (]", str(s).strip())[0]; s = re.sub(r"_MERGED$|_[A-Z]$", "", s.upper()); s = s.replace("W13-7", "W13"); s = re.sub(r"[^A-Z0-9]", "", s); s = re.sub(r"(?<=[A-Z])0+(?=\d)", "", s)
    return ALIAS.get(s, s)

ap = argparse.ArgumentParser()
ap.add_argument("--groups", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--eig_dir"); ap.add_argument("--atac_map"); ap.add_argument("--atac_dir"); ap.add_argument("--chip_dir", nargs="*", default=[])
ap.add_argument("--marks", default="CTCF,H3K27ac,H3K27me3,H3K4me1,H3K4me3,H3K9me3,H3K36me3,H3K4me2", help="histone/TF marks to look for in chip file names")
ap.add_argument("--rna_dir", help="folder with patient_columns.txt and control_columns.txt (RNA-seq table)"); ap.add_argument("--rna_table", help="alternative: TPM table whose columns are samples")
ap.add_argument("--dirs", default="", help="name=dir,name=dir : one entry (file or folder) per sample, e.g. WGS, Nanopore, Bionano")
ap.add_argument("--hic_name", default="Hi-C")
a = ap.parse_args()

G = pd.read_csv(a.groups, sep="\t", header=None, comment="#", dtype=str).iloc[:, :2]; G.columns = ["sample", "group"]; G = G[G["sample"].str.lower() != "sample"]
G["k"] = G["sample"].map(key); G["ctrl"] = [g.lower() == "control" or s.upper().startswith("CTRL") for s, g in zip(G["sample"], G["group"])]
G = G.drop_duplicates("k"); kmap = dict(zip(G.k, range(len(G))))
sets = {}; unmatched = []

def add(assay, names, note=""):
    ks = set();
    for n in names:
        k = key(n)
        if k in kmap: ks.add(k)
        else: unmatched.append((assay, str(n)))
    sets[assay] = sets.get(assay, set()) | ks

if a.eig_dir: add(a.hic_name, [os.path.basename(f)[:-4] for f in glob.glob(f"{a.eig_dir}/*.tsv")])
if a.atac_map and a.atac_dir:
    m = pd.read_csv(a.atac_map, sep=None, engine="python", dtype=str); have = {os.path.basename(f)[:-3] for f in glob.glob(f"{a.atac_dir}/*.bw")}
    ok = [r.iloc[0] for _, r in m.iterrows() if any(i in have for i in re.split(r"[,\s;]+", str(r.iloc[1]).strip()))]
    add("ATAC-seq", ok)
marks = [x for x in a.marks.split(",") if x]
for d in a.chip_dir:
    for f in sorted(glob.glob(f"{d}/*.bw")):
        st = os.path.basename(f)[:-3]; mk = next((m_ for m_ in marks if re.search(m_, st, re.I)), None)
        if not mk: continue
        st = re.sub(mk, "_", st, flags=re.I); st = re.sub(r"[-_]P\d+", "", st); st = st.strip("_-")
        add(mk, [st])
if a.rna_dir:
    for fn, tag in (("patient_columns.txt", "RNA-seq"), ("control_columns.txt", "RNA-seq")):
        f = os.path.join(a.rna_dir, fn)
        if os.path.exists(f): add(tag, [l.strip().split("\t")[0] for l in open(f) if l.strip()])
if a.rna_table:
    add("RNA-seq", list(pd.read_csv(a.rna_table, sep="\t", nrows=1, index_col=0).columns))
for ent in [e for e in a.dirs.split(",") if e.strip()]:
    nm, d = ent.split("=", 1); names = [re.sub(r"\.(zip|bam|vcf|gz|xmap|cmap|bnx|tar)$", "", n, flags=re.I) for n in os.listdir(d)]; add(nm.strip(), names)

order = ([a.hic_name] if a.hic_name in sets else []) + [x for x in sets if x != a.hic_name]
rows = []; byg = {}
for assay in order:
    ks = sets[assay]; sub = G[G.k.isin(ks)]
    rows.append((assay, int((~sub.ctrl).sum()), int(sub.ctrl.sum()), len(sub), sum(1 for x, _ in unmatched if x == assay)))
    byg[assay] = sub.assign(g=np.where(sub.ctrl, "control", sub.group)).groupby("g").size()
S = pd.DataFrame(rows, columns=["assay", "n_patients", "n_controls", "n_total", "n_files_unmatched"]); S.to_csv(a.out + "_samples_per_assay.tsv", sep="\t", index=False)
BG = pd.DataFrame(byg).T.fillna(0).astype(int)
gcols = [c for c in sorted(G[~G.ctrl].group.unique()) if c in BG.columns] + (["control"] if "control" in BG.columns else []); BG = BG.reindex(columns=gcols, fill_value=0); BG.to_csv(a.out + "_assay_by_group.tsv", sep="\t")
man = [(r.sample, "control" if r.ctrl else r.group, assay) for assay in order for r in G[G.k.isin(sets[assay])].itertuples()]
pd.DataFrame(man, columns=["sample", "group", "assay"]).to_csv(a.out + "_manifest.tsv", sep="\t", index=False)
pd.DataFrame(unmatched, columns=["assay", "source_name"]).to_csv(a.out + "_unmatched.tsv", sep="\t", index=False)
print(S.to_string(index=False)); print("\nby group:\n", BG.to_string())
npat = int((~G.ctrl).sum()); nct = int(G.ctrl.sum()); print(f"\ngroups table: {npat} patients + {nct} controls; {len(unmatched)} source names not matched (see {a.out}_unmatched.tsv)")

# ---- figure
n = len(S); fig = plt.figure(figsize=(10, 0.45 * n + 1.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.5], wspace=.08)
ax = fig.add_subplot(gs[0]); y = np.arange(n)[::-1]
ax.barh(y, S.n_patients, color="#7570b3", label=f"B-ALL patients (of {npat})"); ax.barh(y, S.n_controls, left=S.n_patients, color="#fc8d59", label=f"controls (of {nct})")
for yy, r in zip(y, S.itertuples()): ax.text(r.n_total + .8, yy, f"{r.n_patients} + {r.n_controls}" if r.n_controls else f"{r.n_patients}", va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels(S.assay, fontsize=8); ax.set_xlabel("number of samples", fontsize=8); ax.set_xlim(0, max(S.n_total.max() * 1.18, 5)); ax.legend(fontsize=7, loc="lower right", frameon=False)
for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
ax2 = fig.add_subplot(gs[1]); M = BG.reindex(order).values; ax2.imshow(M, cmap="Purples", aspect="auto", vmin=0, vmax=max(M.max(), 1))
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax2.text(j, i, M[i, j], ha="center", va="center", fontsize=7, color="white" if M[i, j] > .6 * M.max() else "black")
ax2.set_xticks(range(len(gcols))); ax2.set_xticklabels([re.sub(r"_.*", "", c) if re.match(r"G\d+_", c) else c for c in gcols], fontsize=8); ax2.xaxis.tick_top(); ax2.set_yticks([])
fig.suptitle("Fig 1b  samples per sequencing approach (right: per genotype group)", fontsize=9, y=.995); fig.savefig(a.out + "_1b.png", dpi=200, bbox_inches="tight"); fig.savefig(a.out + "_1b.pdf", bbox_inches="tight")
print("wrote", a.out + "_1b.png")
