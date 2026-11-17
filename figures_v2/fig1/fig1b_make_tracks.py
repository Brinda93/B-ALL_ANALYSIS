#!/usr/bin/env python3
"""Build the track table for Fig 1b (multi-track example region) and print the region + command.

For each genotype group (and control) it picks up to --per_group samples that have the requested data, then writes one
row per sample and assay: compartment PC1 (eig file), ATAC bigwig, CTCF bigwig.  Samples are matched across naming schemes
with the same key() normalisation as compartments_fig1e.py.  Samples missing an assay are skipped for that assay (reported).

  python fig1b_make_tracks.py --groups $G --eig_dir $A/compartments/eig \
     --atac_map $B/ATAC-seq/atac_map69.tsv --atac_dir $B/ATAC-seq/atac_out/05_bigwig \
     --ctcf_dir $B/Chip-seq/chipseq_out/hg38_old/05_bigwig \
     --gtf $A/gencode.v44.annotation.gtf.gz --gene IKZF1 --flank 1500000 --out out/fig1b_tracks.tsv
then run the printed `fig1_panels.py b ...` command.
"""
import argparse, glob, gzip, os, re
import pandas as pd

ALIAS = {**{f"ND{i}": f"NDALL{i}" for i in range(2, 9)}, "AUS51": "AUSALL51", "AUS84": "AUSALL84", "AUS87": "ALL87", "AUS95": "ALL95", "AUS174": "PALNTB", "ALL19": "AUS19", "ALL53": "AUS53", "ALL57": "AUS57", "ALL59": "AUS59", "PVCRK": "PAVRCK", "PAVYCL": "PAVCYL", "ALL4364": "AALL4364", "W03": "W0", "ALL2": "AUS2", "ALL11": "AUS11", "ALL25": "AUS25", "ALL82": "AUS82", "SJ9": "SJBALL9", "YDP1ALL58HIC": "AUS58", "JRP8AUS174HICS8": "PALNTB"}
def key(s):
    s = re.split(r"[ (]", str(s).strip())[0]; s = re.sub(r"_MERGED$|_[A-Z]$", "", s.upper()); s = s.replace("W13-7", "W13"); s = re.sub(r"[^A-Z0-9]", "", s); s = re.sub(r"(?<=[A-Z])0+(?=\d)", "", s)
    return ALIAS.get(s, s)

ap = argparse.ArgumentParser()
for k in ("groups", "eig_dir", "atac_map", "atac_dir", "ctcf_dir", "gtf", "out"): ap.add_argument("--" + k, required=(k in ("groups", "out")))
ap.add_argument("--gene", default="IKZF1"); ap.add_argument("--flank", type=int, default=1500000); ap.add_argument("--per_group", type=int, default=1)
ap.add_argument("--groups_show", default="", help="comma list of group names to show (default: all patient groups with data + control)")
ap.add_argument("--samples", default="", help="comma list of exact sample names to force (overrides automatic pick)")
ap.add_argument("--require", default="eig,atac,ctcf", help="assays a sample must have to be auto-picked")
a = ap.parse_args()

G = pd.read_csv(a.groups, sep="\t", header=None, comment="#", dtype=str).iloc[:, :2]; G.columns = ["sample", "group"]; G = G[G["sample"].str.lower() != "sample"]
G["k"] = G["sample"].map(key)
eig = {key(os.path.basename(f)[:-4]): f for f in glob.glob(f"{a.eig_dir}/*.tsv")} if a.eig_dir else {}
atac = {}
if a.atac_map and a.atac_dir:
    m = pd.read_csv(a.atac_map, sep=None, engine="python", dtype=str)
    for _, r in m.iterrows():
        ids = re.split(r"[,\s;]+", str(r.iloc[1]).strip())
        for i in ids:
            f = os.path.join(a.atac_dir, f"{i}.bw")
            if os.path.exists(f): atac.setdefault(key(r.iloc[0]), f); break
ctcf = {}
if a.ctcf_dir:
    for f in sorted(glob.glob(f"{a.ctcf_dir}/*CTCF*.bw") + glob.glob(f"{a.ctcf_dir}/*ctcf*.bw")):
        st = os.path.basename(f)[:-3]; st = re.sub(r"[-_]?CTCF[-_]?", "_", st, flags=re.I); st = re.sub(r"[-_]P\d+", "", st); st = st.strip("_-")
        ctcf.setdefault(key(st), f)
have = {"eig": eig, "atac": atac, "ctcf": ctcf}; req = [x for x in a.require.split(",") if x]
print(f"[tracks] eig {len(eig)}, ATAC {len(atac)}, CTCF {len(ctcf)} samples resolved")
isctl = lambda s, g: g.lower() == "control" or s.upper().startswith("CTRL")
G["ctrl"] = [isctl(s, g) for s, g in zip(G["sample"], G["group"])]
groups = [g for g in sorted(G[~G.ctrl].group.unique())]
if a.groups_show: groups = [g for g in groups if g in a.groups_show.split(",")]
picked = []
if a.samples:
    for s_ in a.samples.split(","): r = G[G["sample"] == s_.strip()]; picked += [(r.iloc[0].group if len(r) else "?", s_.strip())] if len(r) else [("?", s_.strip())]
else:
    for g in groups + ["control"]:
        sub = G[(G.ctrl if g == "control" else (G.group == g) & ~G.ctrl)]
        ok = [r for r in sub.itertuples() if all(r.k in have[x] for x in req)]
        if not ok and g == "control": ok = [r for r in sub.itertuples() if r.k in eig]      # controls often lack ATAC/CTCF: PC1 only
        if not ok: print(f"[tracks] {g}: no sample with {req}"); continue
        picked += [(g, r.sample) for r in ok[:a.per_group]]
print("[tracks] picked:", picked)
short = lambda g: "ctrl" if g == "control" else re.match(r"G\d+", g).group(0) if re.match(r"G\d+", g) else g
rows = []
for assay, col, name in (("eig", "#333333", "Compartment PC1 (E1)"), ("atac", "#1b7837", "ATAC-seq"), ("ctcf", "#2166ac", "CTCF")):
    for g, s_ in picked:
        f = have[assay].get(key(s_))
        if f: rows.append((f"{short(g)} {s_}", f, col, 1.0, name))
        else: print(f"[tracks] {s_}: no {assay} file")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
pd.DataFrame(rows, columns=["label", "path", "color", "scale", "assay"]).to_csv(a.out, sep="\t", index=False)
reg = ""
if a.gtf:
    for l in (gzip.open(a.gtf, "rt") if a.gtf.endswith(".gz") else open(a.gtf)):
        if l.startswith("#"): continue
        f = l.rstrip("\n").split("\t")
        if len(f) > 8 and f[2] == "gene" and re.search(rf'gene_name "{a.gene}"', f[8]): reg = f"{f[0]}:{max(1, int(f[3]) - a.flank)}-{int(f[4]) + a.flank}"; break
print(f"\nwrote {a.out} ({len(rows)} tracks); region {reg}")
print(f"python scripts/fig1_panels.py b --tracks {a.out} --region {reg} --gene {a.gene} --gtf {a.gtf or '<gtf>'} --out out/main/fig1b")
