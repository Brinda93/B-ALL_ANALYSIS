#!/usr/bin/env python3
"""Fig 1b (AML-paper layout) for ONE patient: Hi-C triangle, genes, ATAC-seq, H3K27ac, H3K27me3, CTCF, RNA-seq (+)/(-), WGBS.
Every track is optional - give only the bigwigs you have; missing ones are skipped.  Values are drawn as in the bigwig
(normalise to depth first: bamCoverage --normalizeUsing CPM).  y-axis maximum is printed at the top-left of each track.

  python fig1b_plot.py --sample CHLA21 --cool CHLA21.mcool --resolution 10000 \
     --gene IKZF1 --flank 500000 --gtf $A/gencode.v44.annotation.gtf.gz \
     --atac ATAC.bw --k27ac H3K27ac.bw --k27me3 H3K27me3.bw --ctcf CTCF.bw \
     --rna_plus RNA.fwd.bw --rna_minus RNA.rev.bw --wgbs WGBS.bw --out out/main/fig1b_CHLA21
  (or --region chr7:50,000,000-51,000,000;  --tpm TABLE --tpm_sample NAME adds gene-level RNA bars if you have no RNA bigwig)
RNA (+)/(-) from a BAM (strand-specific):
  bamCoverage -b X.bam --filterRNAstrand forward --normalizeUsing CPM -bs 50 -o X.fwd.bw ; same with reverse -> X.rev.bw
"""
import argparse, gzip, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ap = argparse.ArgumentParser()
ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cool"); ap.add_argument("--resolution", type=int, default=10000)
ap.add_argument("--region"); ap.add_argument("--gene"); ap.add_argument("--flank", type=int, default=500000)
ap.add_argument("--gtf", required=True); ap.add_argument("--biotype", default="protein_coding"); ap.add_argument("--max_genes", type=int, default=25)
for k, lab in (("atac", "ATAC-seq"), ("k27ac", "H3K27ac"), ("k27me3", "H3K27me3"), ("ctcf", "CTCF"), ("rna_plus", "RNA-seq (+)"), ("rna_minus", "RNA-seq (-)"), ("wgbs", "WGBS")): ap.add_argument("--" + k, help=f"bigwig for {lab}")
ap.add_argument("--extra", action="append", default=[], help='extra track, repeatable: "LABEL=path.bw" or "LABEL=path.bw=#hexcolour" (e.g. H3K27ac CUT&Tag, WGS depth, HDAC1)')
ap.add_argument("--tpm"); ap.add_argument("--tpm_sample")
ap.add_argument("--bins", type=int, default=1000); ap.add_argument("--hic_vmax", type=float); ap.add_argument("--hic_scale", type=float, default=1e3, help="multiplier on balanced values")
ap.add_argument("--hic_pct", type=float, default=97, help="colour-bar max = this percentile if --hic_vmax not set")
ap.add_argument("--raw", action="store_true", help="use raw counts instead of balanced")
a = ap.parse_args()
STY = [("atac", "ATAC-seq", "#7a1f5c"), ("k27ac", "H3K27ac", "#8b1a1a"), ("k27me3", "H3K27me3", "#2e6f5e"), ("ctcf", "CTCF", "#6b6b1f"),
       ("rna_plus", "RNA-seq (+)", "#cc2222"), ("rna_minus", "RNA-seq (-)", "#e08a1e"), ("wgbs", "WGBS", "#8c8c8c")]

def read_genes():
    rows = []
    for l in (gzip.open(a.gtf, "rt") if a.gtf.endswith(".gz") else open(a.gtf)):
        if l.startswith("#"): continue
        f = l.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != "gene": continue
        nm = re.search(r'gene_name "([^"]+)"', f[8]); bt = re.search(r'gene_(?:type|biotype) "([^"]+)"', f[8]); gi = re.search(r'gene_id "([^"]+)"', f[8])
        if a.biotype != "all" and (not bt or bt.group(1) != a.biotype) and not (a.gene and nm and nm.group(1) == a.gene): continue
        rows.append((f[0], int(f[3]), int(f[4]), f[6], nm.group(1) if nm else "", gi.group(1).split(".")[0] if gi else ""))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "strand", "name", "id"])
GN = read_genes()
if a.region: ch, rest = a.region.split(":"); s, e = [int(x.replace(",", "")) for x in rest.split("-")]
elif a.gene:
    g = GN[GN.name == a.gene]
    if g.empty: raise SystemExit(f"gene {a.gene} not found in GTF")
    g = g.iloc[0]; ch = g.chrom; s = max(1, g.start - a.flank); e = g.end + a.flank
else: raise SystemExit("give --gene or --region")
print(f"region {ch}:{s:,}-{e:,}")
G = GN[(GN.chrom == ch) & (GN.end > s) & (GN.start < e)].copy(); G["len"] = G.end - G.start
if len(G) > a.max_genes: keep = set(G.nlargest(a.max_genes, "len").name) | ({a.gene} if a.gene else set()); G = G[G.name.isin(keep)]
G = G.sort_values("start")

import pyBigWig
x = np.linspace(s, e, a.bins); tracks = []
for key, lab, col in STY:
    p = getattr(a, key)
    if not p: continue
    try:
        bw = pyBigWig.open(p); cc = ch if ch in bw.chroms() else (ch[3:] if ch.startswith("chr") else "chr" + ch)
        v = np.nan_to_num(np.array(bw.stats(cc, s, e, nBins=a.bins, exact=False), dtype=float))
    except Exception as ex: print(f"WARNING: {lab}: {ex} - skipped"); continue
    if key == "rna_minus": v = -np.abs(v)
    tracks.append((key, lab, col, v))
EXC = ["#1b6ca8", "#b8860b", "#6a3d9a", "#33a02c", "#e31a1c", "#ff7f00"]
for j, ent in enumerate(a.extra):
    parts = ent.split("=")
    if len(parts) < 2: print(f"WARNING: bad --extra {ent!r}"); continue
    lab, p_ = parts[0], parts[1]; col = parts[2] if len(parts) > 2 else EXC[j % len(EXC)]
    try:
        bw = pyBigWig.open(p_); cc = ch if ch in bw.chroms() else (ch[3:] if ch.startswith("chr") else "chr" + ch)
        v = np.nan_to_num(np.array(bw.stats(cc, s, e, nBins=a.bins, exact=False), dtype=float))
    except Exception as ex: print(f"WARNING: {lab}: {ex} - skipped"); continue
    tracks.append(("extra", lab, col, v))
rna_gene = None
if a.tpm and a.tpm_sample and not any(k in ("rna_plus", "rna_minus") for k, *_ in tracks):
    D = pd.read_csv(a.tpm, sep="\t", index_col=0); ser = D[a.tpm_sample]; ser.index = ser.index.astype(str).str.replace(r"\.\d+$", "", regex=True); ser = ser[~ser.index.duplicated()]
    rna_gene = [(r.start, r.end, float(np.log2(ser.get(r.id, ser.get(r.name, 0.0)) + 1))) for r in G.itertuples()]

nt = len(tracks) + (1 if rna_gene is not None else 0)
fig = plt.figure(figsize=(6.2, 3.3 + 0.62 * nt)); gs = fig.add_gridspec(2 + nt, 1, height_ratios=[3.3, 1.2] + [1] * nt, hspace=.3)
axh = fig.add_subplot(gs[0]); axh.set_xlim(s, e); axh.axis("off")
if a.cool:
    import cooler
    cp = a.cool if "::" in a.cool or not a.cool.endswith(".mcool") else f"{a.cool}::resolutions/{a.resolution}"; c = cooler.Cooler(cp)
    reg = f"{ch}:{s}-{e}"
    if a.raw: M = np.nan_to_num(c.matrix(balance=False).fetch(reg)).astype(float)
    else:
        try: M = np.nan_to_num(c.matrix(balance=True).fetch(reg)) * a.hic_scale
        except Exception: print("WARNING: no balancing weights - using raw counts"); M = np.nan_to_num(c.matrix(balance=False).fetch(reg)).astype(float)
    n = M.shape[0]; bs = c.binsize; off = (s // bs) * bs; I, J = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    X = off + (I + J) / 2 * bs; Y = (J - I) / 2 * bs; Mm = np.ma.masked_where(np.tril(np.ones_like(M), -1) > 0, M)
    vmax = a.hic_vmax or float(np.percentile(M[np.triu_indices(n, 1)], a.hic_pct)) or 1
    pc = axh.pcolormesh(X, Y, Mm, cmap="Reds", vmin=0, vmax=vmax, shading="flat", rasterized=True); axh.set_ylim(0, (e - s) / 2); axh.set_xlim(s, e)
    cax = axh.inset_axes([.93, .55, .025, .4]); cb = fig.colorbar(pc, cax=cax); cb.set_ticks([0, vmax]); cb.set_ticklabels(["0", f"{vmax:.3g}"]); cb.ax.tick_params(labelsize=6, length=2)
    axh.text(s + (e - s) * .03, (e - s) * .4, "Hi-C", fontsize=8)
else: axh.text(.5, .5, "Hi-C (no --cool given)", ha="center", transform=axh.transAxes, fontsize=8, color="#777")
axh.set_title(f"{ch.replace('chr', 'Chr. ')}: {s:,}-{e:,}   ({a.sample})", fontsize=8, pad=3)

axg = fig.add_subplot(gs[1], sharex=axh); axg.set_xlim(s, e); axg.axis("off"); rows_end = []
for r in G.itertuples():
    mid = (max(r.start, s) + min(r.end, e)) / 2; w = len(r.name) * (e - s) * .011; lo = min(max(r.start, s), mid - w / 2); hi = max(min(r.end, e), mid + w / 2)
    row = next((k for k, en in enumerate(rows_end) if lo > en), len(rows_end))
    if row == len(rows_end): rows_end.append(hi)
    else: rows_end[row] = hi
    y = -row; axg.add_patch(Rectangle((max(r.start, s), y - .12), min(r.end, e) - max(r.start, s), .24, color="#1f78b4", lw=0))
    axg.text(mid, y + .2, r.name, fontsize=5, style="italic", ha="center", va="bottom", fontweight="bold" if r.name == a.gene else "normal")
axg.set_ylim(-max(len(rows_end), 1) + .4, .9); axg.text(s - (e - s) * .01, 0, "Genes", fontsize=7, ha="right", va="center")

axs = []
for i, (key, lab, col, v) in enumerate(tracks):
    ax = fig.add_subplot(gs[2 + i], sharex=axh); ax.fill_between(x, v, color=col, lw=0); ax.set_xlim(s, e)
    if key == "rna_minus": lo_, hi_ = -max(float(np.percentile(-v, 99.8)), 1e-9) * 1.05, 0; ax.set_ylim(lo_, hi_); top = f"{-lo_:.2g}"; ty = lo_ * .75
    elif key == "wgbs": ax.set_ylim(0, 1.0); top = "1"; ty = .78
    else: hi_ = max(float(np.percentile(v, 99.8)), 1e-9) * 1.05; ax.set_ylim(0, hi_); top = f"{hi_:.2g}"; ty = hi_ * .8
    ax.set_yticks([]); ax.tick_params(axis="x", bottom=False, labelbottom=False); ax.text(s + (e - s) * .005, ty, top, fontsize=5, va="center")
    for sp in ("top", "right", "bottom"): ax.spines[sp].set_visible(False)
    ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6.5, color=col, labelpad=4); axs.append(ax)
if rna_gene is not None:
    ax = fig.add_subplot(gs[2 + len(tracks)], sharex=axh); ym = max(max(v for *_, v in rna_gene), 1) * 1.1
    for st, en, v in rna_gene: ax.add_patch(Rectangle((max(st, s), 0), min(en, e) - max(st, s), v, color="#cc2222", lw=0))
    ax.set_ylim(0, ym); ax.set_xlim(s, e); ax.set_yticks([]); ax.tick_params(axis="x", bottom=False, labelbottom=False); ax.text(s + (e - s) * .005, ym * .8, f"{ym:.2g}", fontsize=5)
    for sp in ("top", "right", "bottom"): ax.spines[sp].set_visible(False)
    ax.set_ylabel("RNA-seq\n(gene log2 TPM+1)", rotation=0, ha="right", va="center", fontsize=6, color="#cc2222", labelpad=4); axs.append(ax)
last = axs[-1] if axs else axg; last.spines["bottom"].set_visible(True)
tk = np.linspace(s, e, 6); last.tick_params(axis="x", bottom=True, labelbottom=True, labelsize=6); last.set_xticks(tk); last.set_xticklabels([f"{v/1e6:.2f}" for v in tk], fontsize=6); last.set_xlabel("Mb", fontsize=7)
fig.savefig(a.out + ".png", dpi=250, bbox_inches="tight"); fig.savefig(a.out + ".pdf", bbox_inches="tight"); print("wrote", a.out + ".png", "tracks:", [t[1] for t in tracks])
