#!/usr/bin/env python3
"""Fig 1 (B-ALL version of Xu et al. 2022 Fig 1), rebuilt with updated data incl. ATAC-seq + CTCF.
Subcommands (1d/1e = compartments_fig1e.py `switch`, unchanged):
  a  assay summary dot-matrix (sample x assay)      --manifest
  b  multi-track example region (Hi-C, ATAC, CTCF, RNA, H3K27ac...) --tracks --region [--cool]
  c  PC1 clustering (top 10% variable bins) + genotype bars + example-region PC1 heatmap
  f  ANOVA (P<0.05) differential-compartment bins clustered by group, marked genes
Inputs: eig dir (<sample>.tsv: chrom,start,end,E1; from compartments_fig1e.py eig), groups (sample,group),
        optional genotype table (sample + 0/1 columns)."""
import argparse, os, re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from scipy.stats import f_oneway

PAL = ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666","#1f78b4","#b2df8a"]
CTRL = "#000000"

EXCL = []
def load_groups(p):
    g = pd.read_csv(p, sep=None, engine="python", header=None, comment="#")
    g = g.iloc[:, :2]; g.columns = ["sample", "group"]
    if g.iloc[0, 0] == "sample": g = g.iloc[1:]
    d = dict(zip(g["sample"].astype(str), g["group"].astype(str)))
    ex = [x for x in EXCL if x]
    miss = [x for x in ex if x not in d]
    if miss: print(f"[exclude] WARNING not found in groups table: {miss}")
    d = {k: v for k, v in d.items() if k not in ex}
    if ex: print(f"[exclude] removed {len(ex) - len(miss)} samples")
    return d

def load_eig(d, groups, res=40000, z=False):
    cols = {}
    for s in groups:
        f = os.path.join(d, s + ".tsv")
        if not os.path.exists(f): continue
        t = pd.read_csv(f, sep="\t")
        cols[s] = t.set_index(["chrom", "start"])["E1"]
    if not cols: sys.exit("no eig files matched groups in " + d)
    M = pd.DataFrame(cols)
    if z:   # per sample, per chromosome z-score: E1 is unit-norm per chromosome, so raw magnitudes are not comparable across chromosomes
        M = M.groupby(level=0).transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    print(f"[eig] {M.shape[1]} samples x {M.shape[0]} bins; missing: "
          f"{[s for s in groups if s not in cols]}")
    return M

def gcolors(groups, samples):
    gl = sorted(set(groups[s] for s in samples) - {"control"})
    cm = {g: PAL[i % len(PAL)] for i, g in enumerate(gl)}; cm["control"] = CTRL
    return cm

def cmd_a(a):
    m = pd.read_csv(a.manifest, sep="\t")          # sample, group, then assay columns (0/1 or path/NA)
    assays = [c for c in m.columns if c not in ("sample", "group")]
    B = m[assays].notna() & (m[assays].astype(str) != "0") & (m[assays].astype(str) != "")
    m = m.assign(_n=B.sum(1)).sort_values(["group", "_n", "sample"], ascending=[True, False, True])
    B = B.loc[m.index]
    fig, ax = plt.subplots(figsize=(0.6 * len(assays) + 3, 0.16 * len(m) + 1.5))
    for j, c in enumerate(assays):
        y = np.where(B[c].values)[0]; ax.scatter(np.full(len(y), j), y, s=22, c="#333")
    ax.set_xticks(range(len(assays))); ax.set_xticklabels(assays, rotation=60, ha="right")
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m["sample"], fontsize=5); ax.invert_yaxis()
    cm = gcolors(dict(zip(m["sample"], m["group"])), m["sample"])
    for l, g in zip(ax.get_yticklabels(), m["group"]): l.set_color(cm.get(g, "k"))
    for j, c in enumerate(assays): ax.text(j, -1.5, int(B[c].sum()), ha="center", fontsize=7)
    ax.set_title("Fig 1a  samples x assays"); fig.tight_layout(); fig.savefig(a.out + "_1a.png", dpi=200)
    B.assign(sample=m["sample"], group=m["group"]).to_csv(a.out + "_1a_table.tsv", sep="\t", index=False)

def cmd_b(a):
    import pyBigWig
    T = pd.read_csv(a.tracks, sep="\t")             # label, path, color, [scale]
    ch, rest = a.region.split(":"); s, e = [int(x.replace(",", "")) for x in rest.split("-")]
    n = len(T) + (1 if a.cool else 0)
    fig, axs = plt.subplots(n, 1, figsize=(9, 0.9 * n + 1), sharex=True,
                            gridspec_kw={"height_ratios": ([4] if a.cool else []) + [1] * len(T)})
    axs = np.atleast_1d(axs); k = 0
    if a.cool:
        import cooler
        c = cooler.Cooler(a.cool); mat = c.matrix(balance=True).fetch(f"{ch}:{s}-{e}")
        mat = np.nan_to_num(mat); r = c.binsize
        axs[0].imshow(np.log1p(mat * 1e3), cmap="Reds", extent=[s, e, e, s], aspect="auto")
        axs[0].set_ylabel("Hi-C", fontsize=7); k = 1
    bins = a.bins
    for i, r_ in T.iterrows():
        bw = pyBigWig.open(r_["path"]); v = np.nan_to_num(np.array(bw.stats(ch, s, e, nBins=bins, exact=False), dtype=float))
        sc = float(r_["scale"]) if "scale" in T.columns and pd.notna(r_.get("scale")) else 1.0
        v = v * sc                                   # depth normalisation factor from table
        x = np.linspace(s, e, bins); ax = axs[k + i]
        ax.fill_between(x, v, color=r_.get("color", "#333") if pd.notna(r_.get("color", np.nan)) else "#333", lw=0)
        ax.set_ylabel(r_["label"], rotation=0, ha="right", va="center", fontsize=7); ax.set_yticks([])
        ax.set_ylim(0, np.percentile(v, 99.5) * 1.1 + 1e-9)
    axs[-1].set_xlim(s, e); axs[-1].ticklabel_format(useOffset=False, style="plain"); axs[-1].set_xlabel(a.region)
    fig.suptitle("Fig 1b  example region", fontsize=9); fig.tight_layout(); fig.savefig(a.out + "_1b.png", dpi=200)

def add_controls(a, groups):
    """extra control samples: --controls file (one sample name per line, or sample<TAB>label); label defaults to 'control'"""
    if not a.controls: return groups
    for l in open(a.controls):
        f = l.rstrip("\n").split("\t")
        if f[0] and not f[0].startswith(("#", "sample")): groups[f[0]] = f[1] if len(f) > 1 else "control"
    return groups

def top_bins(a, groups, M):
    """top-variable bins; --select patients picks them on patient samples only; --center removes the shared A/B landscape
    (subtract each bin's mean over all samples / over patients) so correlations reflect deviations, not the common profile."""
    isc = lambda s: re.search(a.control_regex, s + " " + groups[s], re.I) is not None
    pc = [s for s in M.columns if not isc(s)]
    src = M[pc] if (a.select == "patients" and len(pc) >= 5) else M
    if a.bins_file:     # restrict to a given bin set (e.g. the Fig 1f differential bins) - note: clustering then partly circular
        bf = pd.read_csv(a.bins_file, sep="\t"); top = M.loc[M.index.intersection(pd.MultiIndex.from_frame(bf[["chrom", "start"]]))]
        print(f"[bins_file] using {len(top):,} bins from {a.bins_file}")
    else:
        var = src.var(axis=1); top = M[var >= var.quantile(1 - a.top)]
    if a.center == "all": top = top.sub(top.mean(axis=1), axis=0)
    elif a.center == "patients" and pc: top = top.sub(top[pc].mean(axis=1), axis=0)
    return top

def within_filter(a, groups, M):
    """drop samples whose mean PC1 correlation to their own group is low (robust: < group median - k*MAD, or < --min_in_r)"""
    top = top_bins(a, groups, M)
    C = pd.DataFrame(np.corrcoef(top.values.T), index=top.columns, columns=top.columns)
    isctl = lambda s: re.search(a.control_regex, s + " " + groups[s], re.I) is not None
    lab = pd.Series({s: ("control" if isctl(s) else groups[s]) for s in C.index}); rows = []
    for g in lab.unique():
        m = list(lab.index[lab == g]); o = [s for s in lab.index if lab[s] != g]
        for s in m:
            inn = C.loc[s, [x for x in m if x != s]].mean() if len(m) > 1 else np.nan
            rows.append((s, g, len(m), inn, C.loc[s, o].mean()))
    R = pd.DataFrame(rows, columns=["sample", "group", "n_group", "mean_r_within", "mean_r_other"])
    kk = {}
    for g, d in R.groupby("group"):
        if len(d) < 4: kk.update({s: True for s in d["sample"]}); continue
        med = d.mean_r_within.median(); mad = (d.mean_r_within - med).abs().median() * 1.4826 + 1e-9
        kk.update({s: bool(r >= med - a.mad_k * mad and r >= a.min_in_r) for s, r in zip(d["sample"], d.mean_r_within)})
    R["keep"] = R["sample"].map(kk)
    # label-permutation test: is mean(within-group r) - mean(between-group r) larger than for shuffled group labels?
    rng = np.random.default_rng(1); Cv = C.values; labv = lab.values; ug = [g for g in pd.unique(labv) if (labv == g).sum() >= 3]
    def stat(lv, g):
        m = lv == g; sub = Cv[np.ix_(m, m)]; k = m.sum()
        return (sub.sum() - np.trace(sub)) / (k * (k - 1)) - Cv[np.ix_(m, ~m)].mean()
    rows_ = []
    for g in ug:
        obs = stat(labv, g); nul = np.array([stat(rng.permutation(labv), g) for _ in range(a.perm_c)])
        rows_.append((g, int((labv == g).sum()), round(obs, 3), round(float((nul >= obs).mean()), 3) if a.perm_c else np.nan))
    print("[perm 1c] group, n, (mean within r - mean between r), permutation P:")
    for r_ in rows_: print("   ", *r_)
    pd.DataFrame(rows_, columns=["group", "n", "within_minus_between", "perm_P"]).to_csv(a.out + "_1c_group_separation.tsv", sep="\t", index=False)
    print("[corr] mean r within vs other group (per group):")
    print(R.groupby("group")[["mean_r_within", "mean_r_other"]].mean().round(3).to_string())
    print(f"[filter] keeping {int(R.keep.sum())}/{len(R)}; dropped: {list(R['sample'][~R.keep])}")
    return R

def set_region(a):
    if getattr(a, "gene", None):
        if not a.gtf: sys.exit("--gene needs --gtf")
        gp = genes_from_gtf(a.gtf, [a.gene])
        if a.gene not in gp: sys.exit(f"{a.gene} not found in GTF")
        c_, s_, e_ = gp[a.gene]; a.region = f"{c_}:{max(0, s_ - a.flank)}-{e_ + a.flank}"; a._gene = (a.gene, c_, s_, e_)
        print(f"[region] {a.gene} {c_}:{s_:,}-{e_:,}; track {a.region}")

def prune_samples(a, groups, M):
    """Greedy pruning: repeatedly drop the PATIENT sample whose mean r to its own genotype group is lowest relative to other groups,
    (controls never dropped, every group keeps >=3). Returns kept list + the separation score before/after, with a label-shuffle null
    that applies the SAME pruning to random labels - this shows how much 'grouping' pruning creates by chance."""
    top = top_bins(a, groups, M); S = list(top.columns); C = np.corrcoef(top.values.T)
    isc = lambda x: re.search(a.control_regex, x + " " + groups[x], re.I) is not None
    pat = [i for i, x in enumerate(S) if not isc(x)]
    def run(lab, n):
        keep = list(range(len(S))); 
        def score(idx):
            v = []
            for i in idx:
                if i not in pat: continue
                same = [j for j in idx if j != i and j in pat and lab[j] == lab[i]]; oth = [j for j in idx if j in pat and lab[j] != lab[i]]
                if same and oth: v.append(C[i, same].mean() - C[i, oth].mean())
            return np.array(v)
        for _ in range(n):
            cnt = pd.Series([lab[i] for i in keep if i in pat]).value_counts(); best = None
            for i in keep:
                if i not in pat or cnt[lab[i]] <= 3: continue
                same = [j for j in keep if j != i and j in pat and lab[j] == lab[i]]; oth = [j for j in keep if j in pat and lab[j] != lab[i]]
                sc = C[i, same].mean() - C[i, oth].mean()
                if best is None or sc < best[0]: best = (sc, i)
            if best is None: break
            keep.remove(best[1])
        return keep, score(keep).mean()
    lab0 = {i: groups[S[i]] for i in pat}
    _, before = run(lab0, 0); keep, after = run(lab0, a.prune)
    rng = np.random.default_rng(1); nul = []
    for _ in range(20):
        pl = list(lab0.values()); rng.shuffle(pl); lab = dict(zip(pat, pl)); nul.append(run(lab, a.prune)[1])
    print(f"[prune] removed {len(S) - len(keep)} patient samples: {[S[i] for i in range(len(S)) if i not in keep]}")
    print(f"[prune] within-minus-between r: before {before:.3f} -> after {after:.3f}; same pruning on shuffled labels reaches {np.mean(nul):.3f} +/- {np.std(nul):.3f}")
    return [S[i] for i in keep], before, after, float(np.mean(nul)), float(np.std(nul))

def h_select(V, lab, a):
    """Subtype-UNIQUE bins. For each group g (controls pooled as a group) a bin is kept if
    (1) g vs ALL other samples (other subtypes + controls): Welch t-test, BH-FDR < --fdr_h;
    (2) g differs from EVERY other group (subtype vs subtype and vs control): same sign, |mean diff| > --delta and pairwise Welch P < --pair_p.
    Returns {group: DataFrame(idx, dir, q, margin)}."""
    from scipy.stats import ttest_ind
    lv = np.asarray(lab); gs = [g for g in pd.unique(lv) if (lv == g).sum() >= a.min_n]
    idx = {g: np.where(lv == g)[0] for g in gs}; mu = {g: V[:, idx[g]].mean(1) for g in gs}; out = {}
    for g in gs:
        rest = np.where(lv != g)[0]; oth = [o for o in gs if o != g]
        p = np.nan_to_num(ttest_ind(V[:, idx[g]], V[:, rest], axis=1, equal_var=False)[1], nan=1.0); q = bh(p)
        d = np.stack([mu[g] - mu[o] for o in oth], 1)
        pp = np.stack([np.nan_to_num(ttest_ind(V[:, idx[g]], V[:, idx[o]], axis=1, equal_var=False)[1], nan=1.0) for o in oth], 1)
        up = (d > a.delta).all(1) & (pp < a.pair_p).all(1); dn = (d < -a.delta).all(1) & (pp < a.pair_p).all(1)
        ok = ((p if a.raw_p else q) < a.fdr_h) & (up | dn); w = np.where(ok)[0]
        out[g] = pd.DataFrame({"idx": w, "dir": np.where(up[w], "up", "down"), "q": q[w], "margin": np.abs(d[w]).min(1)})
    return out

def cmd_h(a):
    """Subtype-unique compartment bins (subtype vs every other subtype AND control), heatmap, per-group tables, and a
    cross-validation check (bins chosen on half the samples, held-out samples classified) with a shuffled-label null."""
    groups = add_controls(a, load_groups(a.groups)); M = load_eig(a.eig_dir, groups, z=a.zscore).dropna(thresh=int(0.8 * len(groups)))
    M = M.apply(lambda c: c.fillna(c.median())); S = list(M.columns)
    isc = lambda x: re.search(a.control_regex, x + " " + groups[x], re.I) is not None
    if a.patients_only: M = M[[x for x in S if not isc(x)]]; S = list(M.columns)
    lab = pd.Series({x: ("control" if isc(x) else groups[x]) for x in S}); V = M.values
    sel = h_select(V, lab, a); gs = list(sel)
    print(f"[h] groups tested (n>={a.min_n}): { {g: int((lab == g).sum()) for g in gs} }")
    print("[h] unique bins per group:", {g: (int((d.dir == 'up').sum()), int((d.dir == 'down').sum())) for g, d in sel.items()}, "(up, down)")
    # census genes overlapping bins
    gp = {}
    if a.gtf and a.cgc:
        cg = pd.read_csv(a.cgc); gc = [c for c in cg.columns if c.lower().startswith("gene symbol")][0]; gp = genes_from_gtf(a.gtf, set(cg[gc].astype(str)))
    ch = M.index.get_level_values(0).values; st = M.index.get_level_values(1).values
    rows = []
    for g, d in sel.items():
        for _, r in d.iterrows():
            i = int(r.idx); gg = [n for n, (c, s0, e0) in gp.items() if c == ch[i] and s0 < st[i] + a.res and e0 >= st[i]]
            rows.append((g, ch[i], st[i], r.dir, r.q, r.margin, ",".join(sorted(gg))))
    T = pd.DataFrame(rows, columns=["group", "chrom", "start", "direction", "fdr_vs_rest", "min_mean_diff_vs_other_groups", "census_genes"])
    T.to_csv(a.out + "_1h_bins.tsv", sep="\t", index=False)
    if len(T): print(T[T.census_genes != ""].groupby("group").census_genes.apply(lambda x: ",".join(x)).to_string())
    # heatmap
    order = [g for g in gs if g != "control"] + (["control"] if "control" in gs else []); samp = [x for g in order for x in lab.index[lab == g]] + [x for x in lab.index if lab[x] not in order]
    rb = []; rg = []
    for g in order:
        d = sel[g].sort_values("margin", ascending=False).head(a.max_show).sort_values("idx"); rb += list(d.idx); rg += [g] * len(d)
    if rb:
        X = M.iloc[rb][samp]; X = X.sub(M.iloc[rb].mean(axis=1), axis=0); cm = gcolors(lab.to_dict(), samp)
        fig = plt.figure(figsize=(12, 9)); gs_ = fig.add_gridspec(2, 2, width_ratios=[.3, 10], height_ratios=[.4, 10], wspace=.02, hspace=.02)
        ax = fig.add_subplot(gs_[1, 1]); im = ax.imshow(X.values, aspect="auto", cmap="RdBu_r", vmin=-a.vmax, vmax=a.vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for k in np.where(np.array(rg)[1:] != np.array(rg)[:-1])[0]: ax.axhline(k + .5, color="k", lw=.6)
        for k in np.where(lab[samp].values[1:] != lab[samp].values[:-1])[0]: ax.axvline(k + .5, color="k", lw=.6)
        at = fig.add_subplot(gs_[0, 1]); at.imshow(np.array([[matplotlib.colors.to_rgb(cm[lab[x]]) for x in samp]]), aspect="auto"); at.axis("off")
        al = fig.add_subplot(gs_[1, 0]); al.imshow(np.array([[matplotlib.colors.to_rgb(cm[g])] for g in rg]), aspect="auto"); al.axis("off")
        for g in order:
            w = np.where(np.array(rg) == g)[0]
            if len(w): ax.text(-.5 - 0.01 * len(samp), w.mean(), f"{g} (n={len(w)})", ha="right", va="center", fontsize=7, color=cm[g] if g != "control" else "k")
        for g in order:
            w = np.where(lab[samp].values == g)[0]; at.text(w.mean(), -.6, g, ha="left", va="bottom", rotation=45, fontsize=6, color=cm[g] if g != "control" else "k")
        fig.colorbar(im, ax=ax, fraction=.02, pad=.01, label="PC1 (E1) minus bin mean")
        PT = "raw P" if a.raw_p else "FDR"
        fig.suptitle(f"Fig 1h  subtype-unique compartment bins (vs every other subtype and control; {PT}<{a.fdr_h}, pair P<{a.pair_p}, |diff|>{a.delta})", fontsize=9)
        fig.savefig(a.out + "_1h.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    # cross-validation with shuffled-label null
    rng = np.random.default_rng(1); labv = lab.values; big = [g for g in gs if (labv == g).sum() >= 4]; keep = np.where(np.isin(labv, big))[0]
    def cv(shuf):
        tr, te = [], []
        for g in big:
            w = rng.permutation(np.where(labv == g)[0]); h = int(np.ceil(len(w) / 2)); tr += list(w[:h]); te += list(w[h:])
        tr = np.array(tr); te = np.array(te); l_tr = np.array(labv[tr], dtype=object)
        if shuf: rng.shuffle(l_tr)
        ss = h_select(V[:, tr], l_tr, a); b = sorted(set(np.concatenate([d.idx.values for d in ss.values()]))) if ss else []
        if len(b) < 3: return 1 / len(big), 0   # nothing selected -> chance
        mu0 = V[b][:, tr].mean(1); cen = {g: V[b][:, tr[l_tr == g]].mean(1) - mu0 for g in big if (l_tr == g).any()}
        hit = 0
        for j in te:
            v = V[b, j] - mu0; ok = (max(cen, key=lambda g: np.corrcoef(v, cen[g])[0, 1] if np.std(cen[g]) > 0 else -9) == labv[j]); hit += ok
            if not shuf: REC.setdefault(labv[j], []).append(ok)
        return hit / len(te), len(b)
    REC = {}
    R = [cv(False) for _ in range(a.cv)]; N = [cv(True) for _ in range(a.cv)]
    f = lambda L: (np.nanmean([x[0] for x in L]), np.nanstd([x[0] for x in L]), np.mean([x[1] for x in L]))
    (m1, s1, b1), (m0, s0, b0) = f(R), f(N)
    print(f"[h] CV held-out subtype accuracy ({len(big)} groups with n>=4, chance~{1/max(len(big),1):.2f}): real labels {m1:.2f}+/-{s1:.2f} ({b1:.0f} bins/split); shuffled labels {m0:.2f}+/-{s0:.2f} ({b0:.0f} bins/split)")
    print("[h] held-out recall per group (real labels):", {g: round(float(np.mean(v)), 2) for g, v in REC.items()})
    pt = [x for g, v in REC.items() if g != "control" for x in v]
    if pt: print(f"[h] patient-subtype-only held-out accuracy: {np.mean(pt):.2f} (chance ~{1/max(len(big),1):.2f})")

def cmd_c(a):
    groups = add_controls(a, load_groups(a.groups))
    M = load_eig(a.eig_dir, groups, z=a.zscore).dropna(thresh=int(0.8 * len(groups)))
    M = M.apply(lambda c: c.fillna(c.median()))
    nc = sum(1 for s in M.columns if re.search(a.control_regex, s + " " + groups[s], re.I))
    print(f"[controls] {nc} control samples loaded (paper-style 1c/1f need them; use --controls <file> to add)")
    set_region(a)
    run_c(a, groups, M, a.out + "_1c", "all samples")
    R = within_filter(a, groups, M); R.to_csv(a.out + "_1c_within_group_corr.tsv", sep="\t", index=False)
    run_c(a, groups, M[list(R["sample"][R.keep])], a.out + "_1c_filtered", "after removing samples poorly correlated within group")
    if a.prune:
        kp, b, af, nm, ns = prune_samples(a, groups, M)
        run_c(a, groups, M[kp], a.out + "_1c_pruned", f"pruned {a.prune} patients; within-between r {b:.2f}->{af:.2f} (shuffled-label null {nm:.2f}+/-{ns:.2f})")

def run_c(a, groups, M, out, note, Cpre=None):
    """Paper-style 1c: sample x sample PC1 correlation (top-variable bins), dendrogram, group colour bar,
    text columns (group / other alterations), and per-sample PC1 bars for an example region."""
    if Cpre is not None:                     # correlation computed elsewhere (e.g. profiles vs group-consensus PC1)
        C = Cpre.values; S0 = list(Cpre.columns); nb = Cpre.attrs.get("desc", "precomputed profiles")
    else:
        top = top_bins(a, groups, M); C = np.corrcoef(top.values.T); S0 = list(top.columns); nb = f"top {int(a.top*100)}% variable bins ({len(top):,})"
    Z = linkage(C, "complete", metric="euclidean") if a.paper_cluster else linkage(1 - C[np.triu_indices(len(S0), 1)], "average"); o = leaves_list(Z)
    S = [S0[k] for k in o]; C = C[np.ix_(o, o)]; n = len(S)
    isctl = lambda s: re.search(a.control_regex, s + " " + groups[s], re.I) is not None
    lab = {s: ("control" if isctl(s) else groups[s]) for s in S}; cm = gcolors(lab, S)
    ann = {}
    if a.annot:
        t_ = pd.read_csv(a.annot, sep="\t"); t_.iloc[:, 0] = t_.iloc[:, 0].astype(str)
        for _, r in t_.iterrows(): ann[r.iloc[0]] = " ; ".join(str(x) for x in r.iloc[1:] if pd.notna(x))
    fig = plt.figure(figsize=(15, 0.2 * n + 3))
    gs = fig.add_gridspec(2, 6, width_ratios=[1.5, 0.25, 8, 1.6, 3.2, 3.0], height_ratios=[.5, 10], wspace=.04, hspace=.12)
    axd = fig.add_subplot(gs[1, 0]); dendrogram(Z, ax=axd, orientation="left", no_labels=True, color_threshold=0, above_threshold_color="#333")
    axd.invert_yaxis(); axd.axis("off")
    axc = fig.add_subplot(gs[1, 1]); axc.imshow([[matplotlib.colors.to_rgb(cm[lab[s]])] for s in S], aspect="auto"); axc.axis("off")
    ax = fig.add_subplot(gs[1, 2]); offd = C[np.triu_indices(n, 1)]; vlo = a.vmin_c if a.vmin_c is not None else float(np.floor(np.percentile(offd, 1) * 10) / 10)
    im = ax.imshow(C, aspect="auto", cmap="RdBu_r", vmin=vlo, vmax=1, interpolation="nearest")
    ax.set_yticks(range(n)); ax.set_yticklabels(S, fontsize=5); ax.yaxis.tick_left()
    for l, s in zip(ax.get_yticklabels(), S): l.set_color(cm[lab[s]])
    ax.set_xticks(range(n)); ax.set_xticklabels(S, rotation=90, fontsize=4)
    # unsupervised clusters (dendrogram cut at k) boxed like the paper; agreement with genotype reported
    cl = fcluster(Z, a.k, "maxclust")[o]
    k0 = 0
    while k0 < n:
        e = k0
        while e + 1 < n and cl[e + 1] == cl[k0]: e += 1
        if e > k0: ax.add_patch(plt.Rectangle((k0 - .5, k0 - .5), e - k0 + 1, e - k0 + 1, fill=False, ec="k", lw=1.0))
        k0 = e + 1
    ct = pd.crosstab(pd.Series(cl, index=S, name="cluster"), pd.Series({s: lab[s] for s in S}, name="group"))
    def _ari(x, y):
        from math import comb
        tab = pd.crosstab(x, y).values; sa = sum(comb(int(v), 2) for v in tab.sum(1)); sb = sum(comb(int(v), 2) for v in tab.sum(0))
        sij = sum(comb(int(v), 2) for v in tab.ravel()); ex = sa * sb / comb(n, 2); mx = (sa + sb) / 2
        return (sij - ex) / (mx - ex) if mx != ex else 0.0
    gl_ = np.array([lab[s] for s in S]); ari = _ari(cl, gl_); ctrl_ = np.array([lab[s] == "control" for s in S]); ari2 = _ari(cl[~ctrl_], gl_[~ctrl_]) if (~ctrl_).sum() > 2 else float("nan")
    print(f"[unsupervised 1c: {note}] colour range {vlo:.1f}..1; k={a.k}; ARI(clusters, genotype incl. controls) = {ari:.3f}; patients only = {ari2:.3f}  (0 = chance, 1 = identical)")
    print(ct.to_string()); ct.to_csv(out + "_cluster_by_group.tsv", sep="\t")
    cax = fig.add_subplot(gs[0, 2]); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cax.set_title("PC1 correlation", fontsize=7); cax.xaxis.set_ticks_position("top"); cax.tick_params(labelsize=6)
    for col, ttl, fn in ((3, "Consensus group", lambda s: lab[s]), (4, "Other alterations", lambda s: ann.get(s, ""))):
        at = fig.add_subplot(gs[1, col]); at.set_xlim(0, 1); at.set_ylim(n - .5, -.5); at.axis("off"); at.set_title(ttl, fontsize=7)
        for r, s in enumerate(S): at.text(0, r, fn(s), fontsize=5, va="center", color=cm[lab[s]] if col == 3 else "k")
    if a.region:
        ch, rest = a.region.split(":"); s0, e0 = [int(x.replace(",", "")) for x in rest.split("-")]
        sel = [(c, b) for c, b in M.index if c == ch and s0 <= b < e0]
        if sel:
            R = M.loc[sel, S]; x = np.arange(len(sel)); ar = fig.add_subplot(gs[1, 5], sharey=ax)
            for r, s in enumerate(S):
                v = np.clip(R[s].values / a.vmax, -1, 1) * .45
                ar.fill_between(x, r, r - v, where=v >= 0, color="#c0392b", lw=0, step="mid")
                ar.fill_between(x, r, r - v, where=v < 0, color="#2c3e9e", lw=0, step="mid")
            ar.set_xlim(0, len(sel)); ar.set_yticks([]); ar.set_xticks([0, len(sel)]); ar.set_xticklabels([f"{s0/1e6:g}", f"{e0/1e6:g} Mb"], fontsize=6)
            ar.set_title(f"{ch}: PC1" if not getattr(a, "_gene", None) else f"{a._gene[0]} locus ({ch}) PC1", fontsize=7, pad=16)
            if getattr(a, "_gene", None):
                gx = np.searchsorted([b for _, b in sel], a._gene[2]); pass
                ar.text(gx, -1.2, a._gene[0], ha="center", va="bottom", fontsize=7, style="italic")
    fig.suptitle(f"Fig 1c  PC1 correlation, {nb}, {n} samples - {note}", fontsize=9, y=.995)
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(C, index=S, columns=S).to_csv(out + "_corr.tsv", sep="\t")
    pd.Series(S).to_csv(out + "_order.txt", index=False, header=False)

def ctrl_sublabel(s, g):
    m = re.match(r"CTRL_(.+?)_pooled", s)
    if m: return "ctrl " + m.group(1)
    if re.search(r"preB", s, re.I) and s.startswith("YDP15"): return "ctrl preB (ours)"
    if "HSPC" in s.upper(): return "ctrl HSPC"
    return "ctrl " + g

def cmd_g(a):
    """Group-consensus PC1 correlation. (1) group x group correlation of mean PC1 (top variable bins);
    (2) each sample vs each group consensus (own group leave-one-out); prints how often a sample is closest to its own group."""
    groups = add_controls(a, load_groups(a.groups)); M = load_eig(a.eig_dir, groups, z=a.zscore).dropna(thresh=int(0.8 * len(groups)))
    M = M.apply(lambda c: c.fillna(c.median()))
    isctl = lambda s: re.search(a.control_regex, s + " " + groups[s], re.I) is not None
    lab = pd.Series({s: (ctrl_sublabel(s, "control") if (isctl(s) and a.split_controls) else ("control" if isctl(s) else groups[s])) for s in M.columns})
    cnt = lab.value_counts(); print("[groups]", dict(cnt))
    gl = [g for g in cnt.index if cnt[g] >= a.min_n_g or g.startswith("ctrl") or g == "control"]
    gl = sorted([g for g in gl if not (g.startswith("ctrl") or g == "control")]) + sorted([g for g in gl if g.startswith("ctrl") or g == "control"])
    CONS = pd.DataFrame({g: M.loc[:, lab.index[lab == g]].mean(axis=1) for g in gl})
    if a.merged_eig_dir:      # group PC1 recomputed from merged Hi-C of each genotype group (compartments_eig_groups.slurm)
        used = []
        for g in gl:
            f = os.path.join(a.merged_eig_dir, f"GRP_{g}.tsv")
            if os.path.exists(f):
                e_ = pd.read_csv(f, sep="\t").set_index(["chrom", "start"])["E1"]; CONS[g] = e_.reindex(CONS.index).fillna(CONS[g]); used.append(g)
        print(f"[merged PC1] used for: {used}; mean-of-samples kept for the rest (controls)")
    top = CONS[CONS.var(axis=1) >= CONS.var(axis=1).quantile(1 - a.top)]
    C = np.corrcoef(top.values.T); o = leaves_list(linkage(C, "complete", metric="euclidean")); gs_ = [gl[k] for k in o]; C = C[np.ix_(o, o)]
    off = C[np.triu_indices(len(gs_), 1)]; vmin = a.vmin if a.vmin is not None else float(np.floor(off.min() * 20) / 20)
    cm = gcolors({g: ("control" if g.startswith("ctrl") else g) for g in gs_}, gs_); cm = {g: ("#000000" if g.startswith("ctrl") else cm.get(g, "#888")) for g in gs_}
    fig, ax = plt.subplots(figsize=(0.55 * len(gs_) + 4, 0.55 * len(gs_) + 3))
    im = ax.imshow(C, cmap="RdBu_r", vmin=vmin, vmax=1)
    ax.set_xticks(range(len(gs_))); ax.set_xticklabels([f"{g} (n={cnt[g]})" for g in gs_], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(gs_))); ax.set_yticklabels([f"{g} (n={cnt[g]})" for g in gs_], fontsize=7)
    for l, g in zip(ax.get_yticklabels(), gs_): l.set_color(cm[g])
    for i in range(len(gs_)):
        for j in range(len(gs_)): ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=5, color="k" if C[i, j] < .93 else "w")
    fig.colorbar(im, ax=ax, shrink=.6, label="PC1 correlation (group consensus)")
    ax.set_title(f"Group-consensus PC1 correlation, top {int(a.top*100)}% variable bins ({len(top):,})", fontsize=9)
    fig.savefig(a.out + "_1g_groups.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(C, index=gs_, columns=gs_).to_csv(a.out + "_1g_groups_corr.tsv", sep="\t")
    if a.merged_eig_dir:
        print('[note] sample-vs-group view skipped with merged PC1 (a sample is part of its own group, so it would be circular)'); return
    # sample vs group consensus, own group leave-one-out
    S = [s for g in gs_ for s in lab.index[lab == g]]; R = np.full((len(S), len(gs_)), np.nan); T = top.index
    sums = {g: M.loc[T, lab.index[lab == g]].sum(axis=1) for g in gs_}
    for i, s in enumerate(S):
        x = M.loc[T, s]
        for j, g in enumerate(gs_):
            if lab[s] == g:
                if cnt[g] < 2: continue
                v = (sums[g] - x) / (cnt[g] - 1)
            else: v = sums[g] / cnt[g]
            R[i, j] = np.corrcoef(x, v)[0, 1]
    best = [gs_[int(np.nanargmax(R[i]))] for i in range(len(S))]; own = np.array([b == lab[s] for b, s in zip(best, S)])
    print("[closest consensus] fraction of samples whose most-correlated group consensus is their own (leave-one-out):")
    for g in gs_:
        m = np.array([lab[s] == g for s in S]); print(f"    {g:28s} n={m.sum():3d}  own-closest={own[m].mean():.2f}  (chance ~ {1/len(gs_):.2f})")
    print(f"    ALL samples: {own.mean():.2f}")
    fig, ax = plt.subplots(figsize=(0.5 * len(gs_) + 3, 0.16 * len(S) + 2))
    im = ax.imshow(R, aspect="auto", cmap="RdBu_r", vmin=np.nanmin(R) if a.vmin is None else a.vmin, vmax=1)
    ax.set_xticks(range(len(gs_))); ax.set_xticklabels(gs_, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(S))); ax.set_yticklabels(S, fontsize=4)
    for l, s in zip(ax.get_yticklabels(), S): l.set_color(cm[lab[s]])
    k = 0
    for j, g in enumerate(gs_):
        n_ = int((lab[S] == g).sum()); ax.add_patch(plt.Rectangle((j - .5, k - .5), 1, n_, fill=False, ec="k", lw=.8)); k += n_
    fig.colorbar(im, ax=ax, shrink=.5, label="r vs group consensus (own group: leave-one-out)")
    fig.suptitle("Fig 1c (group consensus): sample vs group PC1", fontsize=9); fig.savefig(a.out + "_1g_samples.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(R, index=S, columns=gs_).to_csv(a.out + "_1g_samples_corr.tsv", sep="\t")
    # ---- unsupervised view: describe every sample by its correlation with each group-consensus PC1
    #      (own group leave-one-out, controls merged), then cluster samples with NO labels (complete linkage / Euclidean)
    lab2 = lab.map(lambda g: "control" if g.startswith("ctrl") else g); cnt2 = lab2.value_counts(); basis = [g for g in cnt2.index if cnt2[g] >= 2]
    sums2 = {g: M.loc[T, lab2.index[lab2 == g]].sum(axis=1) for g in basis}; smp = list(M.columns)
    F = np.zeros((len(smp), len(basis)))
    for i, s in enumerate(smp):
        x = M.loc[T, s]
        for j, g in enumerate(basis):
            v = (sums2[g] - x) / (cnt2[g] - 1) if lab2[s] == g else sums2[g] / cnt2[g]
            F[i, j] = np.corrcoef(x, v)[0, 1]
    Cp = pd.DataFrame(np.corrcoef(F), index=smp, columns=smp); Cp.attrs["desc"] = f"profiles vs {len(basis)} group-consensus PC1 (leave-one-out)"
    pd.DataFrame(F, index=smp, columns=basis).to_csv(a.out + "_1g_unsup_features.tsv", sep="\t")
    set_region(a); run_c(a, groups, M, a.out + "_1g_unsup", "unsupervised clustering on group-consensus profiles", Cpre=Cp)

def genes_from_gtf(gtf, names):
    out = {}; names = set(names)
    import gzip
    for l in (gzip.open(gtf, 'rt') if gtf.endswith('.gz') else open(gtf)):
        if l.startswith("#"): continue
        f = l.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != "gene": continue
        m = re.search(r'gene_name "([^"]+)"', f[8])
        if m and m.group(1) in names: out[m.group(1)] = (f[0], int(f[3]), int(f[4]))
    return out

DEFAULT_GENES = ("PAX5 IKZF1 CDKN2A CDKN2B EBF1 ETV6 RUNX1 BTG1 CRLF2 JAK2 JAK1 TP53 TP63 MYC BCL2 CD200 BTLA NRAS KRAS "
                 "FLT3 GATA3 MEIS1 MECOM IGF1R FGF13 ENPP2 DST ETS1 KMT2A TCF3 PBX1 BCL6 LMO2 CD19 CD79A VPREB1 IGLL1 "
                 "RAG1 DNTT SOX4 LEF1 FOXO1 NOTCH1 PTEN RB1 SETD2 CREBBP NR3C1 TBL1XR1 ERG").split()

def spread(ys, gap, lo, hi):
    """label y-positions: sorted, >= gap apart, kept inside [lo, hi]"""
    o = np.argsort(ys); y = np.clip(np.array(ys, float)[o], lo, hi)
    for k in range(1, len(y)): y[k] = max(y[k], y[k-1] + gap)
    if len(y) and y[-1] > hi:
        y[-1] = hi
        for k in range(len(y) - 2, -1, -1): y[k] = min(y[k], y[k+1] - gap)
    out = np.empty_like(y); out[o] = y; return out

def bh(p):
    """Benjamini-Hochberg adjusted p-values"""
    n = len(p); o = np.argsort(p); r = np.empty(n); r[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(r, 0, 1)

def cmd_f(a):
    """Fig 1f: ANOVA across PATIENT genotype groups only (selection), controls shown as extra columns (not used for selection).
    Rows ordered by hierarchical clustering of group means; marked genes limited to --max_genes, inside axis."""
    groups = add_controls(a, load_groups(a.groups)); M = load_eig(a.eig_dir, groups, z=a.zscore).dropna(thresh=int(0.8 * len(groups)))
    M = M.apply(lambda c: c.fillna(c.median()))
    isctl = lambda s: re.search(a.control_regex, s + " " + groups[s], re.I) is not None
    lab = pd.Series({s: ("control" if isctl(s) else groups[s]) for s in M.columns})
    cnt = lab.value_counts(); pat = [g for g in cnt.index if g != "control" and cnt[g] >= a.min_n]
    print("[groups]", dict(cnt), "| ANOVA groups:", pat)
    P = [s for s in M.columns if lab[s] in pat]; Cn = [s for s in M.columns if lab[s] == "control"]
    # Paper (Xu 2022 Methods): one-way ANOVA per bin across ALL groups incl. HSPC/controls (samples = replicates), P<0.05,
    # then keep bins with a sign switch between group means and >=2-fold absolute-value difference -> clustered hierarchically.
    A_cols = P + (Cn if (a.anova_controls and Cn) else []); A_groups = pat + (["control"] if (a.anova_controls and Cn) else [])
    idx = {g: np.where(lab[A_cols].values == g)[0] for g in A_groups}; XP = M[A_cols].values
    p = np.array([f_oneway(*[x[i] for i in idx.values()]).pvalue for x in XP])
    gm = np.column_stack([XP[:, i].mean(1) for i in idx.values()]); delta = gm.max(1) - gm.min(1)
    q = bh(np.nan_to_num(p, nan=1))
    sel = (q < a.p) if a.fdr else (p < a.p); n_anova = int(sel.sum())
    if a.paper_filter:
        switch = (gm.max(1) > 0) & (gm.min(1) < 0)                                   # A<->B between group means
        ab = np.abs(gm); fold = ab.max(1) / np.maximum(ab.min(1), 1e-12) >= a.fold   # literal reading of ">=2-fold absolute-value difference"
        sel &= switch & fold; print(f"[paper filter] ANOVA P<{a.p}: {n_anova:,} bins -> sign switch: {int((sel_ := (p < a.p) & switch).sum()):,} -> and fold>={a.fold}: {int(sel.sum()):,}")
    sel &= delta >= a.min_delta
    if a.top_n and sel.sum() > a.top_n: sel &= p <= np.sort(p[sel])[a.top_n - 1]
    sig = np.where(sel)[0]; print(f"[ANOVA] {len(sig)}/{len(p)} bins selected ({'FDR' if a.fdr else 'P'}<{a.p}; groups: {A_groups})")
    # --- diagnostics: is the selection above chance? (shuffle patient group labels) ---
    rng = np.random.default_rng(0); lblP = lab[A_cols].values; nulls = []
    for _ in range(a.perm):
        sh = rng.permutation(lblP); ix2 = [np.where(sh == g)[0] for g in A_groups]
        pn = np.array([f_oneway(*[x[i] for i in ix2]).pvalue for x in XP[:: max(1, len(XP) // 5000)]]); nulls.append((pn < a.p).mean())
    if nulls:
        fr = float(np.mean(nulls)); obs = (p < a.p).mean()
        print(f"[perm] null fraction of bins P<{a.p}: {fr:.3f} (shuffled labels, {a.perm}x); observed {obs:.3f}; est. false-discovery share among selected ~ {min(1, fr/max(obs,1e-9)):.0%}")
    if Cn:
        pm = M[P].values.mean(1); cmn = M[Cn].values.mean(1); d_pc = pm - cmn
        print(f"[patients vs control] bins with |mean patient - mean control| >= 0.01: {(abs(d_pc)>=.01).sum():,}; >= 0.02: {(abs(d_pc)>=.02).sum():,}; >= 0.03: {(abs(d_pc)>=.03).sum():,}; corr(patient mean, control mean) = {np.corrcoef(pm, cmn)[0,1]:.3f}")
        print(f"[group-mean range across patient groups] median {np.median(delta):.4f}; 95th pct {np.percentile(delta,95):.4f}; selected-bins median {np.median(delta[sig]) if len(sig) else float('nan'):.4f}")
    D = M.iloc[sig]; gmS = gm[sig]
    o = leaves_list(linkage(D[A_cols].values, "ward")) if a.cluster_rows == "samples" else leaves_list(linkage(gmS, "ward")); D = D.iloc[o]
    sord = [s for g in pat for s in np.array(P)[idx[g]]] + Cn; gorder = pat + (["control"] if Cn else [])
    cnts = [len(idx[g]) for g in pat] + ([len(Cn)] if Cn else []); D = D[sord]
    cm = gcolors({s: lab[s] for s in sord}, sord); n = len(D)
    vmax, cbl = a.vmax, "PC1"
    if a.center != "none":                                # show deviation from a baseline so shared A/B signal drops out
        base = D[Cn].mean(axis=1) if (a.center == "control" and Cn) else D.mean(axis=1)
        D = D.sub(base, axis=0); vmax = a.vmax_center; cbl = "PC1 - " + ("control mean" if (a.center == "control" and Cn) else "row mean")
    fig = plt.figure(figsize=(8.5, 9.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[10, 2.2], height_ratios=[.35, 10], hspace=.01, wspace=.02)
    axb = fig.add_subplot(gs[0, 0]); axb.imshow([[matplotlib.colors.to_rgb(cm[lab[s]]) for s in sord]], aspect="auto"); axb.axis("off")
    pos = np.cumsum([0] + cnts) - .5
    for g, a0, a1 in zip(gorder, pos[:-1], pos[1:]): axb.text((a0 + a1) / 2, -.7, g, rotation=45, ha="left", va="bottom", fontsize=6, color=cm[g])
    ax = fig.add_subplot(gs[1, 0]); im = ax.imshow(D.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks([]); ax.set_xticks(range(len(sord))); ax.set_xticklabels(sord, rotation=90, fontsize=4)
    for x in pos[1:-1]: ax.axvline(x, color="w", lw=.8)
    ax.set_ylabel(f"n = {n:,} bins"); axl = fig.add_subplot(gs[1, 1], sharey=ax); axl.axis("off")
    genes = [l.strip() for l in open(a.genes) if l.strip()] if a.genes else list(DEFAULT_GENES); leuk = set(); cg_full = None
    if a.cgc:
        cg = pd.read_csv(a.cgc); cg_full = cg.copy(); gc = [c for c in cg.columns if c.lower().startswith("gene symbol")][0]
        tc = [c for c in cg.columns if "tumour types" in c.lower()]
        if a.cgc_regex and tc: cg = cg[cg[tc].fillna("").astype(str).apply(lambda r: " ".join(r.values), axis=1).str.contains(a.cgc_regex, case=False, regex=True)]
        def _ball(r):   # token-based: B-ALL / pre B-ALL / ALL / AL count; T-ALL and ETP ALL do not
            toks = [x.strip() for c in r.values for x in re.split(r"[,;]", str(c))]
            return any(re.fullmatch(r"(pre[- ]?)?B[- ]?ALL|ALL|AL|Ph\+? ?(B-)?ALL", x, re.I) for x in toks)
        if tc:
            lk = cg[tc].fillna("").apply(_ball, axis=1) if a.ball_strict else cg[tc].fillna("").astype(str).apply(lambda r: " ".join(r.values), axis=1).str.contains(a.cgc_priority_regex, case=False, regex=True)
            leuk = set(cg.loc[lk, gc].astype(str))
        genes = [g for g in dict.fromkeys(cg[gc].astype(str))] if a.cgc_only else genes + [g for g in cg[gc].astype(str) if g not in genes]; print(f"[cgc] {len(cg)} census genes after filter '{a.cgc_regex}'; {len(leuk)} leukaemia/lymphoma-related (priority)")
    if a.gtf:
        gp = genes_from_gtf(a.gtf, genes); ii = {k: r for r, k in enumerate(D.index)}; found = {}; miss = []
        for g in genes:                                   # list order = priority
            if g not in gp: continue
            c, s_, e_ = gp[g]; ks = [ii[(c, b)] for b in range((s_ // a.res) * a.res, e_ + 1, a.res) if (c, b) in ii]
            (found.__setitem__(g, ks[0]) if ks else miss.append(g))
        print(f"[genes] in a significant bin ({len(found)}): {list(found)}; not in a significant bin: {miss}; not in GTF: {[g for g in genes if g not in gp]}")
        dsel = pd.Series(delta[sig][o], index=D.index).reset_index(drop=True).values       # effect size per displayed row
        tier = lambda g: 0 if (not a.cgc_only) and (g in DEFAULT_GENES or (a.genes and g in genes)) else (1 if g in leuk else 2)
        order = sorted(found, key=lambda g: (tier(g), -dsel[found[g]]))                   # B-ALL list > leukaemia/lymphoma census > rest; then largest effect
        if a.priority_only: order = [g for g in order if tier(g) <= 1]
        found = {g: found[g] for g in order[:a.max_genes]}
        print("[genes] marked:", list(found))
        if found:
            ys = [found[g] for g in found]; ty = spread(ys, n * .03, 0, n - 1); axl.set_xlim(0, 1); axl.set_ylim(n - .5, -.5)
            for (g, y0), y1 in zip(found.items(), ty):
                axl.plot([0, .25, .45], [y0, y1, y1], color="k", lw=.5); axl.text(.48, y1, g, fontsize=6, va="center", style="italic")
            mk = pd.DataFrame({"gene": list(found), "tier": [("B-ALL priority list" if ((not a.cgc_only) and (g in DEFAULT_GENES or (a.genes and g in genes))) else ("census, ALL-annotated" if g in leuk else "census, other")) for g in found], "display_row": [found[g] for g in found], "bin_group_range": [dsel[found[g]] for g in found]})
            if cg_full is not None:
                ann = cg_full.set_index(gc)[[c for c in cg_full.columns if c.startswith("Tumour Types") or c == "Role in Cancer"]]
                mk = mk.join(ann[~ann.index.duplicated()], on="gene")
            mk.to_csv(a.out + "_1f_genes_marked.tsv", sep="\t", index=False)
    cax = fig.add_axes([.06, .93, .015, .05]); fig.colorbar(im, cax=cax); cax.set_title(cbl, fontsize=6); cax.tick_params(labelsize=5)
    ttl = (f"Fig 1f  one-way ANOVA across {len(A_groups)} groups ({'incl. controls' if 'control' in A_groups else 'patients only; controls shown, not used'}), "
           f"{'FDR' if a.fdr else 'P'}<{a.p}" + (f", A/B sign switch, >={a.fold:g}-fold" if a.paper_filter else "") + f": {n:,} bins")
    fig.suptitle(ttl, fontsize=8, x=.5, y=.995)
    fig.savefig(a.out + "_1f.png", dpi=200, bbox_inches="tight")
    D.to_csv(a.out + "_1f_matrix.tsv.gz", sep="\t")
    pd.DataFrame({"chrom": [i[0] for i in M.index[sig]], "start": [i[1] for i in M.index[sig]], "p": p[sig], "q": q[sig], "range": delta[sig]}).to_csv(a.out + "_1f_bins.tsv", sep="\t", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for n in "abcfgh":
        s = sp.add_parser(n); s.add_argument("--out", required=True)
    s = sp.choices["a"]; s.add_argument("--manifest", required=True)
    s = sp.choices["b"]; s.add_argument("--tracks", required=True); s.add_argument("--region", required=True); s.add_argument("--cool"); s.add_argument("--bins", type=int, default=600)
    for n in "cfg":
        s = sp.choices[n]; s.add_argument("--eig_dir", required=True); s.add_argument("--groups", required=True); s.add_argument("--zscore", action="store_true", help="per-sample per-chromosome z-score of E1 (comparable effect sizes)")
    s = sp.choices["c"]; s.add_argument("--prune", type=int, default=0, help="greedily drop N patient samples that least agree with their genotype group (reports shuffled-label null)"); s.add_argument("--top", type=float, default=0.10); s.add_argument("--controls", help="file listing control sample names (must have eig files)"); s.add_argument("--mad_k", type=float, default=2.0); s.add_argument("--min_in_r", type=float, default=0.0)
    s.add_argument("--select", choices=["all", "patients"], default="all", help="which samples define the top-variable bins"); s.add_argument("--center", choices=["none", "all", "patients"], default="none", help="subtract per-bin mean before correlating"); s.add_argument("--k", type=int, default=6, help="dendrogram clusters to box / compare with genotype"); s.add_argument("--vmin_c", type=float, default=None, help="colour min (default: auto from 1st percentile of correlations)"); s.add_argument("--perm_c", type=int, default=1000, help="label permutations for the group-separation test (0=skip)"); s.add_argument("--paper_cluster", action=argparse.BooleanOptionalAction, default=True, help="paper: complete linkage, Euclidean distance on the sample correlation matrix"); s.add_argument("--annot", help="tsv: sample, then text column(s) e.g. genotype string"); s.add_argument("--region", default="chr10:12000000-18000000"); s.add_argument("--gene", help="centre the right-hand PC1 track on this gene (needs --gtf), e.g. IKZF1"); s.add_argument("--gtf"); s.add_argument("--flank", type=int, default=3000000); s.add_argument("--bins_file", help="tsv with chrom,start (e.g. out/fig1f_main_1f_bins.tsv): correlate on these bins only")
    s.add_argument("--control_regex", default=r"ctrl|control|healthy|normal|preB|HSPC|PBMC|pro-?B|pre-?B"); s.add_argument("--vmax", type=float, default=0.04)
    s = sp.choices["g"]; s.add_argument("--top", type=float, default=0.10); s.add_argument("--annot"); s.add_argument("--region", default="chr10:12000000-18000000"); s.add_argument("--gene"); s.add_argument("--gtf"); s.add_argument("--flank", type=int, default=3000000); s.add_argument("--k", type=int, default=6); s.add_argument("--vmin_c", type=float, default=None); s.add_argument("--vmax", type=float, default=0.04); s.add_argument("--paper_cluster", action=argparse.BooleanOptionalAction, default=True); s.add_argument("--controls"); s.add_argument("--control_regex", default=r"ctrl|control|healthy|normal|preB|HSPC|PBMC|pro-?B|pre-?B"); s.add_argument("--split_controls", action=argparse.BooleanOptionalAction, default=True); s.add_argument("--min_n_g", type=int, default=2); s.add_argument("--merged_eig_dir", help="dir with GRP_<group>.tsv eig files from merged group Hi-C"); s.add_argument("--vmin", type=float, default=None)
    s = sp.choices["f"]; s.add_argument("--controls"); s.add_argument("--p", type=float, default=0.05); s.add_argument("--min_n", type=int, default=3)
    s.add_argument("--genes", help="one gene symbol per line (default: built-in B-ALL/COSMIC list)"); s.add_argument("--gtf"); s.add_argument("--res", type=int, default=40000)
    s.add_argument("--control_regex", default=r"ctrl|control|healthy|normal|preB|HSPC|PBMC|pro-?B|pre-?B"); s.add_argument("--vmax", type=float, default=0.04)
    s.add_argument("--fdr", action="store_true", help="BH-FDR<p instead of raw P<p"); s.add_argument("--min_delta", type=float, default=0.0, help="min range of group-mean E1"); s.add_argument("--top_n", type=int, default=0); s.add_argument("--anova_controls", action=argparse.BooleanOptionalAction, default=True, help="include controls as a group in the ANOVA (paper: HSPC included)"); s.add_argument("--paper_filter", action=argparse.BooleanOptionalAction, default=True, help="paper: keep bins with A<->B sign switch between group means and >=fold abs difference"); s.add_argument("--fold", type=float, default=2.0); s.add_argument("--cluster_rows", choices=["groups", "samples"], default="samples"); s.add_argument("--cgc", help="COSMIC Cancer Gene Census csv (Gene Symbol col); --cgc_regex filters tumour types"); s.add_argument("--cgc_regex", default=""); s.add_argument("--ball_strict", action=argparse.BooleanOptionalAction, default=True, help="priority = census tumour-type tokens B-ALL/pre B-ALL/ALL/AL only (T-ALL and ETP ALL excluded); --no-ball_strict uses --cgc_priority_regex"); s.add_argument("--priority_only", action="store_true", help="label only ALL/AL-annotated census genes (and the built-in list unless --cgc_only)"); s.add_argument("--cgc_only", action="store_true", help="label ONLY COSMIC census genes (ALL-annotated first), ignore the built-in B-ALL list"); s.add_argument("--cgc_priority_regex", default=r"\bALL\b|\bAL\b|lymphoblastic", help="census genes whose tumour types match (default: ALL / lymphoblastic) are labelled first");
    s.add_argument("--perm", type=int, default=10, help="label-shuffle null repeats (0=skip)"); s.add_argument("--center", choices=["none", "control", "mean"], default="none"); s.add_argument("--vmax_center", type=float, default=0.02); s.add_argument("--max_genes", type=int, default=20)
    s = sp.choices["h"]; s.add_argument("--eig_dir", required=True); s.add_argument("--groups", required=True); s.add_argument("--zscore", action="store_true"); s.add_argument("--controls"); s.add_argument("--gtf"); s.add_argument("--cgc")
    s.add_argument("--min_n", type=int, default=3); s.add_argument("--fdr_h", type=float, default=0.05, help="threshold for the one-vs-rest test"); s.add_argument("--raw_p", action="store_true", help="apply --fdr_h to the raw P instead of BH-FDR"); s.add_argument("--pair_p", type=float, default=0.05); s.add_argument("--delta", type=float, default=0.005, help="min |mean E1 difference| to every other group")
    s.add_argument("--max_show", type=int, default=300); s.add_argument("--vmax", type=float, default=0.02); s.add_argument("--cv", type=int, default=20); s.add_argument("--patients_only", action="store_true", help="drop controls: pure subtype-vs-subtype"); s.add_argument("--res", type=int, default=40000)
    s.add_argument("--control_regex", default=r"ctrl|control|healthy|normal|preB|HSPC|PBMC|pro-?B|pre-?B")
    for n in "cfgh": sp.choices[n].add_argument("--exclude", default="", help="comma-separated exact sample names to drop")
    a = ap.parse_args()
    if getattr(a, "exclude", ""): EXCL.extend([x.strip() for x in a.exclude.split(",")])
    {"a": cmd_a, "b": cmd_b, "c": cmd_c, "f": cmd_f, "g": cmd_g, "h": cmd_h}[a.cmd](a)
