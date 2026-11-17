#!/usr/bin/env python3
"""Editable PowerPoint for Extended Data Fig. 2 (B-ALL).  Everything is a native shape (nothing is a picture):
  a, c  per-sample PC1 tracks = freeform polygons (red A / purple B), gene-bin band, genotype strips, sample labels, TPM bars, gene track
  b     box plots = rectangles, medians, whiskers, points, brackets, P text
Each panel is a PowerPoint group (right-click > Ungroup to edit pieces).
Input: ed_fig2_pptdata.json written by ed_fig2.py.   Output: ED_Fig2_individual_panels.pptx (slides a, b, c) and ED_Fig2_consolidated.pptx
  python build_ed_fig2_pptx.py --data out/main/ed_fig2/ed_fig2_pptdata.json --outdir out/main/ed_fig2
Needs: python-pptx, numpy, matplotlib (only for tick locations)."""
import argparse, json, os
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn
from lxml import etree
from matplotlib.ticker import MaxNLocator

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--outdir", default="."); A = ap.parse_args(); os.makedirs(A.outdir, exist_ok=True)
D = json.load(open(A.data)); FONT = "Arial"; rgb = lambda h: RGBColor.from_string(h.lstrip("#").upper()); E = lambda inch: int(round(inch * 914400))
RED, PUR, TEAL = "B2182B", "7B6FD0", "19B5B5"

def text(shapes, x, y, w, h, s, size=8, bold=False, align=PP_ALIGN.LEFT, color="000000", anchor=MSO_ANCHOR.TOP, rot=0, italic=False):
    tb = shapes.add_textbox(E(x), E(y), E(w), E(h)); tf = tb.text_frame; tf.word_wrap = False if rot else True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = anchor
    for i, ln in enumerate(str(s).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align; r = p.add_run(); r.text = ln
        r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic; r.font.name = FONT; r.font.color.rgb = rgb(color)
    tb.rotation = rot; return tb
def line(shapes, x1, y1, x2, y2, w=.75, color="000000"):
    c = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, E(x1), E(y1), E(x2), E(y2)); c.line.width = Pt(w); c.line.color.rgb = rgb(color); return c
def rect(shapes, x, y, w, h, fill, lw=0.0, alpha=None, shape=MSO_SHAPE.RECTANGLE, lc="000000"):
    r = shapes.add_shape(shape, E(x), E(y), max(E(w), 1), max(E(h), 1)); r.fill.solid(); r.fill.fore_color.rgb = rgb(fill); r.shadow.inherit = False
    if lw: r.line.color.rgb = rgb(lc); r.line.width = Pt(lw)
    else: r.line.fill.background()
    if alpha is not None:
        clr = r.fill._xPr.find(qn("a:solidFill")).find(qn("a:srgbClr")); al = etree.SubElement(clr, qn("a:alpha")); al.set("val", str(int(alpha * 1000)))
    return r
def poly(shapes, pts, color):
    fb = shapes.build_freeform(int(pts[0][0]), int(pts[0][1]), scale=1.0); fb.add_line_segments([(int(a), int(b)) for a, b in pts[1:]], close=True)
    sh = fb.convert_to_shape(); sh.fill.solid(); sh.fill.fore_color.rgb = rgb(color); sh.line.fill.background(); sh.shadow.inherit = False; return sh
def clip(xs, v, sign):
    out = []
    for i in range(len(xs)):
        if i > 0 and v[i - 1] * v[i] < 0: t = v[i - 1] / (v[i - 1] - v[i]); out.append((xs[i - 1] + t * (xs[i] - xs[i - 1]), 0.0))
        out.append((xs[i], max(v[i], 0.0) if sign > 0 else min(v[i], 0.0)))
    return out

def tracks(slide, T, x, y, w, h, strip, rowh_max=.075, name=None):
    grp = slide.shapes.add_group_shape(); grp.name = name or f"{T['gene']} PC1 tracks"; sh = grp.shapes
    rows = T["rows"]; n = len(rows); head, foot, strip_w, lblw = .62, .45, .2, .42
    rh = min(rowh_max, (h - head - foot) / n); ty = y + head; th = rh * n; tx0 = x + strip_w + .04 + lblw + .05; avail = w - (tx0 - x); tw = avail * .73; ex0 = tx0 + tw + .05; ew = avail * .25
    lo, hi = T["lo"], T["hi"]; X = lambda xc: tx0 + (xc - lo) / (hi - lo) * tw
    rect(sh, X(T["b0"]), ty, X(T["b0"] + T["res"]) - X(T["b0"]), th, "F4A460" if T["ref"] != "B" else "7AAED6", alpha=40)
    fs = max(3.0, min(5.0, rh * 72 * .8))
    for i, r in enumerate(rows):
        base = ty + (i + .5) * rh; xs = r["xs"]; v = r["v"]
        for sign, col in ((1, RED), (-1, PUR)):
            if max(abs(q) for q in v if (q * sign) > 0) if any((q * sign) > 0 for q in v) else 0:
                c = clip(xs, v, sign); pts = [(E(X(xs[0])), E(base))] + [(E(X(a_)), E(base - b_ * rh)) for a_, b_ in c] + [(E(X(xs[-1])), E(base))]; poly(sh, pts, col)
        text(sh, x + strip_w + .04, base - rh / 2, lblw, rh, r["label"], size=fs, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
        if r["tpm"] is not None and T["tpm_max"] > 0: rect(sh, ex0, base - .35 * rh, max(r["tpm"] / (T["tpm_max"] * 1.08) * ew, .002), .7 * rh, TEAL)
    cum = 0
    for g in D["groups"]:
        yt = ty + cum * rh; hh = g["count"] * rh
        if strip:
            rect(sh, x, yt, strip_w, hh, g["color"].lstrip("#").upper()); tb = text(sh, x + strip_w / 2 - hh / 2, yt + hh / 2 - .08, hh, .16, g["short"], size=5, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, rot=270)
        cum += g["count"]
        if cum < n: line(sh, tx0 - .02, ty + cum * rh, ex0 + ew, ty + cum * rh, w=.5)
    line(sh, tx0, ty, tx0, ty + th); line(sh, ex0, ty, ex0, ty + th)
    mx = T["tpm_max"]; ax_end = ex0 + ew; line(sh, ex0, ty + th, ax_end, ty + th)
    for val in sorted({0, round(mx / 2, 1 if mx < 20 else 0), round(mx, 1 if mx < 20 else 0)}):
        xx = ex0 + val / (mx * 1.08) * ew; line(sh, xx, ty + th, xx, ty + th + .03); text(sh, xx - .2, ty + th + .04, .4, .12, f"{val:g}", size=4.5, align=PP_ALIGN.CENTER)
    text(sh, ex0, ty + th + .17, ew, .13, "Expression (TPM)", size=5, align=PP_ALIGN.CENTER)
    text(sh, tx0, y, tw, .18, f"{T['chrom']}: {lo / 1e6:.1f} Mb-{hi / 1e6:.1f} Mb", size=7, align=PP_ALIGN.CENTER)
    for g_ in T["genes"]:
        a_, b_ = max(g_["start"], lo), min(g_["end"], hi); rect(sh, X(a_), y + .42, max(X(b_) - X(a_), .01), .1, "000000" if g_["focus"] else "555555")
        if g_["focus"]: text(sh, X((a_ + b_) / 2) - .5, y + .25, 1.0, .15, g_["name"], size=7, bold=True, italic=True, align=PP_ALIGN.CENTER)
    return grp

def box(slide, bd, x, y, w, h):
    grp = slide.shapes.add_group_shape(); grp.name = f"box {bd['title']}"; sh = grp.shapes
    B, Av = np.array(bd["B"], float), np.array(bd["A"], float); allv = np.concatenate([B, Av]) if len(B) + len(Av) else np.array([0., 1.]); span = (allv.max() - allv.min()) or 1.0; hi_d = allv.max()
    ylo = (min(0, allv.min()) - .05 * span) if allv.min() < 0 else -.03 * span; yhi = hi_d + .28 * span
    axx, axw, axy, axh = x + .62, w - .7, y + .3, h - .3 - .32; Y = lambda v: axy + (yhi - v) / (yhi - ylo) * axh; XP = lambda p: axx + axw * (p + .6) / 2.2; bw = axw * .6 / 2.2
    line(sh, axx, axy, axx, axy + axh); line(sh, axx, axy + axh, axx + axw, axy + axh)
    for t in [t for t in MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(ylo, yhi) if ylo <= t <= yhi]:
        line(sh, axx - .03, Y(t), axx, Y(t)); text(sh, axx - .52, Y(t) - .06, .46, .12, f"{t:g}", size=6, align=PP_ALIGN.RIGHT)
    rng = np.random.default_rng(2)
    for pos, arr, col, lab in ((0, B, "4F7EA8", "B"), (1, Av, "E8876A", "A")):
        cx = XP(pos)
        if len(arr):
            q1, med, q3 = np.percentile(arr, [25, 50, 75]); iq = q3 - q1; lw_ = arr[arr >= q1 - 1.5 * iq].min(); hw_ = arr[arr <= q3 + 1.5 * iq].max()
            line(sh, cx, Y(q3), cx, Y(hw_)); line(sh, cx, Y(q1), cx, Y(lw_)); line(sh, cx - bw * .25, Y(hw_), cx + bw * .25, Y(hw_)); line(sh, cx - bw * .25, Y(lw_), cx + bw * .25, Y(lw_))
            rect(sh, cx - bw / 2, Y(q3), bw, max(Y(q1) - Y(q3), .004), col, lw=.7); line(sh, cx - bw / 2, Y(med), cx + bw / 2, Y(med), w=1.5)
            for v in arr: rect(sh, cx + rng.uniform(-.14, .14) * axw / 2.2 * 1.0 - .018, Y(v) - .018, .036, .036, "000000", shape=MSO_SHAPE.OVAL)
        text(sh, cx - .2, axy + axh + .05, .4, .15, lab, size=8, align=PP_ALIGN.CENTER)
    yb = hi_d + .07 * span; line(sh, XP(0), Y(hi_d + .04 * span), XP(0), Y(yb)); line(sh, XP(1), Y(hi_d + .04 * span), XP(1), Y(yb)); line(sh, XP(0), Y(yb), XP(1), Y(yb))
    p = bd["p"]; ps = "P = n/a" if p is None else (f"P = {p:.1E}" if p < .01 else f"P = {p:.2g}"); text(sh, XP(.5) - .6, Y(yb) - .17, 1.2, .14, ps, size=6.5, align=PP_ALIGN.CENTER)
    text(sh, axx, y + .02, axw, .2, bd["title"], size=9, bold=True, align=PP_ALIGN.CENTER)
    text(sh, x - .04, axy + axh / 2 - .08, .3, .16, bd["ylab"], size=6.5, align=PP_ALIGN.CENTER, rot=270); return grp

def key(slide, x, y):
    sh = slide.shapes
    for i, (col, lab, al) in enumerate(((RED, "PC1 > 0 (A)", None), (PUR, "PC1 < 0 (B)", None), ("F4A460", "gene bin; control = A (A to B in patients)", 40), ("7AAED6", "gene bin; control = B (B to A in patients)", 40))):
        xx = x + [0, 1.4, 2.8, 5.6][i]; rect(sh, xx, y, .22, .12, col, alpha=al); text(sh, xx + .27, y - .01, 3.0, .14, lab, size=7)
def letter(slide, x, y, ch): text(slide.shapes, x, y, .4, .3, ch, size=16, bold=True)
def title(slide, s): text(slide.shapes, .45, .12, 9, .3, s, size=11, bold=True)

genes = D["genes"]; Tk = D["tracks"]; Bx = D["boxes"]
# ---------------- individual panels
prs = Presentation(); prs.slide_width, prs.slide_height = E(13.333), E(7.5); blank = prs.slide_layouts[6]
s = prs.slides.add_slide(blank); title(s, f"Extended Data Fig. 2a | {genes[0]}: PC1 and expression across samples"); letter(s, .3, .45, "a"); tracks(s, Tk[genes[0]], .85, .6, 7.2, 6.55, True); key(s, .9, 7.2)
s = prs.slides.add_slide(blank); title(s, f"Extended Data Fig. 2b | {genes[0]} promoter (TSS ± 1 kb)"); letter(s, .3, .45, "b")
for j, bd in enumerate(Bx): box(s, bd, .9 + (j % 2) * 4.7, .9 + (j // 2) * 3.1, 4.3, 2.9)
text(s.shapes, .9, 7.0, 11.5, .4, D["caption_b"], size=7)
s = prs.slides.add_slide(blank); title(s, "Extended Data Fig. 2c | PC1 and expression across samples"); letter(s, .3, .45, "c")
for j, nm in enumerate(genes[1:4]): tracks(s, Tk[nm], .75 + j * 4.2, .6, 4.1, 6.55, j == 0)
key(s, .9, 7.2); prs.save(f"{A.outdir}/ED_Fig2_individual_panels.pptx")
# ---------------- consolidated (paper layout)
prs = Presentation(); prs.slide_width, prs.slide_height = E(14.5), E(12.9); blank = prs.slide_layouts[6]; s = prs.slides.add_slide(blank)
title(s, "Extended Data Fig. 2 | Examples of genes with A/B compartment switch (B-ALL)")
letter(s, .2, .5, "a"); tracks(s, Tk[genes[0]], .7, .55, 6.9, 6.0, True)
letter(s, 7.8, .5, "b")
for j, bd in enumerate(Bx): box(s, bd, 8.3 + (j % 2) * 3.1, .8 + (j // 2) * 2.85, 2.9, 2.7)
letter(s, .2, 6.75, "c")
for j, nm in enumerate(genes[1:4]): tracks(s, Tk[nm], .7 + j * 4.6, 6.8, 4.4, 5.7, j == 0)
key(s, 1.0, 12.6); prs.save(f"{A.outdir}/ED_Fig2_consolidated.pptx")
nsh = sum(len(g.shapes) for sl in prs.slides for g in sl.shapes if g.shape_type == 6)
print("wrote ED_Fig2_individual_panels.pptx (slides a, b, c) and ED_Fig2_consolidated.pptx;", f"{nsh} editable shapes on the consolidated slide")
