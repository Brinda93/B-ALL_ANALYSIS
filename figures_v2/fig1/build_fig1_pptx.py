#!/usr/bin/env python3
"""Build editable PowerPoint files for Fig 1 (B-ALL version of Xu et al. 2022).

Writes two files in --outdir:
  Fig1_individual_panels.pptx  one slide per panel / panel set (16:9)
  Fig1_consolidated.pptx       one portrait slide with all panels arranged a-h

What is editable:
  1d  native PowerPoint stacked-column chart (right-click > Edit Data)
  1e  box plots drawn from native shapes (boxes, whiskers, medians, brackets, P values, axis, labels), one group per panel
  1a/1b  placeholders (data not built yet)
  1c, 1f, 1g, 1h  heat maps are raster images (PowerPoint cannot draw a 58x58 or 7,000-row heat map as shapes); titles/letters are text.

Inputs (all produced by fig1_panels.py and compartments_fig1e.py):
  --de   folder with fig1e_boxstats.json + switch_counts_per_patient.tsv   (compartments_fig1e.py --outdir)
  --c --f --g --h --g2 --a --b   PNG paths (missing files are skipped and shown as placeholders)
Usage:
  python build_fig1_pptx.py --de out/main/fig1de --c out/main/fig1c_final_1c_filtered.png --f out/main/fig1f_main_1f.png \
      --g out/main/fig1g_main_1g_groups.png --g2 out/supplementary/fig1g_main_1g_samples.png --h out/archive_1h/fig1h_raw_1h.png --outdir out/main
Needs: pip install --user python-pptx pillow
"""
import argparse, json, os
import numpy as np, pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from matplotlib.ticker import MaxNLocator

ap = argparse.ArgumentParser()
ap.add_argument("--de", required=True); ap.add_argument("--outdir", default=".")
for k in ("a", "b", "c", "f", "g", "g2", "h"): ap.add_argument("--" + k)
A = ap.parse_args(); os.makedirs(A.outdir, exist_ok=True)
FONT = "Arial"
rgb = lambda h: RGBColor.from_string(h.lstrip("#").upper())

# ------------------------------------------------------------------ helpers
def text(shapes, x, y, w, h, s, size=9, bold=False, align=PP_ALIGN.LEFT, color="000000", anchor=MSO_ANCHOR.TOP, rot=0, italic=False):
    tb = shapes.add_textbox(Emu(int(x)), Emu(int(y)), Emu(int(w)), Emu(int(h))); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = anchor
    for i, line in enumerate(str(s).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align; r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic; r.font.name = FONT; r.font.color.rgb = rgb(color)
    tb.rotation = rot; return tb

def line(shapes, x1, y1, x2, y2, w=0.75, color="000000"):
    c = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(int(x1)), Emu(int(y1)), Emu(int(x2)), Emu(int(y2))); c.line.width = Pt(w); c.line.color.rgb = rgb(color); return c

def rect(shapes, x, y, w, h, fill, lw=0.75):
    r = shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(int(x)), Emu(int(y)), Emu(int(w)), Emu(int(h)))
    r.fill.solid(); r.fill.fore_color.rgb = rgb(fill); r.line.color.rgb = rgb("000000"); r.line.width = Pt(lw); r.shadow.inherit = False; return r

def letter(slide, x, y, ch):
    text(slide.shapes, x, y, Inches(.35), Inches(.3), ch, size=16, bold=True)

def placeholder(slide, x, y, w, h, label):
    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(int(x)), Emu(int(y)), Emu(int(w)), Emu(int(h))); r.fill.background(); r.line.color.rgb = rgb("999999"); r.line.dash_style = 4; r.shadow.inherit = False
    tf = r.text_frame; tf.text = label
    for p in tf.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        for rn in p.runs: rn.font.size = Pt(10); rn.font.color.rgb = rgb("777777"); rn.font.name = FONT

def picture(slide, path, x, y, w, h, label):
    """fit image inside the box, keep aspect, centred; placeholder if the file is missing"""
    if not path or not os.path.exists(path): placeholder(slide, x, y, w, h, f"{label}\n(file not found)"); return
    iw, ih = Image.open(path).size; s = min(w / iw, h / ih); pw, ph = iw * s, ih * s
    slide.shapes.add_picture(path, Emu(int(x + (w - pw) / 2)), Emu(int(y)), Emu(int(pw)), Emu(int(ph)))

# ------------------------------------------------------------------ 1e: native-shape box plot
def boxpanel(slide, P, x, y, w, h, fs=7):
    """draw one box-plot panel (dict from fig1e_boxstats.json) as a group of native shapes inside box (x, y, w, h)"""
    grp = slide.shapes.add_group_shape(); sh = grp.shapes
    L, R, T, B = .13 * w, .02 * w, .13 * h, .24 * h; px, py, pw, ph = x + L, y + T, w - L - R, h - T - B
    lo, hi = P["ylim"]; Y = lambda v: py + ph * (1 - (v - lo) / (hi - lo)); n = len(P["boxes"])
    line(sh, px, py, px, py + ph); line(sh, px, py + ph, px + pw, py + ph)
    for t in MaxNLocator(5).tick_values(lo, hi):
        if lo <= t <= hi:
            line(sh, px - Inches(.04), Y(t), px, Y(t)); text(sh, x, Y(t) - Pt(fs) * .55, L - Inches(.07), Pt(fs) * 1.2, f"{t:g}", size=fs, align=PP_ALIGN.RIGHT)
    text(sh, x + Inches(.09) - Inches(.6), py + ph / 2 - Inches(.09), Inches(1.2), Inches(.18), P["ylabel"], size=fs, align=PP_ALIGN.CENTER, rot=270, anchor=MSO_ANCHOR.MIDDLE)
    for i, b in enumerate(P["boxes"]):
        cx = px + pw * (i + .5) / n; bw = pw * .5 / n
        line(sh, cx, Y(b["hi"]), cx, Y(b["q3"])); line(sh, cx, Y(b["q1"]), cx, Y(b["lo"]))
        line(sh, cx - bw * .25, Y(b["hi"]), cx + bw * .25, Y(b["hi"])); line(sh, cx - bw * .25, Y(b["lo"]), cx + bw * .25, Y(b["lo"]))
        rect(sh, cx - bw / 2, Y(b["q3"]), bw, max(Y(b["q1"]) - Y(b["q3"]), 1), b["color"]); line(sh, cx - bw / 2, Y(b["med"]), cx + bw / 2, Y(b["med"]), w=1.5)
        text(sh, cx - pw / n / 2, py + ph + Inches(.05), pw / n, B - Inches(.05), b["label"], size=fs, align=PP_ALIGN.CENTER)
    yl = min(b["lo"] for b in P["boxes"]); yh = max(b["hi"] for b in P["boxes"]); rg = max(yh - yl, 1e-6)
    for k, c in enumerate(P["comps"], start=1):
        yb = yh + rg * (.06 + .14 * (k - 1)); x1 = px + pw * (c["a"] + .5) / n; x2 = px + pw * (c["b"] + .5) / n; t = rg * .02
        line(sh, x1, Y(yb - t), x1, Y(yb)); line(sh, x1, Y(yb), x2, Y(yb)); line(sh, x2, Y(yb), x2, Y(yb - t))
        text(sh, min(x1, x2), Y(yb) - Pt(fs + 2) * 1.15, abs(x2 - x1), Pt(fs + 2) * 1.2, f"P = {c['p']:.2e}  {c['star']}", size=fs, align=PP_ALIGN.CENTER)
    text(sh, x, y, w, T - Inches(.03), P["title"], size=fs, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.BOTTOM)
    return grp

def legend_for(slide, x, y, panels, fs=8):
    """colour legend (control + patient classes) from the box colours of the first panel"""
    seen = []; [seen.append((b["label"].split(" (")[0], b["color"])) for b in panels[0]["boxes"]]; cx = x
    for lab, col in seen:
        rect(slide.shapes, cx, y + Inches(.02), Inches(.18), Inches(.12), col); text(slide.shapes, cx + Inches(.23), y, Inches(1.2), Inches(.16), lab, size=fs); cx += Inches(1.35)

# ------------------------------------------------------------------ 1d: native chart
def chart_1d(slide, x, y, w, h, fs=7):
    f = os.path.join(A.de, "switch_counts_per_patient.tsv")
    if not os.path.exists(f): placeholder(slide, x, y, w, h, "1d\n(switch_counts_per_patient.tsv not found)"); return
    F = pd.read_csv(f, sep="\t").sort_values("pct_switch"); cd = CategoryChartData(); cd.categories = list(F["sample"])
    for c in ("stable_A", "stable_B", "A_to_B", "B_to_A"): cd.add_series(c.replace("_", " "), [int(v) for v in F[c]])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_STACKED, Emu(int(x)), Emu(int(y)), Emu(int(w)), Emu(int(h)), cd); ch = gf.chart
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(fs + 1); ch.legend.font.name = FONT
    for s_, col in zip(ch.plots[0].series, ("3182BD", "FD8D3C", "31A354", "DE2D26")): s_.format.fill.solid(); s_.format.fill.fore_color.rgb = rgb(col)
    ch.plots[0].gap_width = 25; ch.category_axis.tick_labels.font.size = Pt(max(fs - 3, 4)); ch.value_axis.tick_labels.font.size = Pt(fs)
    ch.value_axis.has_major_gridlines = False; ch.value_axis.has_title = True; ch.value_axis.axis_title.text_frame.text = "40-kb bins"
    ch.value_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(fs); ch.has_title = False

BJ = json.load(open(os.path.join(A.de, "fig1e_boxstats.json"))) if os.path.exists(os.path.join(A.de, "fig1e_boxstats.json")) else []
def panels(trait, name=None): return [p for p in BJ if p["trait"] == trait and (name is None or p["title"].startswith(name))]
ALLP = [p for p in panels("genotype") if p["title"].startswith("All B-ALL")]

# ------------------------------------------------------------------ individual slides (16:9)
pr = Presentation(); pr.slide_width = Inches(13.333); pr.slide_height = Inches(7.5); blank = pr.slide_layouts[6]
def new_slide(title):
    s = pr.slides.add_slide(blank); text(s.shapes, Inches(.4), Inches(.25), Inches(12.5), Inches(.4), title, size=16, bold=True); return s
s = new_slide("Fig 1a  Sample x assay overview"); placeholder(s, Inches(.6), Inches(.9), Inches(12), Inches(6.2), "1a - to be built (sample x assay dot matrix)")
s = new_slide("Fig 1b  Example region: Hi-C, compartment PC1, ATAC, CTCF, RNA"); placeholder(s, Inches(.6), Inches(.9), Inches(12), Inches(6.2), "1b - to be built (multi-track example region)")
s = new_slide("Fig 1c  PC1 correlation between samples (top 10% variable bins) with IKZF1 locus PC1"); picture(s, A.c, Inches(.6), Inches(.9), Inches(12), Inches(6.3), "1c")
s = new_slide("Fig 1d  Compartment switches per patient vs control consensus (editable chart)"); chart_1d(s, Inches(.6), Inches(.9), Inches(12), Inches(6.2), fs=9)
for trait, tt in (("sex", "sex"), ("ancestry", "ancestry")):
    ps = panels(trait)
    if ps:
        s = new_slide(f"Fig 1e  Expression of genes in recurrent compartment-switch bins, by {tt} (editable shapes)"); legend_for(s, Inches(.6), Inches(.75), ps)
        for k, P in enumerate(ps[:2]): boxpanel(s, P, Inches(.6 + 6.2 * k), Inches(1.0), Inches(5.9), Inches(5.9), fs=10)
gp = panels("genotype")
for i in range(0, len(gp), 4):
    s = new_slide("Fig 1e  Expression of genes in recurrent compartment-switch bins, by genotype (editable shapes)"); legend_for(s, Inches(.6), Inches(.75), gp)
    for k, P in enumerate(gp[i:i + 4]): boxpanel(s, P, Inches(.6 + 6.2 * (k % 2)), Inches(1.0 + 3.1 * (k // 2)), Inches(5.9), Inches(2.9), fs=8)
s = new_slide("Fig 1f  One-way ANOVA bins (A/B sign switch, >=2-fold) across genotype groups and controls"); picture(s, A.f, Inches(.6), Inches(.9), Inches(12), Inches(6.4), "1f")
s = new_slide("Fig 1g  Group-consensus PC1 correlation"); picture(s, A.g, Inches(.6), Inches(.9), Inches(12), Inches(6.4), "1g")
if A.g2: s = new_slide("Fig 1g (cont.)  Sample vs group-consensus PC1"); picture(s, A.g2, Inches(.6), Inches(.9), Inches(12), Inches(6.4), "1g samples")
s = new_slide("Fig 1h  Subtype-unique compartment bins"); picture(s, A.h, Inches(.6), Inches(.9), Inches(12), Inches(6.4), "1h")
pr.save(os.path.join(A.outdir, "Fig1_individual_panels.pptx"))

# ------------------------------------------------------------------ consolidated slide (portrait 12 x 16 in)
W, H = 12.0, 16.0; pr = Presentation(); pr.slide_width = Inches(W); pr.slide_height = Inches(H); s = pr.slides.add_slide(pr.slide_layouts[6]); M = .3
def box(col, row_y, wd, ht, ch, x0): letter(s, Inches(x0), Inches(row_y), ch); return Inches(x0 + .35), Inches(row_y + .05), Inches(wd - .35), Inches(ht - .05)
# row 1: a, b
x, y, w, h = box(0, M, 5.5, 2.6, "a", M); placeholder(s, x, y, w, h, "a  sample x assay\n(to be built)")
x, y, w, h = box(0, M, 5.6, 2.6, "b", M + 5.8); placeholder(s, x, y, w, h, "b  example region tracks\n(to be built)")
# row 2: c (left), d (right) + e All B-ALL below d
r2 = M + 2.8
x, y, w, h = box(0, r2, 6.4, 5.4, "c", M); picture(s, A.c, x, y, w, h, "c")
x, y, w, h = box(0, r2, 5.3, 2.6, "d", M + 6.5); chart_1d(s, x, y, w, h, fs=6)
r3 = r2 + 5.5
# row 3: e (all B-ALL, sex, ancestry) - 6 small panels
letter(s, Inches(M), Inches(r3), "e")
sp = panels("sex")[:2]; an = panels("ancestry")[:2]; allp = ALLP[:2]
groups = [allp, sp, an]; cw = (W - 2 * M - .35) / 3
for gi, g in enumerate(groups):
    for k, P in enumerate(g): boxpanel(s, P, Inches(M + .35 + gi * cw + k * cw / 2), Inches(r3 + .05), Inches(cw / 2 - .05), Inches(2.5), fs=5)
if not BJ: placeholder(s, Inches(M + .35), Inches(r3), Inches(W - 2 * M - .35), Inches(2.5), "e (fig1e_boxstats.json not found)")
r4 = r3 + 2.7
# row 4: f, g, h
x, y, w, h = box(0, r4, 4.8, 4.9, "f", M); picture(s, A.f, x, y, w, h, "f")
x, y, w, h = box(0, r4, 3.5, 4.9, "g", M + 4.85); picture(s, A.g, x, y, w, h, "g")
x, y, w, h = box(0, r4, 3.5, 4.9, "h", M + 8.4); picture(s, A.h, x, y, w, h, "h")
pr.save(os.path.join(A.outdir, "Fig1_consolidated.pptx")); print("wrote", A.outdir)
