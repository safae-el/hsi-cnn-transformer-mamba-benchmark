"""PRISMA-based study selection workflow for the survey (Fig. 1).

Usage:  python prisma_survey.py            (writes results/figures/prisma_diagram.{pdf,png})
        python prisma_survey.py --out figures
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# ---- counts (edit here only) -------------------------------------------------
N = dict(identified=5737, window=716, dupes=373, screened=4648, excl_screen=4410,
         sought=238, not_retrieved=8, assessed=230, excl_full=132, scopus=98,
         manual=23, eligible=121, consolidated=23, corpus=98)
assert N["identified"] - N["window"] - N["dupes"] == N["screened"]
assert N["screened"] - N["excl_screen"] == N["sought"]
assert N["sought"] - N["not_retrieved"] == N["assessed"]
assert N["assessed"] - N["excl_full"] == N["scopus"]
assert N["scopus"] + N["manual"] == N["eligible"]
assert N["eligible"] - N["consolidated"] == N["corpus"]
f = lambda k: f"{N[k]:,}"

# Full-text exclusion reasons. Set counts to integers (from the screening log) to print
# them; leave as None to print reasons only. If given, counts must sum to excl_full.
EXCL_REASONS = [
    ("Not HSI segmentation / pixel-wise", None),
    ("Medical / non-RS hyperspectral", None),
    ("Hardware or sensor design only", None),
    ("Detection without segmentation", None),
    ("Reconstruction / denoising only", None),
    ("Insufficient methodological detail", None),
]
_c = [c for _, c in EXCL_REASONS]
if any(c is not None for c in _c):
    assert all(c is not None for c in _c), "give a count for every reason or none"
    assert sum(_c) == N["excl_full"], f"reason counts sum to {sum(_c)}, not {N['excl_full']}"

# ---- style ------------------------------------------------------------------
NAVY, INK = "#2c3e6b", "#1a1a1a"
RED_E, RED_F = "#c0392b", "#fbf1f0"
GRN_E, GRN_F = "#4a7c59", "#eef2ee"
GRY_E, GRY_F = "#6b6b6b", "#f5f5f5"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "pdf.fonttype": 42, "ps.fonttype": 42})   # embedded TrueType fonts
FS_B, FS = 15, 13.5

X0, X1, Y0, Y1 = 20, 1915, 100, 1800  # visible canvas in "pixels" (y grows downward)
fig = plt.figure(figsize=((X1 - X0) / 100, (Y1 - Y0) / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.axis("off")

MX0, MX1 = 145, 1265                    # main column
RX0, RX1 = 1308, 1900                   # right column
MC = (MX0 + MX1) / 2

def box(x0, y0, x1, y1, ec=INK, fc="white", ls="-", lw=2.0):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, ec=ec, fc=fc, lw=lw, ls=ls))

def text(x, y, s, bold=False, size=None, color=INK):
    ax.text(x, y, s, ha="center", va="center", color=color,
            fontsize=size or (FS_B if bold else FS), fontweight="bold" if bold else "normal")

def lines(x, y0, y1, items):
    """items: list of (string, bold); spread evenly between y0 and y1."""
    step = (y1 - y0) / (len(items) + 1)
    for i, (s, b) in enumerate(items, 1):
        text(x, y0 + i * step, s, bold=b)

def arrow(p, q, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16,
                                 lw=2.0, color=INK, ls=ls, shrinkA=0, shrinkB=0))

def stage(y0, y1, label):
    ax.add_patch(Rectangle((35, y0), 87, y1 - y0, fc=NAVY, ec="none"))
    ax.text(78.5, (y0 + y1) / 2, label, rotation=90, ha="center", va="center",
            color="white", fontsize=13, fontweight="bold")


# Identification
stage(115, 560, "IDENTIFICATION")
box(MX0, 115, MX1, 270)
text(MC, 192, f"Records identified from Scopus database searches   (n = {f('identified')})", bold=True)
arrow((MC, 277), (MC, 330))
box(MX0, 337, MX1, 559)
lines(MC, 300, 596, [("Records removed before screening:", True),
                     (f"Published in 2025–2026, outside the 2015–2024 screening window   (n = {f('window')})", False),
                     (f"Duplicate records removed   (n = {f('dupes')})", False)])
arrow((MC, 566), (MC, 636))

# Screening
stage(643, 790, "SCREENING")
box(MX0, 643, MX1, 789)
text(MC, 716, f"Records screened by title and abstract   (n = {f('screened')})", bold=True)
box(RX0, 655, RX1, 778, ec=RED_E, fc=RED_F)
text((RX0 + RX1) / 2, 716, f"Records excluded   (n = {f('excl_screen')})")
arrow((MX1 + 8, 716), (RX0 - 8, 716))
arrow((MC, 796), (MC, 866))

# Retrieval
stage(873, 1020, "RETRIEVAL")
box(MX0, 873, MX1, 1019)
text(MC, 946, f"Reports sought for full-text retrieval   (n = {f('sought')})", bold=True)
box(RX0, 885, RX1, 1008, ec=RED_E, fc=RED_F)
text((RX0 + RX1) / 2, 946, f"Reports not retrieved   (n = {f('not_retrieved')})")
arrow((MX1 + 8, 946), (RX0 - 8, 946))
arrow((MC, 1026), (MC, 1096))

# Eligibility
stage(1103, 1250, "ELIGIBILITY")
box(MX0, 1103, MX1, 1249)
text(MC, 1176, f"Full-text reports assessed for eligibility   (n = {f('assessed')})", bold=True)
box(RX0, 1042, RX1, 1366, ec=RED_E, fc=RED_F)
_step = (1388 - 1018) / (len(EXCL_REASONS) + 2)
text((RX0 + RX1) / 2, 1018 + _step, f"Reports excluded   (n = {f('excl_full')})", bold=True)
for i, (reason, cnt) in enumerate(EXCL_REASONS, 2):
    y = 1018 + i * _step
    if cnt is None:
        text((RX0 + RX1) / 2, y, reason)
    else:
        ax.text(RX0 + 30, y, reason, ha="left", va="center", fontsize=FS, color=INK)
        ax.text(RX1 - 30, y, f"{cnt:,}", ha="right", va="center", fontsize=FS, color=INK)
arrow((MX1 + 8, 1176), (RX0 - 8, 1176))
arrow((MC, 1256), (MC, 1326))

# Included: systematic corpus
stage(1333, 1787, "INCLUDED")
box(MX0, 1333, MX1, 1787, ec=GRN_E, fc=GRN_F)
lines(MC, 1288, 1832, [
    (f"Studies selected from Scopus screening   (n = {f('scopus')})", True),
    (f"Supplementary references (foundational and 2025–2026 studies)   (n = {f('manual')})", False),
    (f"→  Eligible corpus after screening   (n = {f('eligible')})", False),
    (f"Omitted during reference consolidation   (n = {f('consolidated')})", False),
    (f"→  Final cited review corpus   (n = {f('corpus')})", True)])

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/figures")
out = ap.parse_args().out; os.makedirs(out, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"prisma_diagram.{ext}"), dpi=300, bbox_inches="tight",
                pad_inches=0.15, facecolor="white")
print("saved", os.path.join(out, "prisma_diagram.{pdf,png}"), "| final cited corpus =", N["corpus"])