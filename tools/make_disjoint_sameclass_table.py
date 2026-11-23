"""Random-vs-disjoint table with Indian Pines scored on the same classes in both splits.

analyze.py writes T_disjoint.tex with the random column scored on all 16 Indian Pines classes.
This script replaces the Indian Pines rows so that both splits are scored on the classes present
in each disjoint test set (T_ip_sameclasses.csv), recomputes the drops, and re-runs the paired
Wilcoxon tests with Holm correction for those rows. Output: T_disjoint_sameclass.tex.

  python tools/make_disjoint_sameclass_table.py [results/revision/analysis]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision"))
argv, sys.argv = sys.argv, [sys.argv[0]]
import analyze as A  # noqa: E402

out = Path(argv[1]) if len(argv) > 1 else ROOT / "results" / "revision" / "analysis"
t = pd.read_csv(out / "T_ip_sameclasses.csv")
P = pd.DataFrame({"dataset": "indian_pines", "setting": "random_same", "model": t.model, "seed": t.seed,
                  "OA": t.random_sameclasses_OA, "AA": t.random_sameclasses_AA,
                  "mIoU": t.random_sameclasses_mIoU})
P["Kappa"] = np.nan
P["MacroF1"] = np.nan
A.METRICS = ["OA", "AA", "mIoU"]
tests = A.pairwise(P)
cells = {met: A.table_cells(P, tests, "random_same", "indian_pines", met) for met in ["OA", "mIoU"]}

lines, out_lines, in_ip = (out / "T_disjoint.tex").read_text().split("\n"), [], False
for line in lines:
    if "Indian Pines" in line:
        in_ip = True
    elif line.startswith(r"\midrule") or line.startswith(r"\bottomrule"):
        in_ip = False
    m = re.match(r"^(.*?) & (HybridSN|SpectralFormer|S2Mamba) & (.*) \\\\$", line)
    if in_ip and m:
        mo, parts = m.group(2), [p.strip() for p in m.group(3).split("&")]
        g = t[t.model == mo]
        parts[0] = cells["OA"][mo]
        parts[2] = f"${100 * (g.random_sameclasses_OA.mean() - g.disjoint_OA.mean()):.1f}$"
        parts[3] = cells["mIoU"][mo]
        parts[5] = f"${100 * (g.random_sameclasses_mIoU.mean() - g.disjoint_mIoU.mean()):.1f}$"
        line = f"{m.group(1)} & {mo} & " + " & ".join(parts) + r" \\"
    out_lines.append(line)
s = "\n".join(out_lines).replace(
    r" On Indian Pines, classes absent from a disjoint test set are excluded from mIoU.}",
    r" On Indian Pines, 12--14 of the 16 classes occur in each disjoint test set; both splits are scored on "
    r"these classes for the same seed.}")
(out / "T_disjoint_sameclass.tex").write_text(s)
print("written", out / "T_disjoint_sameclass.tex")
print(tests[["metric", "model_a", "model_b", "p_holm", "significant"]].to_string(index=False))
