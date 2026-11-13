"""Compact reporting of fixed-point commutation tests; no new training."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def show_conjugacy_check(fit):
    from IPython.display import Markdown,display
    rows=json.loads((ROOT/'build/conjugacy-check/results.json').read_text())
    sha=hashlib.sha256((fit/'models.npz').read_bytes()).hexdigest()
    row=next((r for r in rows if r['checkpoint_sha256']==sha),None)
    if row is None:
        display(Markdown('Conjugacy check has not been run for this checkpoint.'));return
    display(Markdown(f"**Approximate dynamics agreement:** at physical quotient fixed points, the largest one-step defect $\\|f(E(x_*))-E(x_*)\\|$ is **{row['max_fixed_point_defect']:.5g}** (sample interval {row['time_step']:.5g}). The learned lifts’ verified local indices match the analytical indices. Transfer to the physical ODE requires suitable encoder coordinates and a neighborhood-wide error bound small enough to preserve isolation along a continuation; the sampled residual does not establish these conditions. The restored `build/conjugacy-check/claim-scope.md` gives the check details."))
