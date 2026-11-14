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
    from notebook_support import table
    table(['Quantity','Value'], [
        ('Largest fixed-point defect, one step', f"{row['max_fixed_point_defect']:.7f}"),
        ('Largest fixed-point defect, CMGDB horizon', f"{row['max_cmgdb_horizon_fixed_point_defect']:.7f}"),
        ('Encoder geometry sample count', row['encoder_geometry']['sample_count'])])
