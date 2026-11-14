"""Reuse Bernardo's pinned palette and map semantic roles to his native IDs."""
from pathlib import Path
import importlib.util

SOURCE = Path(__file__).resolve().parents[1] / 'vendor/latent-equivariant/code/examples/figure_style.py'
spec = importlib.util.spec_from_file_location('bernardo_figure_style', SOURCE)
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
# render_sphere_flow.NODE and render_o3's explicit_ids assertion.
SPHERE_IDS = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}  # S, A-, C, A+, N
O3_IDS = {'A-': 0, 'A+': 1, 'C': 2, 'O': 3}
SPHERE_COLORS = {k: style.morse(v) for k,v in SPHERE_IDS.items()}
O3_COLORS = {k: style.morse(v) for k,v in O3_IDS.items()}

# .485 of the manuscript's six-inch text width, including cropped margins.
PANEL_WIDTH_BP = .485 * 432
TITLE_SIZE_BP = 10


def size_panel_title(fig, ax):
    """Match the printed column heading size after tight cropping."""
    fig.canvas.draw()
    width = ax.get_tightbbox(fig.canvas.get_renderer()).width / fig.dpi * 72 + 6
    ax.title.set_fontsize(TITLE_SIZE_BP * width / PANEL_WIDTH_BP)


def save_graph(source, path, *, crop=True):
    """Use identical 10 bp headings at the LaTeX panel width."""
    import re
    import subprocess
    import tempfile
    from graphviz import Source
    with tempfile.TemporaryDirectory() as tmp:
        raw, cropped = Path(tmp)/'raw.pdf', Path(tmp)/'cropped.pdf'
        for _ in range(4):
            raw.write_bytes(Source(source).pipe(format='pdf'))
            if crop:
                subprocess.run(['pdfcrop','--margins','3',str(raw),str(cropped)],
                               check=True,capture_output=True)
            width=style.pdf_size_bp(cropped if crop else raw)[0]
            current=float(re.search(r'fontsize=([\d.]+)',source)[1])
            desired=TITLE_SIZE_BP*width/PANEL_WIDTH_BP
            if abs(current-desired)<.005:break
            source=re.sub(r'fontsize=[\d.]+',f'fontsize={desired:.4f}',source,count=1)
    path.with_suffix('.dot').write_text(source)
    for ext in ('svg','png','pdf'):
        path.with_suffix('.'+ext).write_bytes(Source(source).pipe(format=ext))
