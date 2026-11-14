"""Notebook setup, native image display, and pinned analytical replay."""
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def workspace():
    candidates = [Path.cwd(), *Path.cwd().parents]
    root = next((p for p in candidates if (p / 'upstream.json').is_file()), None)
    if root is None:
        root = next((p / 'paper-notebooks' for p in candidates
                     if (p / 'paper-notebooks/upstream.json').is_file()), None)
    if root is None:
        raise FileNotFoundError('Open the notebook from paper-notebooks.')
    from replay_bundle import restore
    restore(root)
    pin = json.loads((root / 'upstream.json').read_text())
    repo = root / pin['checkout']
    for relative, expected in pin['files'].items():
        if hashlib.sha256((repo / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Upstream source differs from the pinned revision: {relative}')
    examples = repo / pin['examples']
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(examples))
    for key, suffix in [('MPLCONFIGDIR', 'matplotlib'), ('XDG_CACHE_HOME', 'cache')]:
        folder = root / 'build' / suffix
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    from render import verify_data
    from importlib.metadata import version
    if version('CMGDB') != '1.5.3+fork.7':
        raise RuntimeError('This replay requires CMGDB 1.5.3+fork.7; use paper-notebooks/.venv.')
    data = examples / 'data'
    verify_data(data)
    print(f"Pinned dependencies {pin['revision'][:7]} and numerical input checksums verified.")
    return root, examples, data


def new_run(root, example):
    parent = root / 'build/notebooks' / example / 'replays'
    parent.mkdir(parents=True, exist_ok=True)
    # Upstream replay requires a new directory. Repeated Run All is safe.
    return Path(tempfile.mkdtemp(prefix='run-', dir=parent))


def table(headers, rows):
    from IPython.display import HTML, display
    cells = lambda values, tag: ''.join(
        f'<{tag} style="text-align:left;vertical-align:top;padding:6px 10px">'
        f'{html.escape(str(v))}</{tag}>' for v in values)
    body = ''.join('<tr>'+cells(row, 'td')+'</tr>' for row in rows)
    display(HTML('<table><thead><tr>'+cells(headers, 'th')+'</tr></thead><tbody>'+body+'</tbody></table>'))


def gallery(items, width=1050, max_height=None, fit_height=None, image_max_width=None):
    """Display native PNG output while retaining the source vector files."""
    import io
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    from IPython.display import Image, display
    images = []
    for title, path in items:
        path = Path(path)
        if path.suffix == '.svg':
            png = path.with_suffix('.png')
            subprocess.run(['rsvg-convert', '--format=png', '--width=1800',
                            '--background-color=white', '--output', str(png), str(path)],
                           check=True, capture_output=True)
            path = png
        with PILImage.open(path) as source:
            images.append((title, source.convert('RGB').copy()))
    if len(images) == 1 and not images[0][0]:
        buffer = io.BytesIO()
        images[0][1].save(buffer, format='PNG')
        display(Image(data=buffer.getvalue(), width=width))
        return
    # Equal-width columns keep comparison panels together on GitHub.
    height = max(im.height / im.width for _, im in images) * width / len(images)
    if fit_height is not None:
        height = min(height, fit_height)
    fig, axes = plt.subplots(1, len(images), squeeze=False,
                             figsize=(width / 150, (height + 65) / 150))
    for ax, (title, im) in zip(axes[0], images):
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.tight_layout(pad=.5)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    display(Image(data=buffer.getvalue(), width=width))


def render_panels(examples, run, example):
    figures = run / 'figures'
    command = [sys.executable, '-B', str(examples / 'render.py'), '--example', example,
               '--data', str(run), '--out', str(figures)]
    result = subprocess.run(command, capture_output=True, text=True)
    (run / 'render.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'Upstream rendering failed; see {run / "render.log"}\n{result.stderr}')
    panels = sorted((figures / example).rglob('*.pdf'))
    if len(panels) != 6:
        raise RuntimeError(f'Expected six {example} panels; got {len(panels)}.')
    for pdf in panels:
        subprocess.run(['pdftoppm', '-png', '-singlefile', '-scale-to', '1500',
                        str(pdf), str(pdf.with_suffix(''))], check=True, capture_output=True)
    print(f'Six {example} reference panels regenerated.')
    return figures / example


def save_run(root, example, run, checks, observation_shape):
    from importlib.metadata import version
    pin = json.loads((root / 'upstream.json').read_text())
    record = {'example': example, 'upstream_commit': pin['revision'],
              'cmgdb_version': version('CMGDB'), 'explicit_enclosures_recomputed': True,
              'fitted_enclosures_source': 'Supplied upstream numerical arrays',
              'fitted_cmgdb_recomputed': True, 'model_refitted': False,
              'figure_panels_regenerated': 6, 'checks': checks,
              'demonstration_observation_shape': list(observation_shape),
              'run_directory': str(run.relative_to(root))}
    (run / 'verification.json').write_text(json.dumps(record, indent=2)+'\n')
    (root / 'build/notebooks' / example / 'latest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(f'Saved figures and replay results: {run.relative_to(root)}')
    return record
