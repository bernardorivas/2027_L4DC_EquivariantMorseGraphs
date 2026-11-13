"""Execute the editable notebook and generate a local inspection preview."""
from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parents[1]
for key, relative in {
    'MPLCONFIGDIR': 'build/matplotlib',
    'XDG_CACHE_HOME': 'build/cache',
    'IPYTHONDIR': 'build/ipython',
    'JUPYTER_CONFIG_DIR': 'build/jupyter/config',
    'JUPYTER_DATA_DIR': 'build/jupyter/data',
    'JUPYTER_RUNTIME_DIR': 'build/jupyter/runtime',
}.items():
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

name = sys.argv[1] if len(sys.argv) > 1 else "01_sphere.ipynb"
if name not in {"01_sphere.ipynb", "02_o3_flow.ipynb"}:
    raise ValueError("Choose an existing example notebook.")
path = root / "notebooks" / name
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv[0] = sys.executable
client = NotebookClient(notebook, timeout=1200, km=manager,
                        resources={'metadata': {'path': str(path.parent)}})
client.execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
html, _ = HTMLExporter().from_notebook_node(notebook)
html = html.replace('<head>', '<head>\n<base href="../../../notebooks/">', 1)
example = 'sphere' if name == '01_sphere.ipynb' else 'o3'
preview = root / 'build/notebooks' / example / 'preview.html'
preview.write_text(html)
print('Notebook executed top-to-bottom and saved with outputs.')
print('Analytical replay, fit diagnostics, and CMGDB comparisons saved. Cell outputs record whether the fit was loaded or trained.')
