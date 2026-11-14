"""Morse graphs and Conley indices of equivariant maps, computed on the orbit space with CMGDB."""

import warnings

# tqdm, which EMLP imports, warns in a notebook that its progress bars need ipywidgets; none are shown.
warnings.filterwarnings("ignore", message="IProgress not found")
