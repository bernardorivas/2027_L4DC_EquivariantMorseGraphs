"""Choice of the device for training, made before JAX starts.

The order of preference is an Apple GPU (MPS), then CUDA, then the CPU. EMLP cannot
use an Apple GPU: with Apple's `jax-metal` plugin the GPU has no float64 and no `eigh`,
which EMLP uses to build its equivariant bases, and EMLP's PyTorch layers keep their
projection matrices on the CPU. So on a Mac the CPU is used, and several seeds train
as parallel processes instead. CUDA is used when JAX has a CUDA build installed.

Call `configure` at the start of a process, before anything imports JAX.
"""
import importlib.util
import os
import platform
import sys

MPS_REASON = ("Apple GPU unavailable for EMLP: jax-metal lacks float64 and eigh, and EMLP's "
              "PyTorch layers keep their projections on the CPU")


def cuda_available():
    return any(importlib.util.find_spec(name) is not None
               for name in ("jax_cuda12_plugin", "jax_cuda12_pjrt", "jax_cuda13_plugin"))


def configure(precision="float64", device=None):
    """Set the JAX platform and precision for this process and return the choice.

    `device` may force "cuda" or "cpu"; otherwise MPS, CUDA and CPU are tried in order.
    """
    if "jax" in sys.modules:
        raise RuntimeError("Configure the device before JAX is imported.")
    if precision not in ("float32", "float64"):
        raise ValueError("Use float32 or float64.")
    remarks = []
    if device is None:
        if sys.platform == "darwin" and platform.machine() == "arm64":
            remarks.append(MPS_REASON)
        device = "cuda" if cuda_available() else "cpu"
    if device not in ("cuda", "cpu"):
        raise ValueError("EMLP runs on CUDA or on the CPU.")
    os.environ["JAX_PLATFORMS"] = device
    os.environ["JAX_PLATFORM_NAME"] = device
    os.environ["JAX_ENABLE_X64"] = "1" if precision == "float64" else "0"
    if device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    return dict(device=device, precision=precision, remarks=remarks)
