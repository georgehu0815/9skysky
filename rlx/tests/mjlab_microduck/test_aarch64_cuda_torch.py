"""Keep Torch/CUDA isolated to the optional mjlab MicroDuck integration.

Base RLX is MLX-native and must not depend on Torch. The GPU-oriented mjlab
extra pins Torch directly so that RLX's MLX CUDA backend and Torch use the same
CUDA 12.9 runtime on Linux. The CUDA index is also required on Linux aarch64,
where PyPI's Torch wheel is CPU-only.
"""

import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CUDA_INDEX = "https://download.pytorch.org/whl/cu"


def _packages(name):
    lock = tomllib.loads((_ROOT / "uv.lock").read_text())
    return [p for p in lock["package"] if p["name"] == name]


def _registry(pkg):
    return pkg.get("source", {}).get("registry", "")


def _markers(pkg):
    return " ".join(pkg.get("resolution-markers", []))


def _linux_entries(pkgs):
    """Return lock entries selected for Linux."""
    return [p for p in pkgs if "sys_platform == 'linux'" in _markers(p)]


def test_torch_is_only_a_direct_dependency_of_mjlab_extra():
    """Keep base RLX Torch-free while making uv source routing deterministic."""
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    base_dependencies = pyproject["project"]["dependencies"]
    mjlab_dependencies = pyproject["project"]["optional-dependencies"][
        "mjlab-microduck"
    ]
    torch_dependency = (
        "torch==2.9.1; python_version >= '3.12' and python_version < '3.13'"
    )
    mjlab_dependency = (
        "mjlab==1.3.0; python_version >= '3.12' and python_version < '3.13'"
    )

    assert not any(
        dependency.partition(";")[0].split("=")[0].strip() == "torch"
        for dependency in base_dependencies
    )
    assert mjlab_dependencies.count(torch_dependency) == 1
    assert mjlab_dependency in mjlab_dependencies


def test_torch_source_is_cuda_12_9_on_linux():
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    uv_cfg = pyproject["tool"]["uv"]
    sources = uv_cfg["sources"]["torch"]
    indexes = {item["name"]: item["url"] for item in uv_cfg["index"]}
    assert len(sources) == 1
    assert sources[0]["marker"] == "sys_platform == 'linux'"
    assert indexes[sources[0]["index"]].startswith(_CUDA_INDEX)


def test_lockfile_routes_linux_torch_to_cuda_12_9():
    linux = _linux_entries(_packages("torch"))
    assert linux, "no Linux torch entry found"
    for package in linux:
        assert _registry(package).startswith(_CUDA_INDEX)
        wheels = " ".join(wheel["url"] for wheel in package["wheels"])
        assert "%2Bcu129" in wheels or "+cu129" in wheels


def test_torch_version_identical_across_platforms():
    versions = {package["version"].split("+")[0] for package in _packages("torch")}
    assert versions == {"2.9.1"}


def _on_spark():
    return (
        sys.platform == "linux"
        and platform.machine() == "aarch64"
        and shutil.which("nvidia-smi") is not None
        and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    )


@pytest.mark.skipif(not _on_spark(), reason="not a linux-aarch64 machine with a GPU")
def test_installed_torch_actually_sees_the_gpu():
    """Verify the optional GPU stack seen by mjlab's select_gpus()."""
    import torch

    assert torch.cuda.device_count() > 0, (
        f"torch {torch.__version__} (cuda={torch.version.cuda}) sees no GPU "
        "although nvidia-smi reports one -> select_gpus() will raise IndexError."
    )
