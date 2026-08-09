"""Automatic machine description. Standard library only, so it runs on the device itself.

Every probe here degrades to None rather than raising. A benchmark run must not fail
because a board does not expose a compiler version, and a missing field is honest whereas
a guessed one is not.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import Any, Dict, List, Optional

# Flags that actually change which kernels an inference backend selects. Recording the
# full flag list is noise; these are the ones that explain a throughput difference.
_INTERESTING_FLAGS = {
    # aarch64
    "asimd", "asimddp", "asimdhp", "sve", "sve2", "i8mm", "bf16", "fphp",
    # x86_64
    "avx", "avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8", "amx_bf16",
    "f16c", "fma", "sse4_2",
}


def _run(cmd, timeout: float = 5.0) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", errors="replace")


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def cpu_model() -> Optional[str]:
    system = platform.system()
    if system == "Linux":
        text = _read("/proc/cpuinfo") or ""
        for key in ("model name", "Model name", "Hardware", "Model"):
            m = re.search(rf"^{key}\s*:\s*(.+)$", text, re.MULTILINE)
            if m:
                return m.group(1).strip()
        # Arm boards frequently leave model name blank and put the board here instead.
        dt = _read("/proc/device-tree/model")
        if dt:
            return dt.strip("\x00 \n")
    elif system == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
    elif system == "Windows":
        out = _run(["wmic", "cpu", "get", "name", "/value"])
        if out:
            m = re.search(r"Name=(.+)", out)
            if m:
                return m.group(1).strip()
        env = os.environ.get("PROCESSOR_IDENTIFIER")
        if env:
            return env.strip()
    return platform.processor() or None


def board_model() -> Optional[str]:
    """Raspberry Pi and similar boards expose a device-tree model string."""
    dt = _read("/proc/device-tree/model")
    return dt.strip("\x00 \n") if dt else None


def cpu_flags() -> Optional[List[str]]:
    text = _read("/proc/cpuinfo")
    if not text:
        return None
    m = re.search(r"^(?:flags|Features)\s*:\s*(.+)$", text, re.MULTILINE)
    if not m:
        return None
    present = [f for f in m.group(1).split() if f in _INTERESTING_FLAGS]
    return sorted(set(present)) or None


def physical_cores() -> Optional[int]:
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.physicalcpu"])
        if out and out.strip().isdigit():
            return int(out.strip())
    elif system == "Linux":
        text = _read("/proc/cpuinfo")
        if text:
            pairs = set()
            phys = None
            for line in text.splitlines():
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    pairs.add((phys, line.split(":")[1].strip()))
            if pairs:
                return len(pairs)
            # Arm cores report neither field; every listed processor is a real core.
            n = len(re.findall(r"^processor\s*:", text, re.MULTILINE))
            if n:
                return n
    elif system == "Windows":
        out = _run(["wmic", "cpu", "get", "NumberOfCores", "/value"])
        if out:
            nums = [int(x) for x in re.findall(r"NumberOfCores=(\d+)", out)]
            if nums:
                return sum(nums)
    return None


def total_ram_bytes() -> Optional[int]:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
        except Exception:
            return None
    return None


def total_swap_bytes() -> Optional[int]:
    text = _read("/proc/meminfo")
    if text:
        m = re.search(r"^SwapTotal:\s+(\d+) kB$", text, re.MULTILINE)
        if m:
            return int(m.group(1)) * 1024
    return None


def distro() -> Optional[str]:
    text = _read("/etc/os-release")
    if text:
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?$', text, re.MULTILINE)
        if m:
            return m.group(1).strip()
    if platform.system() == "Darwin":
        out = _run(["sw_vers", "-productVersion"])
        if out:
            return "macOS " + out.strip()
    if platform.system() == "Windows":
        return platform.platform()
    return None


def kernel() -> Optional[str]:
    if platform.system() == "Linux":
        out = _read("/proc/version")
        if out:
            return out.split()[2] if len(out.split()) > 2 else out.strip()
    return platform.release() or None


def compiler() -> Optional[str]:
    """The system compiler, which is what built any locally compiled backend."""
    for cmd in (["cc", "--version"], ["gcc", "--version"], ["clang", "--version"]):
        out = _run(cmd)
        if out:
            return out.splitlines()[0].strip()
    if platform.system() == "Windows":
        return None  # cl.exe is not on PATH outside a developer prompt; report nothing.
    return None


def libc() -> Optional[str]:
    try:
        name, version = platform.libc_ver()
    except (OSError, AttributeError):
        return None
    return f"{name} {version}".strip() or None


def gpu() -> Optional[str]:
    """Best-effort GPU name. Present so the abstraction is real rather than aspirational.

    There is no GPU machine in this lab yet, so this is deliberately a thin probe: it
    reports what it can find and None otherwise, and no analysis depends on it.
    """
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if out and out.strip():
        return out.strip().splitlines()[0].strip()
    if platform.system() == "Windows":
        out = _run(["wmic", "path", "win32_VideoController", "get", "name", "/value"])
        if out:
            names = [m.strip() for m in re.findall(r"Name=(.+)", out) if m.strip()]
            if names:
                return names[0]
    return None


def library_versions(*modules: str) -> Dict[str, str]:
    """Versions of whichever named modules are importable. Never imports eagerly."""
    import importlib

    found: Dict[str, str] = {}
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        version = getattr(mod, "__version__", None)
        if version:
            found[name] = str(version)
    return found


def collect(device_id: Optional[str] = None, kind: str = "local") -> Dict[str, Any]:
    """One dict describing this machine, shaped to drop straight into DeviceInfo."""
    return {
        "device_id": device_id or platform.node() or "unknown",
        "kind": kind,
        "os": platform.system(),
        "os_release": platform.release(),
        "distro": distro(),
        "kernel": kernel(),
        "machine": platform.machine(),
        "cpu": cpu_model(),
        "cpu_flags": cpu_flags(),
        "logical_cores": os.cpu_count(),
        "physical_cores": physical_cores(),
        "ram_bytes": total_ram_bytes(),
        "swap_bytes": total_swap_bytes(),
        "board_model": board_model(),
        "gpu": gpu(),
        "python": platform.python_version(),
        "compiler": compiler(),
        "libc": libc(),
    }


def default_thread_sweep(max_threads: Optional[int] = None) -> List[int]:
    """A small, well-spaced thread sweep: 1, half, all physical, all logical.

    Decode saturates at a low thread count while prefill keeps scaling, so a dense sweep
    mostly buys duplicate points. What matters is bracketing the knee and showing prefill
    still climbing past it.
    """
    logical = max_threads or os.cpu_count() or 4
    phys = min(physical_cores() or logical, logical)
    candidates = {1, max(1, phys // 2), phys, logical}
    return sorted(t for t in candidates if 1 <= t <= logical)
