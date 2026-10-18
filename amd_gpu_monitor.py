#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Luxs
"""amd_gpu_monitor - see what is using your AMD GPUs on Linux: memory per
program, load, clocks, temperatures, power and running model servers.

Graphics programs are read from /proc/<pid>/fdinfo. GPU memory belongs to a
GPU client (an open file), which several processes can hold at once. Each
client is counted once and credited to one holder: the owner the kernel
records (debugfs, root only, newer kernels) if it is a holder; else the only
holder that is not a system process (systemd's descriptor store, logind);
else, as a marked guess, the oldest holder. Compute programs (ROCm, e.g. Ollama) are read
from /sys/class/kfd. Which kernel counter feeds which figure, and why, is
documented next to REQUESTED_KEY below.

Requires Linux, an AMD GPU driven by amdgpu, and Python 3.7 or newer.
Run as root to include every user's programs.
"""
# Keeps annotations as unevaluated strings, so the modern syntax used in
# them (int | None, list[str]) also runs on Python 3.7-3.9. Do not remove,
# and do not evaluate annotations at runtime (e.g. typing.get_type_hints).
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
import unicodedata
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime

# ===========================================================================
#  EDIT HERE: name and credit line
#  CREDIT_LINE is printed at the bottom of every view, in --help and in
#  --version. Set it to "" to print nothing.
# ===========================================================================
PROGRAM = "amd_gpu_monitor"
CREDIT_LINE = ("(c) Luxs 2026 · GitHub: https://github.com/luxs-code/amd_gpu_monitor"
               " · Web: https://mycoding.uk/")
# ===========================================================================

__version__ = "2.10.0"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2          # used by argparse for invalid arguments
EXIT_UNSUPPORTED = 3

MIB = 1024 ** 2
GIB = 1024 ** 3
DEFAULT_REFRESH = 2.0
MIN_REFRESH = 0.2
CRITICAL_FREE = GIB             # below this, the desktop will lag
TIGHT_FREE = int(2.5 * GIB)     # below this, warn that room is tight
EVICTED_NOTICE = 512 * MIB      # mention evicted memory above this
EVICTED_HIGHLIGHT = 64 * MIB    # highlight a program's evicted memory above this
TEMP_WARN_MARGIN = 10.0         # warn this many degrees below a sensor's critical limit
# Per-program GPU time: fdinfo drm-engine-<name> is each client's cumulative
# busy time per engine type, in ns (amdgpu_fdinfo.c). GPU% = growth of the
# shader engines' time between two refreshes / elapsed time; several rings of
# one type running at once can push it above 100%. ROCm work runs on
# user-mode queues outside this accounting (KFD's per-process cu_occupancy
# exists only on gfx9 GPUs), so ROCm rows show n/a.
SHADER_ENGINES = ("gfx", "compute")

# GPU sensors (hwmon), checked against the amdgpu driver of Linux 5.19-6.17 and
# its own documentation (DOC: hwmon in drivers/gpu/drm/amd/pm/amdgpu_pm.c):
#
#   Only a hwmon chip whose "name" is "amdgpu" is interpreted. The hwmon ABI
#   lets other chips use the same attribute names with other units (some
#   report temperatures in millivolts), so names alone prove nothing.
#   temp[1-3]_input/_crit/_emergency: millidegrees Celsius. Labels: "edge"
#     (die edge), "junction" (hotspot, hottest point), "mem" (VRAM). Channels
#     2 and 3 exist on SOC15 discrete GPUs only. emergency = chip shutdown.
#   power1_average: average SoC power, microwatts (the driver converts
#     explicitly). power1_input: instantaneous SoC power (Linux 6.12+). These
#     are different sensors and are reported separately. On APUs both
#     include the CPU, so they are labelled "SoC", not "GPU".
#   freq1_input: shader (gfx/compute) clock, freq2_input: memory clock, both
#     in hertz (all versions 5.19-6.17), read under the same sleep rule.
#   power1_cap: the selected power limit, microwatts. power1_label: "PPT"
#     (or "slowPPT"/"fastPPT" on one APU).
#   Side effect: up to Linux 6.14, reading gpu_busy_percent or any of these
#   wakes a runtime-suspended GPU (pm_runtime_get_sync); from 6.15 the read
#   fails instead. So nothing is read while power/runtime_status says the GPU
#   is suspended. mem_info_vram_* are plain counters and never wake it.
#   Values outside the ranges below are treated as bad readings.
TEMP_RANGE = (-40.0, 200.0)     # degrees Celsius
POWER_RANGE = (0.0, 3000.0)     # watts
CLOCK_RANGE = (1.0, 10000.0)    # MHz
TEMP_NAMES = {"edge": "edge", "junction": "hotspot", "mem": "memory"}
ASLEEP_STATES = {"suspended", "suspending"}

# Live view averages, like top's load averages. The kernel only reports
# current values, so these are built from the tool's own samples: the plain
# mean of every sample taken in the last 1, 5 and 15 minutes (top uses
# exponentially damped averages instead). A single snapshot has none.
AVERAGE_WINDOWS = ((60, "1 min"), (300, "5 min"), (900, "15 min"))

PCI_VENDORS = {"0x1002": "AMD", "0x10de": "NVIDIA", "0x8086": "Intel",
               "0x1a03": "ASPEED", "0x15ad": "VMware", "0x1af4": "virtio",
               "0x1234": "QEMU"}
UNIT_BYTES = {"": 1, "b": 1, "bytes": 1, "kib": 1024, "kb": 1024,
              "mib": MIB, "mb": MIB, "gib": GIB, "gb": GIB}
# Kernel counters and what they mean, checked against the amdgpu source of
# Linux 5.19, 6.2, 6.4, 6.8, 6.10-6.17 and against real output from 6.14+:
#
#   amd-requested-vram  (6.4+)  buffers that prefer VRAM. 6.4-6.13: VRAM is one
#                       of the preferred domains; 6.14+: VRAM is the top
#                       preferred domain, buffers without memory included.
#   amd-evicted-vram    (6.4+)  the part of those buffers that has memory, but
#                       not where it prefers: 6.4-6.13: not in VRAM; 6.14+:
#                       outside EVERY preferred domain (a buffer that also
#                       allows GTT and sits there is not counted). Always a
#                       subset of requested. Where such memory is: GTT or plain
#                       system memory (amdgpu_evict_flags), and system memory
#                       can be swapped out (ttm_tt_swapout), so the tool says
#                       "placed elsewhere", not "in system RAM". A rising count
#                       means buffers were moved out OR newly placed outside;
#                       it does not prove anything was pushed out.
#   mem_info_gtt_used   all GTT in use: system memory mapped for the GPU,
#                       including buffers placed there on purpose. Not a
#                       measure of eviction or spill-over.
#   drm-resident-vram   (6.13+) and drm-memory-vram (all versions; an alias
#                       on 6.13+): buffers currently in VRAM. Can never exceed
#                       requested, but is implausibly large for long-running
#                       clients on some 6.14+ kernels, so it is checked.
#   drm-total-vram      NOT used: on 6.13 it counts by current placement, on
#                       6.14+ it equals amd-requested-vram, before 6.13 absent.
#   KFD vram_<gpu_id>   ROCm allocations made with the VRAM flag, counted at
#                       allocation time and subtracted at free; imported
#                       buffers never count. It says nothing about location;
#                       on APUs with apu_prefer_gtt these "VRAM" allocations
#                       are even placed in system memory. So it only ever
#                       feeds "requested", never "in VRAM".
#
# "In VRAM" = the resident counter when plausible, else requested minus
# evicted, shown as an estimate. On 6.4-6.13 that difference equals resident
# memory; on 6.14+ it is an upper bound, because buffers that are in another
# preferred domain, or have no memory yet, are not counted as evicted.
#
# Bounds. Requested memory has NO upper bound tied to the card: the DRM
# usage-stats spec counts buffers whose memory is not allocated yet, and
# amdgpu only limits each single buffer to the size of its domain
# (amdgpu_bo_validate_size), never the sum. A request larger than the card
# is therefore real (an overcommitted program) and is reported, not dropped.
# KFD refuses ROCm VRAM allocations beyond the card on discrete GPUs, but not
# on APUs that place them in system memory (apu_prefer_gtt), so ROCm figures
# are not checked against the card either. Only memory that must physically
# be in VRAM is bounded: a resident figure above the card's size is
# implausible, and every in-VRAM ESTIMATE is capped at what the whole card is
# currently using (mem_info_vram_used), since no client can hold more.
# Exception: GPUs in one XGMI hive (xgmi_hive_info/xgmi_hive_id) map each
# other's VRAM into their address spaces, so a client's "VRAM" may physically
# be on a peer GPU; for them the whole hive's size and use are the limits.
REQUESTED_KEY = "amd-requested-vram"
EVICTED_KEY = "amd-evicted-vram"
RESIDENT_KEYS = ("drm-resident-vram", "drm-memory-vram")
SYSTEM_HOLDERS = {"systemd", "systemd-logind"}
LINE_RE = re.compile(r"^([\w.-]+):[ \t]*(\S+)(?:[ \t]+(\S+))?[ \t]*$", re.M)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

DESCRIPTIONS = {                # shown in the table; anything else stays blank
    # compute
    "llama-server": "LLM server (llama.cpp)", "llama-cli": "LLM (llama.cpp)",
    "ollama": "Ollama", "python": "Python program", "pt_main_thread": "PyTorch program",
    "blender": "Blender",
    # desktops and compositors
    "gnome-shell": "desktop & compositor", "kwin_wayland": "KDE compositor",
    "kwin_x11": "KDE window manager", "plasmashell": "KDE desktop shell",
    "Xorg": "X server", "Xwayland": "X11 app support", "sway": "compositor (sway)",
    "Hyprland": "compositor (Hyprland)", "weston": "compositor (Weston)",
    "gamescope": "game compositor", "mutter-x11-frames": "X11 window frames",
    "xdg-desktop-portal": "desktop portal", "gsd-xsettings": "GNOME settings helper",
    # applications
    "firefox": "web browser", "RDD Process": "Firefox video decoder",
    "chrome": "web browser", "chromium": "web browser", "steam": "Steam client",
    "obs": "OBS Studio", "mpv": "video player", "vlc": "video player",
    "nautilus": "file manager", "gnome-system-monitor": "system monitor",
    "radeontop": "GPU monitor", "code": "VS Code",
}


class UnsupportedSystemError(Exception):
    """The platform or GPU cannot be monitored by this tool."""


# ---------------------------------------------------------------------------
# Data model (all sizes in bytes)
# ---------------------------------------------------------------------------
@dataclass
class PciGpu:
    address: str
    vendor: str
    driver: str
    path: str


@dataclass
class TempSensor:
    channel: int              # N in tempN_*
    label: str                # kernel label ("edge", "junction", "mem")
    name: str                 # display name ("edge", "hotspot", "memory")
    celsius: float
    crit: float | None = None       # critical limit
    emergency: float | None = None  # shutdown limit


@dataclass
class PowerReading:
    """SoC power from amdgpu; on APUs it includes the CPU."""
    average_w: float | None   # power1_average
    input_w: float | None     # power1_input (instantaneous, Linux 6.12+)
    cap_w: float | None       # power1_cap
    label: str | None         # power1_label, e.g. "PPT"


@dataclass
class GpuDevice:
    address: str
    name: str
    vram_total: int
    vram_used: int
    gtt_total: int
    gtt_used: int
    busy_percent: int | None
    temperatures: list[tuple[str, float]]   # kept for compatibility
    power_watts: float | None                # kept: average, else instantaneous
    temperature_sensors: list[TempSensor] = field(default_factory=list)
    power: PowerReading | None = None
    runtime_status: str | None = None        # "active", "suspended", ...
    xgmi_hive: str | None = None             # GPUs in one hive share VRAM mappings
    vram_used_reported: int | None = None    # raw reading; vram_used is capped at total
    gtt_used_reported: int | None = None
    sclk_mhz: float | None = None            # shader (gfx/compute) clock
    mclk_mhz: float | None = None            # memory clock

    @property
    def asleep(self) -> bool:
        return self.runtime_status in ASLEEP_STATES

    @property
    def vram_free(self) -> int:
        return max(0, self.vram_total - self.vram_used)


@dataclass
class ClientUsage:
    """One GPU client (open DRM file) and how it was credited to a process."""
    client_id: str
    gpu: str | None
    pasid: int | None         # address-space id; links ROCm to a client (<= 6.14)
    requested: int | None     # amd-requested-vram (None: kernel older than 6.4)
    evicted: int | None       # amd-evicted-vram (None: kernel older than 6.4)
    resident_reported: int | None  # resident counter exactly as reported
    in_vram: int              # best figure for memory currently in VRAM
    in_vram_source: str       # "resident", "requested-minus-evicted" or "unavailable"
    holders: list[int]        # every process holding this client
    owner: int                # the process it is credited to
    attribution: str          # "kernel", "sole-holder", "system" or "heuristic"
    in_vram_capped: bool = False   # estimate was capped at the card's VRAM use
    engine_ns: dict[str, int] = field(default_factory=dict)   # drm-engine-*: busy time


@dataclass
class ComputeUsage:
    """A process's ROCm (KFD) memory on one GPU and how it was merged."""
    gpu: str | None
    kfd_vram: int             # KFD vram_<gpu_id>: VRAM allocations, counted at allocation
    linked_client: str | None # drm-client-id of the address space KFD uses
    link: str                 # "pasid", "only-client", "guess" or "none"
    beyond_client: int        # KFD memory the linked client does not show; at
                              # least this much is allocated but not yet mapped
    in_vram: int              # VRAM in the linked client's address space (DRM)
    estimated: bool           # that figure is requested-minus-evicted
    in_vram_known: bool = True     # False: no linked client visible


@dataclass
class GpuPart:
    """One process's memory on one GPU. With several GPUs, the table shows one
    row per part, its bar scaled to that GPU."""
    gpu: str | None
    in_vram: int = 0
    compute_in_vram: int = 0  # part of in_vram in the ROCm address space
    evicted: int = 0
    compute_evicted: int = 0
    rocm_allocated: int = 0   # KFD allocations on this GPU (not a location)
    in_vram_estimated: bool = False
    graphics_estimated: bool = False
    evicted_known: bool = True
    in_vram_known: bool = True
    clients: int = 0


@dataclass
class ProcessUsage:
    pid: int
    name: str
    description: str
    kind: str                 # "compute" or "graphics"
    requested: int = 0        # buffers that prefer VRAM (see REQUESTED_KEY)
    in_vram: int = 0          # best figure for memory currently in VRAM
    evicted: int = 0          # part of requested that is elsewhere
    compute_in_vram: int = 0  # part of in_vram that is ROCm (KFD) memory
    compute_evicted: int = 0  # part of evicted that is ROCm (KFD) memory
    in_vram_estimated: bool = False   # some of in_vram is an estimate
    evicted_known: bool = True        # False on kernels older than 6.4
    in_vram_capped: bool = False      # an estimate was capped at the card's VRAM use
    rocm_allocated: int = 0           # KFD: ROCm VRAM allocations (not a location)
    in_vram_known: bool = True        # False: ROCm memory with no visible address space
    clients: int = 0
    gpus: list[str] = field(default_factory=list)   # PCI addresses
    client_details: list[ClientUsage] = field(default_factory=list)
    compute_details: list[ComputeUsage] = field(default_factory=list)
    parts: list[GpuPart] = field(default_factory=list)   # one per GPU used

    @property
    def graphics_in_vram(self) -> int:
        return max(0, self.in_vram - self.compute_in_vram)

    @property
    def graphics_evicted(self) -> int:
        return max(0, self.evicted - self.compute_evicted)

    @property
    def graphics_estimated(self) -> bool:
        linked = {c.linked_client for c in self.compute_details}
        return any(c.in_vram_source != "resident" for c in self.client_details
                   if c.client_id not in linked)

    @property
    def attribution_guessed(self) -> bool:
        """True if any of this program's memory was credited by guesswork."""
        return any(c.attribution == "heuristic" for c in self.client_details)


@dataclass
class ModelServer:
    """A local model server and what it reports."""
    kind: str                 # "Ollama", "llama.cpp", "vLLM", ...
    pid: int | None           # None: found at its default address, not by process
    address: str              # "127.0.0.1:11434", or "" if its port is not known
    status: str               # "ok", "no reply", "refused", "HTTP 404",
                              # "unexpected reply", "port unknown"
    models: list[str] = field(default_factory=list)          # served model names
    loaded: list[OllamaModel] = field(default_factory=list)  # Ollama: with sizes


@dataclass
class OllamaModel:
    name: str
    size: int
    size_vram: int


@dataclass
class Snapshot:
    timestamp: datetime
    devices: list[GpuDevice]
    processes: list[ProcessUsage]
    models: list[OllamaModel]
    is_root: bool
    warnings: list[str] = field(default_factory=list)
    compute_by_device: dict[str, int] = field(default_factory=dict)
    # live view only: GPU address -> metric -> window label -> mean
    averages: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    averaged_seconds: float = 0.0     # how much time the averages cover so far
    ollama_running: bool = False      # Ollama answered, even with no model loaded
    servers: list[ModelServer] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "program": PROGRAM,
            "version": __version__,
            "timestamp": self.timestamp.isoformat(timespec="seconds"),
            "root": self.is_root,
            "devices": [dict(asdict(d), vram_free=d.vram_free,
                             compute_used=self.compute_by_device.get(d.address, 0))
                        for d in self.devices],
            "processes": [dict(asdict(p), graphics_in_vram=p.graphics_in_vram,
                               graphics_evicted=p.graphics_evicted)
                          for p in self.processes],
            "ollama_models": [asdict(m) for m in self.models],
            "warnings": self.warnings,
            "averages": self.averages,                 # live view only
            "averaged_seconds": self.averaged_seconds,
            "ollama_running": self.ollama_running,
            "servers": [asdict(x) for x in self.servers],
        }


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _read(path: str) -> str | None:
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _safe(text: str) -> str:
    """Untrusted text with control and format characters replaced by '?'.

    Process and model names are chosen by whoever starts the program; printed
    raw, escape sequences in them could move the cursor, recolour or clear the
    terminal, and format characters could reverse the text direction.
    """
    return "".join("?" if unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp") else c
                   for c in text)


def _int(text: str | None) -> int:
    text = (text or "").strip()
    return int(text) if text.isdigit() else 0


def _comm(pid: str) -> str:
    return (_read(f"/proc/{pid}/comm") or "?").strip()


def _program_name(pid: str) -> str:
    """Program name, taken from the command line if the kernel truncated it."""
    name = _comm(pid)
    if len(name) == 15:
        argv0 = (_read(f"/proc/{pid}/cmdline") or "").split("\0")[0]
        base = os.path.basename(argv0)
        if base.startswith(name):
            return base
    return name


def _counter(info: dict[str, tuple[str, str]], key: str,
             bad_units: set[str]) -> int | None:
    """One fdinfo counter in bytes; None if absent or not understood."""
    if key not in info:
        return None
    value = _to_bytes(*info[key])
    if value is None:
        bad_units.add(f"{key}: {' '.join(info[key])}")
    return value


def _row_size(p: ProcessUsage) -> int:
    """Size used to order table rows; ROCm-only processes by their allocations."""
    return max(p.in_vram + p.evicted, p.rocm_allocated if not p.in_vram_known else 0)


def _requested_or_resident(c: ClientUsage) -> int:
    """A client's requested VRAM; on kernels older than 6.4, its resident VRAM."""
    return c.requested if c.requested is not None else c.in_vram


def _linked_client(details: list[ClientUsage],
                   kfd_pasid: int | None) -> tuple[ClientUsage | None, str]:
    """The DRM client whose address space KFD uses, and how it was found.

    Up to Linux 6.14, KFD gives that client its own PASID and publishes it in
    sysfs, so the match is exact. From 6.15 on, the PASID file reads 0: then
    the process's only client on that GPU is taken, else the largest is
    guessed. If the PASID is known but no visible client has it, the linked
    client's descriptor has been closed and nothing is merged.
    """
    if kfd_pasid:
        match = next((c for c in details if c.pasid == kfd_pasid), None)
        return (match, "pasid") if match else (None, "none")
    if len(details) == 1:
        return details[0], "only-client"
    if details:
        return max(details, key=_requested_or_resident), "guess"
    return None, "none"


def _start_time(pid: str) -> int:
    """Process start time in clock ticks since boot (immune to PID reuse)."""
    stat = _read(f"/proc/{pid}/stat") or ""
    fields = stat[stat.rfind(")") + 2:].split()      # comm may contain spaces
    return int(fields[19]) if len(fields) > 19 and fields[19].isdigit() else 0


def _is_system_holder(pid: str) -> bool:
    """systemd (descriptor store) or logind, judged by executable, not name."""
    if pid == "1":
        return True
    try:
        exe = os.path.basename(os.readlink(f"/proc/{pid}/exe"))
        exe = exe.replace(" (deleted)", "")             # binary upgraded on disk
    except OSError:
        exe = _comm(pid)                                # not root: best effort
    return exe in SYSTEM_HOLDERS


def _debugfs_device(clients_path: str) -> tuple[str, str] | None:
    """(driver, PCI address) of the device a debugfs 'clients' file belongs
    to, from the 'name' file beside it: "<driver> dev=<address> ... unique=..."
    (same directory in every kernel version: per minor up to 6.x, per device
    later, with the minor numbers as symlinks to it)."""
    words = (_read(os.path.join(os.path.dirname(clients_path), "name")) or "").split()
    address = next((w[4:] for w in words[1:] if w.startswith("dev=")), "")
    return (words[0], address.lower()) if words and address else None


def _kernel_client_owners() -> dict[tuple[str, str], int]:
    """(PCI address, drm-client-id) -> owner PID as recorded by the kernel.

    For render nodes the kernel updates the owner to the process that last
    used the client; for primary nodes it stays the process that opened it.
    Needs root, a mounted debugfs and a kernel whose 'clients' file has an
    'id' column. Returns {} when any of that is missing.

    Keys include the device. Client IDs are in fact system-wide (amdgpu's
    fence context up to 6.4, then a global counter in DRM core), but the
    lookup does not rely on that: each file is tied to its device by the
    'name' file beside it, and files of other drivers are skipped.
    """
    owners: dict[tuple[str, str], int] = {}
    for path in glob.glob("/sys/kernel/debug/dri/*/clients"):
        device = _debugfs_device(path)
        if device is None or device[0] != "amdgpu":
            continue                    # device unknown, or not an AMD GPU
        lines = (_read(path) or "").splitlines()
        if not lines:
            continue
        header = lines[0].split()
        if len(header) < 3 or header[0] != "command" or header[-1] != "id":
            continue                                    # unknown format
        columns = header[1:]           # parse from the right: command has spaces
        for line in lines[1:]:
            parts = line.split()
            if len(parts) <= len(columns):
                continue
            row = dict(zip(columns, parts[-len(columns):]))
            pid, cid = row.get("tgid") or row.get("pid"), row.get("id")
            if pid and pid.isdigit() and cid and cid.isdigit():
                owners[(device[1], cid)] = int(pid)
    return owners


def _attribute(pids: set[str], kernel_owner: int | None) -> tuple[str, str]:
    """Pick the process to credit a shared GPU client to, and say how."""
    candidates = sorted((p for p in pids if not _is_system_holder(p)), key=int)
    if kernel_owner is not None and str(kernel_owner) in candidates:
        return str(kernel_owner), "kernel"
    if len(candidates) == 1:
        return candidates[0], "sole-holder"
    if not candidates:              # only system processes still hold it
        return min(pids, key=_age_key), "system"
    return min(candidates, key=_age_key), "heuristic"


def _age_key(pid: str) -> tuple[int, int]:
    """Sort key: oldest process first; PID only breaks exact ties."""
    return _start_time(pid), int(pid)


def _describe(name: str) -> str:
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    return next((text for key, text in DESCRIPTIONS.items()
                 if name.startswith(key)), "")


def _parse_fdinfo(text: str) -> dict[str, tuple[str, str]]:
    return {m.group(1): (m.group(2), (m.group(3) or "").lower())
            for m in LINE_RE.finditer(text)}


def _to_bytes(value: str, unit: str) -> int | None:
    if not value.isdigit() or unit not in UNIT_BYTES:
        return None
    return int(value) * UNIT_BYTES[unit]


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
def detect_gpus() -> list[PciGpu]:
    """All display devices the kernel knows about, of any vendor."""
    found: dict[str, PciGpu] = {}
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        if not re.fullmatch(r"card\d+", os.path.basename(card)):
            continue                      # skip connectors like card1-DP-1
        dev = os.path.join(card, "device")
        if not os.path.exists(dev):
            continue
        address = os.path.basename(os.path.realpath(dev))
        vendor_id = (_read(f"{dev}/vendor") or "").strip().lower()
        driver_link = os.path.join(dev, "driver")
        driver = (os.path.basename(os.path.realpath(driver_link))
                  if os.path.exists(driver_link) else "no driver")
        vendor = PCI_VENDORS.get(vendor_id, f"vendor {vendor_id or 'unknown'}")
        found.setdefault(address, PciGpu(address, vendor, driver, dev))
    return list(found.values())


def find_supported_gpus() -> list[PciGpu]:
    """AMD GPUs this tool can monitor; raises UnsupportedSystemError if none."""
    if not sys.platform.startswith("linux"):
        raise UnsupportedSystemError(
            f"Linux is required; this system is {sys.platform!r}.")
    if not os.path.isdir("/sys/class/drm"):
        raise UnsupportedSystemError(
            "/sys/class/drm is missing: no GPU driver is loaded, "
            "or /sys is not available (for example inside a container).")
    gpus = detect_gpus()
    amd = [g for g in gpus if g.vendor == "AMD"]
    supported = [g for g in amd if g.driver == "amdgpu"
                 and _int(_read(f"{g.path}/mem_info_vram_total"))]
    if supported:
        return supported
    if any(g.driver == "amdgpu" for g in amd):
        raise UnsupportedSystemError(
            "the amdgpu driver does not report VRAM statistics for this GPU; "
            "a newer kernel is needed.")
    if amd:
        drivers = ", ".join(sorted({g.driver for g in amd}))
        raise UnsupportedSystemError(
            f"AMD GPU found, but it uses the '{drivers}' driver; "
            "the 'amdgpu' driver is required.")
    if gpus:
        vendors = ", ".join(sorted({g.vendor for g in gpus}))
        hint = (" For NVIDIA GPUs, use nvidia-smi."
                if any(g.vendor == "NVIDIA" for g in gpus) else "")
        raise UnsupportedSystemError(
            f"no AMD GPU found (found: {vendors}). Only AMD GPUs using the "
            f"amdgpu driver are supported.{hint}")
    raise UnsupportedSystemError("no GPU found.")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
_name_cache: dict[str, str] = {}


def _gpu_name(gpu: PciGpu) -> str:
    if gpu.address not in _name_cache:
        name = (_read(f"{gpu.path}/product_name") or "").strip()
        if not name:
            try:
                out = subprocess.run(["lspci", "-s", gpu.address],
                                     capture_output=True, text=True,
                                     timeout=2).stdout
                m = re.search(r"\[(Radeon[^\]]*)\]", out)
                name = ("AMD " + m.group(1)) if m else out.split(": ", 1)[-1]
            except (OSError, subprocess.SubprocessError):
                name = ""
        _name_cache[gpu.address] = _safe(name.strip()) or "AMD GPU"
    return _name_cache[gpu.address]


def _scaled(path: str, divisor: float, valid: tuple[float, float]) -> float | None:
    """A hwmon integer divided by divisor; None if absent, unreadable or implausible."""
    text = (_read(path) or "").strip()
    if not text.lstrip("-").isdigit():
        return None
    value = int(text) / divisor
    return value if valid[0] <= value <= valid[1] else None


def _amdgpu_hwmon(path: str) -> tuple[list[TempSensor], PowerReading | None,
                                       dict[str, float]]:
    """Temperatures, power and clocks (MHz) from the GPU's own amdgpu hwmon chip."""
    for hw in sorted(glob.glob(f"{path}/hwmon/hwmon*")):
        if (_read(f"{hw}/name") or "").strip() != "amdgpu":
            continue                  # not amdgpu's chip: units not guaranteed
        sensors: list[TempSensor] = []
        for t in glob.glob(f"{hw}/temp*_input"):
            m = re.fullmatch(r"temp(\d+)_input", os.path.basename(t))
            celsius = _scaled(t, 1000, TEMP_RANGE) if m else None
            if m is None or celsius is None:
                continue
            base = f"{hw}/temp{m.group(1)}"
            label = (_read(f"{base}_label") or "").strip() or f"temp{m.group(1)}"
            sensors.append(TempSensor(
                channel=int(m.group(1)), label=label,
                name=TEMP_NAMES.get(label, label), celsius=celsius,
                crit=_scaled(f"{base}_crit", 1000, TEMP_RANGE),
                emergency=_scaled(f"{base}_emergency", 1000, TEMP_RANGE)))
        sensors.sort(key=lambda x: x.channel)
        power = PowerReading(
            average_w=_scaled(f"{hw}/power1_average", 1e6, POWER_RANGE),
            input_w=_scaled(f"{hw}/power1_input", 1e6, POWER_RANGE),
            cap_w=_scaled(f"{hw}/power1_cap", 1e6, POWER_RANGE),
            label=(_read(f"{hw}/power1_label") or "").strip() or None)
        if power.average_w is None and power.input_w is None:
            power = None
        clocks = {name: mhz for name, attr in (("sclk", "freq1_input"), ("mclk", "freq2_input"))
                  for mhz in [_scaled(f"{hw}/{attr}", 1e6, CLOCK_RANGE)] if mhz is not None}
        return sensors, power, clocks
    return [], None, {}


def read_device(gpu: PciGpu) -> GpuDevice:
    status = (_read(f"{gpu.path}/power/runtime_status") or "").strip() or None
    asleep = status in ASLEEP_STATES          # reading sensors could wake it
    busy = "" if asleep else (_read(f"{gpu.path}/gpu_busy_percent") or "").strip()
    sensors, power, clocks = ([], None, {}) if asleep else _amdgpu_hwmon(gpu.path)
    legacy_power = None if power is None else (
        power.average_w if power.average_w is not None else power.input_w)
    total = _int(_read(f"{gpu.path}/mem_info_vram_total"))
    raw_vram_used = _int(_read(f"{gpu.path}/mem_info_vram_used"))
    gtt_total = _int(_read(f"{gpu.path}/mem_info_gtt_total"))
    raw_gtt_used = _int(_read(f"{gpu.path}/mem_info_gtt_used"))
    hive = (_read(f"{gpu.path}/xgmi_hive_info/xgmi_hive_id") or "").strip()  # "0"/absent: none
    return GpuDevice(
        address=gpu.address, name=_gpu_name(gpu), vram_total=total,
        vram_used=min(raw_vram_used, total), vram_used_reported=raw_vram_used,
        gtt_total=gtt_total,
        gtt_used=min(raw_gtt_used, gtt_total) if gtt_total else raw_gtt_used,
        gtt_used_reported=raw_gtt_used,
        busy_percent=int(busy) if busy.isdigit() and int(busy) <= 100 else None,
        temperatures=[(x.label, x.celsius) for x in sensors],
        power_watts=legacy_power, temperature_sensors=sensors, power=power,
        runtime_status=status, xgmi_hive=hive if hive.strip("0") else None,
        sclk_mhz=clocks.get("sclk"), mclk_mhz=clocks.get("mclk"))


def _kfd_gpu_addresses() -> dict[str, str]:
    """Map KFD gpu_id -> PCI address, using the KFD topology in sysfs."""
    out: dict[str, str] = {}
    for node in glob.glob("/sys/class/kfd/kfd/topology/nodes/*"):
        gpu_id = (_read(f"{node}/gpu_id") or "").strip()
        if not gpu_id.isdigit() or gpu_id == "0":       # 0 = CPU node
            continue
        props = {}
        for line in (_read(f"{node}/properties") or "").splitlines():
            parts = line.split()
            if len(parts) == 2:
                props[parts[0]] = parts[1]
        location, domain = props.get("location_id", ""), props.get("domain", "0")
        if location.isdigit() and domain.isdigit():
            loc = int(location)                          # (bus << 8) | devfn
            out[gpu_id] = (f"{int(domain):04x}:{loc >> 8:02x}:"
                           f"{(loc >> 3) & 0x1f:02x}.{loc & 7:x}")
    return out


def collect_processes(capacities: dict[str, int], used: dict[str, int] | None = None
                      ) -> tuple[list[ProcessUsage], dict[str, int], list[str]]:
    """Per-program VRAM use, compute use per GPU, and data-quality warnings.

    capacities and used map each monitored GPU's PCI address to its VRAM size
    and current VRAM use. Reported figures are never dropped for being large
    (see "Bounds" next to REQUESTED_KEY): only resident figures are checked
    against the card's size, and in-VRAM estimates are capped at its use. If
    a client's GPU cannot be identified, the largest card's figures are used.
    """
    largest = max(capacities.values())
    only = next(iter(capacities)) if len(capacities) == 1 else None

    used = used or capacities

    def capacity(address: str | None) -> int:
        return capacities[address] if address else largest

    def in_use(address: str | None) -> int:
        return used.get(address, capacity(address)) if address else max(used.values())

    # Graphics clients, each counted once even if several processes hold it
    clients: dict[tuple[str | None, str], list] = {}   # -> [req, ev, res, pasid, pids]
    bad_units: set[str] = set()
    without_stats = 0
    for path in glob.glob("/proc/[0-9]*/fdinfo/*"):
        text = _read(path)
        if not text or "drm-driver" not in text:
            continue
        info = _parse_fdinfo(text)
        if info.get("drm-driver", ("",))[0] != "amdgpu":
            continue
        address = info["drm-pdev"][0].lower() if "drm-pdev" in info else only
        if address is not None and address not in capacities:
            continue                      # a GPU this tool does not monitor
        requested = _counter(info, REQUESTED_KEY, bad_units)
        evicted = _counter(info, EVICTED_KEY, bad_units)
        resident = next((_counter(info, k, bad_units)
                         for k in RESIDENT_KEYS if k in info), None)
        if "drm-client-id" not in info or (requested is None and resident is None):
            without_stats += 1
            continue
        pasid = info.get("pasid", ("",))[0]
        cid = (address, info["drm-client-id"][0])
        engine = {k[len("drm-engine-"):]: int(v) for k, (v, unit) in info.items()
                  if k.startswith("drm-engine-") and "capacity" not in k
                  and v.isdigit() and unit in ("ns", "")}
        clients.setdefault(cid, [requested, evicted, resident,
                                 int(pasid) if pasid.isdigit() else None,
                                 set(), engine])[4].add(path.split("/")[2])

    # Credit each client to one holder, per GPU
    owners = _kernel_client_owners()
    graphics: dict[tuple[str, str | None], list[ClientUsage]] = {}
    for (address, cid), (requested, evicted, resident, pasid, pids, engine) in clients.items():
        cap = capacity(address)
        if requested is not None and evicted is not None:
            evicted = min(evicted, requested)
        # Resident memory is a subset of requested memory in every kernel
        # version (a buffer can only be in VRAM if it lists VRAM as preferred).
        capped = False
        if resident is not None and resident <= cap and (
                requested is None or resident <= requested):
            in_vram, source = resident, "resident"
        elif requested is None:               # pre-6.4 kernel, implausible resident
            in_vram, source = 0, "unavailable"
        else:
            estimate = max(0, requested - (evicted or 0))
            in_vram, source = min(estimate, in_use(address)), "requested-minus-evicted"
            capped = estimate > in_vram
        pid, how = _attribute(pids, owners.get((address, cid)) if address else None)
        graphics.setdefault((pid, address), []).append(ClientUsage(
            client_id=cid, gpu=address, pasid=pasid, requested=requested, evicted=evicted,
            resident_reported=resident, in_vram=in_vram, in_vram_source=source,
            in_vram_capped=capped, engine_ns=engine,
            holders=sorted(int(p) for p in pids), owner=int(pid), attribution=how))

    # Compute (ROCm) memory, per GPU
    kfd_map = _kfd_gpu_addresses()
    compute: dict[tuple[str, str | None], int] = {}
    kfd_pasids: dict[str, int] = {}     # nonzero only on Linux 6.14 and older
    for d in glob.glob("/sys/class/kfd/kfd/proc/[0-9]*"):
        pid = os.path.basename(d)
        if _int(_read(f"{d}/pasid")):
            kfd_pasids[pid] = _int(_read(f"{d}/pasid"))
        for f in glob.glob(f"{d}/vram_*"):
            size = _int(_read(f))
            if not size:
                continue
            address = kfd_map.get(f.rsplit("_", 1)[-1], only)
            if address is not None and address not in capacities:
                continue
            compute[(pid, address)] = compute.get((pid, address), 0) + size

    # Merge graphics and compute, per GPU. KFD runs a process's ROCm work in
    # the address space of ONE of its DRM clients (the "linked" client); a
    # ROCm buffer mapped on that GPU is that client's own buffer, so the
    # client's fdinfo counts it, along with page tables and any other buffers
    # in that address space. The KFD counter is an allocation total with no
    # location. Therefore:
    #   in VRAM      = DRM clients only (the only location figures);
    #   compute part = the linked client: "VRAM in the ROCm address space";
    #   requested   += max(0, KFD - linked client's requested): at least that
    #                  much ROCm memory is allocated but not mapped, location
    #                  not reported (shown in the status box, not guessed).
    # Which client is linked only changes the compute/graphics split, never
    # the total: a guessed link (Linux 6.15+, several clients) is flagged.
    usage: dict[str, ProcessUsage] = {}
    compute_by_device: dict[str, int] = {}
    for pid, address in sorted(set(graphics) | set(compute), key=str):
        details = graphics.get((pid, address), [])
        kfd = compute.get((pid, address), 0)
        linked, link = (_linked_client(details, kfd_pasids.get(pid))
                        if kfd else (None, "none"))
        u = usage.setdefault(pid, ProcessUsage(int(pid), "", "", "graphics"))
        others = [c for c in details if c is not linked]
        g_in = sum(c.in_vram for c in others)
        c_in = linked.in_vram if linked else 0
        g_estimated = any(c.in_vram_source != "resident" for c in others)
        c_estimated = linked is not None and linked.in_vram_source != "resident"
        # A process cannot hold more VRAM on a GPU than that GPU (or its hive)
        # uses. Measured figures respect that already; trim estimated ones.
        limit = in_use(address)
        if g_in + c_in > limit:
            u.in_vram_capped = True
            if c_estimated:
                c_in = max(0, min(c_in, limit - (0 if g_estimated else g_in)))
            g_in = max(0, min(g_in, limit - c_in))
        u.in_vram += g_in + c_in
        u.requested += sum(_requested_or_resident(c) for c in details)
        u.evicted += sum(c.evicted or 0 for c in details)
        u.in_vram_estimated |= g_estimated or c_estimated
        u.in_vram_capped |= any(c.in_vram_capped for c in details)
        if kfd:
            beyond = max(0, kfd - (_requested_or_resident(linked) if linked else 0))
            u.kind = "compute"
            u.rocm_allocated += kfd
            u.requested += beyond
            u.compute_in_vram += c_in
            u.compute_evicted += (linked.evicted or 0) if linked else 0
            if linked is None:
                u.in_vram_known = False
            u.compute_details.append(ComputeUsage(
                gpu=address, kfd_vram=kfd,
                linked_client=linked.client_id if linked else None, link=link,
                beyond_client=beyond, in_vram=c_in, estimated=c_estimated,
                in_vram_known=linked is not None))
            if address and linked is not None:
                compute_by_device[address] = compute_by_device.get(address, 0) + c_in
        u.evicted_known &= all(c.evicted is not None for c in details)
        u.clients += len(details)
        u.client_details.extend(details)
        if address and address not in u.gpus:
            u.gpus.append(address)
        u.parts.append(GpuPart(
            gpu=address, in_vram=g_in + c_in, compute_in_vram=c_in,
            evicted=sum(c.evicted or 0 for c in details),
            compute_evicted=(linked.evicted or 0) if linked else 0,
            rocm_allocated=kfd, in_vram_estimated=g_estimated or c_estimated,
            graphics_estimated=g_estimated,
            evicted_known=all(c.evicted is not None for c in details),
            in_vram_known=not (kfd and linked is None), clients=len(details)))

    for pid, u in usage.items():
        u.name = _safe(_program_name(pid))
        u.description = _describe(u.name)

    # Total compute memory on a GPU cannot exceed what the GPU uses either.
    compute_by_device = {a: min(v, in_use(a)) for a, v in compute_by_device.items()}
    warnings = []
    warnings += [f"Unrecognised kernel value skipped: {b}"
                 for b in sorted(bad_units)]
    if without_stats and not clients:
        warnings.append("This kernel does not report per-program GPU memory, "
                        "so only compute (ROCm) programs are listed. A newer "
                        "kernel is needed for the full table.")
    elif without_stats:
        warnings.append(f"{without_stats} GPU client(s) report no usable memory "
                        "statistics and are not included in the table.")
    ordered = sorted(usage.values(), key=lambda u: (u.kind != "compute", -_row_size(u)))
    return ordered, compute_by_device, warnings


# Model servers. Recognised by process name; their listening TCP ports come
# from /proc/<pid>/fd and /proc/net/tcp{,6} (root is needed for other users'
# processes); their models are asked over HTTP: Ollama /api/ps (loaded models
# with sizes), the others the OpenAI-compatible /v1/models. Children of a
# server (Ollama's internal llama-server, worker processes) are skipped.
MODEL_SERVERS = (       # label, executable names, Python module prefixes, API
    ("Ollama", ("ollama",), (), "ollama"),
    ("llama.cpp", ("llama-server", "llamafile"), (), "openai"),
    ("vLLM", ("vllm",), ("vllm.",), "openai"),
    ("SGLang", (), ("sglang.",), "openai"),
    ("LM Studio", ("lm studio", "lm-studio", "lmstudio", "llmster"), (), "openai"),
    ("KoboldCpp", ("koboldcpp",), (), "openai"),
    ("LocalAI", ("local-ai",), (), "openai"),
    ("TGI", ("text-generation-launcher", "text-generation-router"), (), "openai"),
)
SERVER_TIMEOUT = 0.5    # seconds per request; requests run in parallel
SERVER_PORTS_TRIED = 3  # a server listening on many ports: ask at most this many


def _server_kind(pid: str) -> tuple[str, str] | None:
    """(label, API) if the process is a known model server."""
    argv = [a for a in (_read(f"/proc/{pid}/cmdline") or "").split("\0") if a]
    names = {_comm(pid).lower()} | ({os.path.basename(argv[0]).lower()} if argv else set())
    for label, executables, modules, api in MODEL_SERVERS:
        for name in names:
            for exe in executables:
                if (name == exe or name.startswith((exe + "-", exe + "."))
                        or (exe == "llamafile" and name.endswith(".llamafile"))):
                    return label, api
        if any(arg.startswith(modules) for arg in argv[1:]) if modules else False:
            return label, api
    return None


def _parent(pid: str) -> str | None:
    stat = _read(f"/proc/{pid}/stat") or ""
    fields = stat[stat.rfind(")") + 2:].split()
    return fields[1] if len(fields) > 1 and fields[1].isdigit() else None


def _decode_ip(hex_ip: str) -> str:
    """An address from /proc/net/tcp{,6}: hex words in little-endian order."""
    raw = bytes.fromhex(hex_ip)
    if len(raw) == 4:
        return socket.inet_ntop(socket.AF_INET, raw[::-1])
    words = b"".join(raw[i:i + 4][::-1] for i in range(0, 16, 4))
    return socket.inet_ntop(socket.AF_INET6, words)


def _listening_sockets() -> dict[str, tuple[str, int]]:
    """socket inode -> (address, port) for every listening TCP socket."""
    out: dict[str, tuple[str, int]] = {}
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in (_read(path) or "").splitlines()[1:]:
            f = line.split()
            if len(f) > 9 and f[3] == "0A":           # 0A = LISTEN
                ip, port = f[1].split(":")
                try:
                    out[f[9]] = (_decode_ip(ip), int(port, 16))
                except ValueError:
                    continue
    return out


def _ports_of(pid: str, listening: dict[str, tuple[str, int]]) -> list[tuple[str, int]]:
    ports = []
    for fd in glob.glob(f"/proc/{pid}/fd/*"):
        try:
            link = os.readlink(fd)
        except OSError:
            continue
        if link.startswith("socket:[") and link[8:-1] in listening:
            ports.append(listening[link[8:-1]])
    return sorted(set(ports), key=lambda x: x[1])


def _url_host(ip: str) -> str:
    if ip in ("0.0.0.0", "127.0.0.1"):
        return "127.0.0.1"
    if ip in ("::", "::1"):
        return "[::1]"
    return f"[{ip}]" if ":" in ip else ip


def _ask(url: str) -> tuple[str, object]:
    """GET url and parse JSON. Status: "ok", "no reply", "refused",
    "HTTP <code>", "unexpected reply" (not JSON) or "unreachable"."""
    try:
        with urllib.request.urlopen(url, timeout=SERVER_TIMEOUT) as response:
            return "ok", json.load(response)
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}", None
    except socket.timeout:
        return "no reply", None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return "refused", None
        if isinstance(exc.reason, socket.timeout):
            return "no reply", None
        return "unreachable", None
    except (ValueError, OSError):             # invalid JSON, broken connection
        return "unexpected reply", None


def _query_server(kind: str, api: str, pid: int | None,
                  ports: list[tuple[str, int]]) -> ModelServer:
    if not ports:
        return ModelServer(kind, pid, "", "port unknown")
    result = None
    for ip, port in ports[:SERVER_PORTS_TRIED]:
        base = f"http://{_url_host(ip)}:{port}"
        status, data = _ask(base + ("/api/ps" if api == "ollama" else "/v1/models"))
        address = f"{ip}:{port}" if ":" not in ip else f"[{ip}]:{port}"
        if status == "ok" and api == "ollama" and isinstance(data, dict) \
                and isinstance(data.get("models", []), list):
            loaded = [OllamaModel(_safe(str(m.get("name", "?"))), int(m.get("size", 0)),
                                  int(m.get("size_vram", 0)))
                      for m in data.get("models") or [] if isinstance(m, dict)]
            return ModelServer(kind, pid, address, "ok", [x.name for x in loaded], loaded)
        if status == "ok" and api == "openai" and isinstance(data, dict) \
                and isinstance(data.get("data"), list):
            names = [_safe(str(m.get("id", "?"))) for m in data["data"] if isinstance(m, dict)]
            return ModelServer(kind, pid, address, "ok", names)
        if status == "ok":
            status = "unexpected reply"
        result = result or ModelServer(kind, pid, address, status)
    return result


def find_model_servers() -> list[ModelServer]:
    """Running model servers and their models. Ollama is also looked for at
    OLLAMA_HOST (default 127.0.0.1:11434) when no Ollama process is visible,
    and then listed only if something answers there."""
    listening = _listening_sockets()
    jobs: list[tuple[str, str, int | None, list[tuple[str, int]]]] = []
    for d in glob.glob("/proc/[0-9]*"):
        pid = os.path.basename(d)
        kind = _server_kind(pid)
        if kind is None:
            continue
        parent = _parent(pid)
        if parent and _server_kind(parent) is not None:
            continue                          # a server's child or worker
        jobs.append((kind[0], kind[1], int(pid), _ports_of(pid, listening)))
    fallback = not any(kind == "Ollama" for kind, _, _, _ in jobs)
    if fallback:
        host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
        host = host.split("://", 1)[-1].rstrip("/")
        name, _, port = host.rpartition(":")
        if port.isdigit():
            jobs.append(("Ollama", "ollama", None, [(name or "127.0.0.1", int(port))]))
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        servers = list(pool.map(lambda job: _query_server(*job), jobs))
    return [x for x in servers
            if x.pid is not None or x.status not in ("refused", "unreachable")]



def device_metrics(dev: GpuDevice) -> dict[str, float]:
    """The values of one GPU that the live view averages."""
    m = {"vram_used": float(dev.vram_used)}
    if dev.sclk_mhz is not None:
        m["sclk"] = dev.sclk_mhz
    if dev.mclk_mhz is not None:
        m["mclk"] = dev.mclk_mhz
    if dev.gtt_total:
        m["gtt_used"] = float(dev.gtt_used)
    if dev.asleep:
        m["busy"] = 0.0                       # a suspended GPU is idle by definition
    elif dev.busy_percent is not None:
        m["busy"] = float(dev.busy_percent)
    p = dev.power
    watts = (p.average_w if p.average_w is not None else p.input_w) if p else dev.power_watts
    if watts is not None:
        m["power"] = watts
    for label, celsius in ([(x.label, x.celsius) for x in dev.temperature_sensors]
                           or dev.temperatures):
        m[f"temp:{label}"] = celsius
    return m


class History:
    """Samples of each GPU's metrics over the longest averaging window."""

    def __init__(self, windows: tuple[tuple[int, str], ...] = AVERAGE_WINDOWS):
        self.windows = windows
        self.horizon = max(seconds for seconds, _ in windows)
        self.samples: dict[str, deque] = {}   # address -> deque of (time, metrics)
        self.started: float | None = None

    def add(self, snap: Snapshot, now: float) -> None:
        """Record a snapshot taken at monotonic time now, and drop old samples."""
        if self.started is None:
            self.started = now
        for dev in snap.devices:
            queue = self.samples.setdefault(dev.address, deque())
            queue.append((now, device_metrics(dev)))
            while queue and queue[0][0] < now - self.horizon:
                queue.popleft()

    def averages(self, address: str, now: float) -> dict[str, dict[str, float]]:
        """metric -> window label -> mean of the samples inside that window.
        A metric missing from a sample (e.g. no temperature while asleep) is
        averaged over the samples that have it."""
        out: dict[str, dict[str, float]] = {}
        samples = self.samples.get(address, ())
        for seconds, label in self.windows:
            sums: dict[str, float] = {}
            counts: dict[str, int] = {}
            for when, metrics in samples:
                if when >= now - seconds:
                    for key, value in metrics.items():
                        sums[key] = sums.get(key, 0.0) + value
                        counts[key] = counts.get(key, 0) + 1
            for key, total in sums.items():
                out.setdefault(key, {})[label] = total / counts[key]
        return out

    def covered(self, now: float) -> float:
        return 0.0 if self.started is None else now - self.started

    def apply(self, snap: Snapshot, now: float) -> None:
        """Record snap and attach the averages to it."""
        self.add(snap, now)
        snap.averages = {d.address: self.averages(d.address, now) for d in snap.devices}
        snap.averaged_seconds = self.covered(now)


def vram_limits(devices: list[GpuDevice]) -> tuple[dict[str, int], dict[str, int]]:
    """Per GPU: the most VRAM a client on it can be reported with (size), and
    the most it can currently hold (use). Normally that GPU's own figures; for
    GPUs in one XGMI hive, the hive's combined figures (peer VRAM mappings)."""
    sizes: dict[str, int] = {}
    uses: dict[str, int] = {}
    for d in devices:
        members = [m for m in devices if d.xgmi_hive and m.xgmi_hive == d.xgmi_hive] or [d]
        sizes[d.address] = sum(m.vram_total for m in members)
        uses[d.address] = sum(m.vram_used for m in members)
    return sizes, uses


def collect(gpus: list[PciGpu]) -> Snapshot:
    devices = [read_device(g) for g in gpus]
    capacities, used = vram_limits(devices)
    processes, compute_by_device, warnings = collect_processes(capacities, used)
    is_root = os.geteuid() == 0
    for d in devices:                         # the display caps these; say so if it had to
        if d.vram_used_reported is not None and d.vram_used_reported > d.vram_total:
            warnings.append(f"GPU {d.address} reports {_fmt(d.vram_used_reported)} of VRAM "
                            f"used on a {_fmt(d.vram_total)} card, which is impossible; "
                            "shown as full.")
        if d.gtt_total and d.gtt_used_reported is not None and d.gtt_used_reported > d.gtt_total:
            warnings.append(f"GPU {d.address} reports {_fmt(d.gtt_used_reported)} of GTT used "
                            f"out of {_fmt(d.gtt_total)}, which is impossible.")
    if not is_root:
        warnings.append("Not running as root: graphics memory of other users' "
                        "programs may be missing (Ollama's compute memory is "
                        "still shown). Use sudo for the full picture.")
    servers = find_model_servers()
    ollama = next((x for x in servers if x.kind == "Ollama" and x.status == "ok"), None)
    return Snapshot(datetime.now(), devices, processes, ollama.loaded if ollama else [],
                    is_root, warnings, compute_by_device,
                    ollama_running=ollama is not None, servers=servers)


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
MIN_WIDTH, MAX_WIDTH = 40, 110   # columns; output is cut to fit, never wrapped
METRIC_LABEL_WIDTH, METRIC_VALUE_WIDTH = 11, 9
ROW_SWAP_MARGIN = 0.10           # live view: a row passes the one above only when 10% larger
PROGRAM_ROWS_MIN = 3             # live view: the program table always has room for 3 rows
LAYOUT_SETTLE_SECONDS = 60       # live view: a part keeps the largest size it needed this long

CHARS_UNICODE = dict(tl="╭", tr="╮", bl="╰", br="╯", h="─", v="│",
                     full="█", desk="▓", evict="▒", free="░", tiny="▏",
                     ok="✔", warn="⚠", bad="✖", info="•", dot="·", deg="°C",
                     approx="≈", ellipsis="…", dash="–", atleast="≥")
CHARS_ASCII = dict(tl="+", tr="+", bl="+", br="+", h="-", v="|",
                   full="#", desk="=", evict="~", free=".", tiny="|",
                   ok="OK", warn="!!", bad="XX", info="-", dot="|", deg="C",
                   approx="~", ellipsis="~", dash="-", atleast=">=")
LIVE_START = "\x1b[?1049h\x1b[?25l"   # alternate screen, cursor hidden
LIVE_END = "\x1b[?25h\x1b[?1049l"     # cursor shown, normal screen back
_ANSI_SPLIT = re.compile(r"(\x1b\[[0-9;]*m)")


class Theme:
    """Characters and colours; chosen automatically for the terminal."""
    CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32",
             "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36",
             "gray": "90"}

    def __init__(self, color: bool, unicode: bool):
        self.color = color
        self.ch = CHARS_UNICODE if unicode else CHARS_ASCII

    @classmethod
    def auto(cls, stream=None) -> Theme:
        stream = stream or sys.stdout
        tty = stream.isatty()
        color = (tty and not os.environ.get("NO_COLOR")
                 and os.environ.get("TERM") != "dumb")
        unicode = "utf" in (getattr(stream, "encoding", "") or "").lower()
        return cls(color, unicode)

    def paint(self, text: str, *styles: str) -> str:
        if not self.color or not styles or not text:
            return text
        return f"\x1b[{';'.join(self.CODES[s] for s in styles)}m{text}\x1b[0m"


# --- text measured in terminal columns, not characters ----------------------
def _char_width(c: str) -> int:
    """Columns one character occupies: 0 for combining and format characters,
    2 for East Asian wide/fullwidth ones (most CJK, emoji), else 1. Ambiguous
    characters (box drawing, blocks, "·") count as 1, as in most terminals."""
    if unicodedata.combining(c) or unicodedata.category(c) in ("Mn", "Me", "Cf"):
        return 0
    return 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1


def _width(text: str) -> int:
    """Terminal columns of text; colour codes take none."""
    return sum(_char_width(c) for c in ANSI_RE.sub("", text))


def _fit(text: str, width: int, ellipsis: str = "…") -> str:
    """Text cut to at most width columns, ending in an ellipsis if cut.
    Colour codes are kept, and reset if the cut falls inside a coloured part."""
    if _width(text) <= width:
        return text
    budget = max(0, width - _width(ellipsis))
    out: list[str] = []
    used = 0
    for part in _ANSI_SPLIT.split(text):
        if _ANSI_SPLIT.fullmatch(part):
            out.append(part)
            continue
        for c in part:
            w = _char_width(c)
            if used + w > budget:
                reset = "\x1b[0m" if "\x1b[" in text else ""
                return "".join(out) + (ellipsis if width >= _width(ellipsis) else "") + reset
            out.append(c)
            used += w
    return "".join(out)


def _ljust(text: str, width: int, ellipsis: str = "…") -> str:
    """Exactly width columns, left-aligned."""
    text = _fit(text, width, ellipsis)
    return text + " " * (width - _width(text))


def _rjust(text: str, width: int, ellipsis: str = "…") -> str:
    """Exactly width columns, right-aligned."""
    text = _fit(text, width, ellipsis)
    return " " * (width - _width(text)) + text


# --- numbers with a fixed width, so a live view does not jump ---------------
def _fmt(nbytes: int) -> str:
    """Size for prose: '20.8 GiB', '131 MiB'."""
    return f"{nbytes / GIB:.1f} GiB" if nbytes >= GIB else f"{nbytes / MIB:.0f} MiB"


def _size(nbytes: int) -> str:
    """Size in a fixed 8 columns: '20.8 GiB', ' 131 MiB', '   8 MiB'."""
    return f"{_fmt(nbytes):>8}"


def _temp_color(sensor: TempSensor) -> str | None:
    if sensor.crit is None:
        return None
    if sensor.celsius >= sensor.crit:
        return "red"
    return "yellow" if sensor.celsius >= sensor.crit - TEMP_WARN_MARGIN else None


def _cells(nbytes: int, total: int, width: int) -> int:
    return min(width, round(nbytes / total * width)) if total else 0


def _box(title: str, lines: list[str], width: int, t: Theme,
         color: str = "blue") -> list[str]:
    ch, inner = t.ch, width - 4
    title = f" {_fit(title, max(0, width - 6), ch['ellipsis'])} "
    out = [t.paint(ch["tl"] + ch["h"], color) + t.paint(title, "bold")
           + t.paint(ch["h"] * max(0, width - 3 - _width(title)) + ch["tr"], color)]
    for line in lines:
        out.append(t.paint(ch["v"], color) + " " + _ljust(line, inner, ch["ellipsis"])
                   + " " + t.paint(ch["v"], color))
    out.append(t.paint(ch["bl"] + ch["h"] * (width - 2) + ch["br"], color))
    return out


def _metric_lines(dev: GpuDevice, averages: dict[str, dict[str, float]],
                  covered: float, t: Theme) -> list[str]:
    """A small table: each metric now and, in the live view, its averages."""
    ch, lw, vw = t.ch, METRIC_LABEL_WIDTH, METRIC_VALUE_WIDTH
    windows = [label for _, label in AVERAGE_WINDOWS] if averages else []
    head = " " * lw + "".join(_rjust(x, vw) for x in ["now"] + windows)
    if windows:
        longest = AVERAGE_WINDOWS[-1][0]
        minutes, seconds = divmod(int(min(covered, longest)), 60)
        head += (f"   averages over {minutes}:{seconds:02d} of {longest // 60}:00 min so far"
                 if covered < longest else "   averages complete")
    lines = [t.paint(head, "dim")]

    def row(label: str, key: str, now: float | None, fmt, note: str = "",
            color: str | None = None) -> None:
        if now is None and key not in averages:
            return
        first = _rjust(fmt(now) if now is not None else ch["dash"], vw)
        cells = [t.paint(first, color) if color and now is not None else first]
        cells += [_rjust(fmt(averages[key][w]) if w in averages.get(key, {}) else ch["dash"], vw)
                  for w in windows]
        lines.append(t.paint(_ljust(label, lw), "bold") + "".join(cells)
                     + (t.paint("   " + note, "dim") if note else ""))

    def pct(v: float) -> str:
        return f"{v:.0f}%"

    def watts(v: float) -> str:
        return f"{v:.0f} W"

    def deg(v: float) -> str:
        return f"{v:.0f}{ch['deg']}"

    row("GPU busy", "busy", 0.0 if dev.asleep else dev.busy_percent, pct,
        "asleep (runtime-suspended): sensors not read, so it can stay asleep"
        if dev.asleep else "")
    p = dev.power
    if p is not None:
        kind, now = (("average", p.average_w) if p.average_w is not None
                     else ("instantaneous", p.input_w))
        note = f"SoC, {kind} sensor" + (f" {ch['dot']} limit {p.cap_w:.0f} W" if p.cap_w else "")
        row("Power", "power", now, watts, note)
    else:
        row("Power", "power", dev.power_watts, watts)

    def mhz(v: float) -> str:
        return f"{v:.0f} MHz"

    row("GPU clock", "sclk", dev.sclk_mhz, mhz, "shader (gfx/compute) clock")
    row("Mem clock", "mclk", dev.mclk_mhz, mhz)
    shown = set()
    for x in dev.temperature_sensors:
        shown.add(x.label)
        row(x.name[:1].upper() + x.name[1:], f"temp:{x.label}", x.celsius, deg,
            f"limit {x.crit:.0f}{ch['deg']}" if x.crit is not None else "", _temp_color(x))
    for label, celsius in ([] if dev.temperature_sensors else dev.temperatures):
        shown.add(label)
        row(label[:1].upper() + label[1:], f"temp:{label}", celsius, deg)
    for key in sorted(k for k in averages if k.startswith("temp:") and k[5:] not in shown):
        name = TEMP_NAMES.get(key[5:], key[5:])          # sensor not readable right now
        row(name[:1].upper() + name[1:], key, None, deg)
    row("VRAM used", "vram_used", float(dev.vram_used), _fmt)
    if dev.gtt_total:
        row("GTT used", "gtt_used", float(dev.gtt_used), _fmt,
            "system memory mapped for the GPU, incl. buffers placed there on purpose")
    return lines


SERVER_STATUS_TEXT = {
    "no reply": f"listening, but no reply within {SERVER_TIMEOUT:g} s",
    "refused": "connection refused",
    "unexpected reply": "unexpected reply (not this server's API?)",
    "unreachable": "unreachable",
    "port unknown": "running; port not visible (run as root)",
}


def _server_lines(servers: list[ModelServer], width: int, t: Theme) -> list[str]:
    """Box content: one line per server, and per loaded Ollama model."""
    ch, lw = t.ch, METRIC_LABEL_WIDTH
    lines = []
    for x in servers:
        head = t.paint(_ljust(x.kind, lw), "bold") + _ljust(x.address or "-", 22) + " "
        if x.status != "ok":
            lines.append(head + t.paint(SERVER_STATUS_TEXT.get(x.status, x.status), "yellow"))
        elif x.loaded:
            for m in x.loaded:
                pct = f" ({m.size_vram / m.size * 100:>3.0f}%)" if m.size else ""
                lines.append(head + f"{m.name} {ch['dot']} {_size(m.size)} total "
                                    f"{ch['dot']} {_size(m.size_vram)} on GPU{pct}")
                head = " " * (lw + 23)
        elif x.models:
            lines.append(head + ", ".join(x.models))
        else:
            lines.append(head + t.paint("running, no model loaded", "dim"))
    return lines


def _gpu_section(dev: GpuDevice, compute: int | None, width: int, t: Theme, stamp: str,
                 averages: dict[str, dict[str, float]] | None = None,
                 covered: float = 0.0, compact: bool = False) -> list[str]:
    ch, inner = t.ch, width - 4
    total, used, free = dev.vram_total, dev.vram_used, dev.vram_free
    tail = f"  {_size(used)} / {_size(total)} used"
    bw = max(4, inner - len("VRAM ") - len(tail))
    compute = min(compute or 0, used)
    c = _cells(compute, total, bw)
    d = max(0, _cells(used, total, bw) - c)
    bar = (t.paint(ch["full"] * c, "magenta") + t.paint(ch["desk"] * d, "cyan")
           + t.paint(ch["free"] * max(0, bw - c - d), "gray"))
    if compute:
        legend = (t.paint(ch["full"], "magenta") + f" compute (ROCm) {_size(compute)}  "
                  + t.paint(ch["desk"], "cyan") + f" graphics & driver {_size(used - compute)}  "
                  + t.paint(ch["free"], "gray") + f" free {_size(free)}")
    else:
        legend = (t.paint(ch["desk"], "cyan") + f" used {_size(used)}  "
                  + t.paint(ch["free"], "gray") + f" free {_size(free)}")
    lines = [t.paint("VRAM ", "bold") + bar + tail, "     " + legend] + ([] if compact else [""])
    averages = averages or {}

    lines += _metric_lines(dev, averages, covered, t)

    title = f"GPU {dev.address} {ch['dot']} {dev.name}"
    if stamp:
        title += f" {ch['dot']} {stamp}"
    return _box(title, lines, width, t)


def _parts(p: ProcessUsage) -> list[GpuPart]:
    """A process's per-GPU parts; a single part from its totals if it has
    none (a ProcessUsage built by hand rather than by collect_processes)."""
    return p.parts or [GpuPart(
        gpu=p.gpus[0] if len(p.gpus) == 1 else None, in_vram=p.in_vram,
        compute_in_vram=p.compute_in_vram, evicted=p.evicted,
        compute_evicted=p.compute_evicted, rocm_allocated=p.rocm_allocated,
        in_vram_estimated=p.in_vram_estimated, graphics_estimated=p.graphics_estimated,
        evicted_known=p.evicted_known, in_vram_known=p.in_vram_known, clients=p.clients)]


def _part_size(part: GpuPart) -> int:
    return max(part.in_vram + part.evicted,
               part.rocm_allocated if not part.in_vram_known else 0)


TableRow = tuple  # (line, ProcessUsage or None, bytes in VRAM shown on that line)


def _engine_percent(p: ProcessUsage, part: GpuPart, before: dict, elapsed: float) -> str:
    """Shader time of one program on one GPU since the last refresh, as text."""
    if part.rocm_allocated:
        return "n/a"                          # ROCm work is not counted per program
    busy = None
    for c in p.client_details:
        if part.gpu is not None and c.gpu != part.gpu:
            continue                          # another GPU's row (unknown GPU: all)
        old = before.get((c.gpu, c.client_id))
        if old is not None:                   # a client seen in both refreshes
            delta = sum(c.engine_ns.get(e, 0) - old.get(e, 0) for e in SHADER_ENGINES)
            busy = (busy or 0) + max(0, delta)
    if busy is None or elapsed <= 0:
        return "-"
    return f"{min(999.0, busy / (elapsed * 1e9) * 100):.0f}%"


def _process_table(processes: list[ProcessUsage], devices: list[GpuDevice], width: int,
                   t: Theme, previous: Snapshot | None = None, elapsed: float = 0.0,
                   show_engine: bool = False) -> tuple[list[str], list[TableRow], list[str]]:
    """Table head, rows and foot. With one GPU: one row per program. With
    several: one row per program and GPU, with a GPU column. Every bar is the
    share of the VRAM of the GPU in its row; a row whose GPU is unknown gets no
    bar rather than a misleading one."""
    ch, ell, nw = t.ch, t.ch["ellipsis"], 20
    multi = len(devices) > 1
    sizes = {d.address: d.vram_total for d in devices}
    position = {d.address: i for i, d in enumerate(devices)}   # same order as the GPU boxes
    gw = 7 if multi else 0                    # "83:00.0": bus:device.function
    dw = 22 if width >= 96 + (gw + 1 if gw else 0) else 0
    fixed = (1 + nw + 1 + 8 + 2 + (gw + 1 if gw else 0) + (dw + 1 if dw else 0)
             + 9 + 1 + 9 + 2 + (6 if show_engine else 0))
    before = ({(c.gpu, c.client_id): c.engine_ns for q in previous.processes
               for c in q.client_details} if previous is not None else {})
    bw = max(4, width - fixed - 1)
    gpu_head = _ljust("GPU", gw) + " " if gw else ""
    desc_head = _ljust("WHAT IT IS", dw) + " " if dw else ""
    head = [t.paint(f" {_ljust('PROGRAM', nw)} {'PID':>8}  {gpu_head}{desc_head}"
                    f"{'IN VRAM':>9} {'EVICTED':>9}{'  GPU%' if show_engine else ''}  "
                    "USAGE", "bold"),
            t.paint(" " + ch["h"] * (width - 2), "gray")]
    rows: list[TableRow] = []
    sum_vram = sum_evicted = 0
    sum_estimated = False
    evicted_known = evicted_unknown = False   # did rows report eviction, or not?
    for p in processes:
        for part in sorted(_parts(p), key=lambda x: position.get(x.gpu, len(position))):
            if _part_size(part) < MIB:
                continue
            graphics = max(0, part.in_vram - part.compute_in_vram)
            sum_vram += graphics
            sum_evicted += max(0, part.evicted - part.compute_evicted)
            sum_estimated |= part.graphics_estimated
            if graphics or part.clients:
                evicted_known |= part.evicted_known
                evicted_unknown |= not part.evicted_known
            scale = sizes.get(part.gpu) if part.gpu else (
                devices[0].vram_total if not multi and devices else 0)
            bar = ""
            if scale:
                cc = _cells(part.compute_in_vram, scale, bw)
                gc = min(_cells(graphics, scale, bw), bw - cc)
                ec = min(_cells(part.evicted, scale, bw), bw - cc - gc)
                if part.evicted >= EVICTED_HIGHLIGHT:
                    ec = max(ec, 1)
                bar = t.paint(ch["full"] * cc, "magenta") + t.paint(ch["desk"] * gc, "cyan")
                if not cc and not gc and part.in_vram:
                    bar = t.paint(ch["tiny"], "magenta" if part.compute_in_vram else "cyan")
                bar += t.paint(ch["evict"] * ec, "yellow")
            ev_text = ("n/a" if not part.evicted_known
                       else _fmt(part.evicted) if part.evicted else "-")
            ev = _rjust(ev_text, 9)
            if part.evicted >= EVICTED_HIGHLIGHT:
                ev = t.paint(ev, "yellow")
            name = (_ljust(_fit(p.name, nw - 1, ell) + "*", nw) if p.attribution_guessed
                    else _ljust(p.name, nw, ell))
            gpu = _ljust(part.gpu[5:] if part.gpu else "?", gw) + " " if gw else ""
            desc = _ljust(p.description, dw, ell) + " " if dw else ""
            vram = _rjust("n/a" if not part.in_vram_known and not part.in_vram else
                          (ch["approx"] if part.in_vram_estimated else "")
                          + _fmt(part.in_vram), 9)
            engine = (" " + _rjust(_engine_percent(p, part, before, elapsed), 5)
                      if show_engine else "")
            rows.append((f" {name} {p.pid:>8}  {gpu}{desc}{vram} {ev}{engine}  {bar}", p,
                         part.in_vram))
    if not rows:
        rows.append((t.paint(" (no programs are using GPU memory)", "dim"), None, 0))
    total = _rjust((ch["approx"] if sum_estimated else "") + _fmt(sum_vram), 9)
    label = _ljust("graphics memory total",
                   nw + 1 + 8 + 2 + (gw + 1 if gw else 0) + (dw + 1 if dw else 0) - 1)
    ev_total = ("n/a" if evicted_unknown and not evicted_known       # nothing reported
                else (ch["atleast"] if evicted_unknown else "") + _fmt(sum_evicted))
    foot = [t.paint(" " + ch["h"] * (width - 2), "gray"),
            t.paint(f" {label} {total} {_rjust(ev_total, 9)}", "bold")]
    return head, rows, foot


def _status_lines(snap: Snapshot, previous: Snapshot | None, width: int,
                  t: Theme) -> list[str]:
    """Status messages. They state what was measured, and say "can" where the
    consequence depends on what programs do next."""
    lines: list[str] = []
    inner = width - 4
    deg = t.ch["deg"]

    def say(symbol: str, color: str, text: str) -> None:
        prefix = t.ch[symbol] + " "
        parts = textwrap.wrap(text, max(10, inner - len(prefix))) or [""]
        lines.append(t.paint(t.ch[symbol], color, "bold") + " " + parts[0])
        lines.extend(" " * len(prefix) + p for p in parts[1:])

    for dev in snap.devices:
        free = dev.vram_free
        pct = free / dev.vram_total * 100 if dev.vram_total else 0.0
        prefix = f"GPU {dev.address}: " if len(snap.devices) > 1 else ""
        if free < CRITICAL_FREE:
            say("bad", "red", f"{prefix}only {_fmt(free)} VRAM free ({pct:.0f}%). If a "
                "program needs more, the driver has to move buffers out of VRAM, "
                "which can cause stutter.")
        elif free < TIGHT_FREE:
            say("warn", "yellow", f"{prefix}{_fmt(free)} VRAM free ({pct:.0f}%): "
                "little headroom left.")
        else:
            say("ok", "green", f"{prefix}{_fmt(free)} VRAM free ({pct:.0f}%).")
        for x in dev.temperature_sensors:
            if x.crit is None:
                continue
            if x.celsius >= x.crit:
                say("bad", "red", f"{prefix}GPU {x.name} at {x.celsius:.0f}{deg}, at or "
                    f"above its critical limit of {x.crit:.0f}{deg}.")
            elif x.celsius >= x.crit - TEMP_WARN_MARGIN:
                say("warn", "yellow", f"{prefix}GPU {x.name} at {x.celsius:.0f}{deg}, "
                    f"within {TEMP_WARN_MARGIN:.0f}{deg} of its critical limit "
                    f"({x.crit:.0f}{deg}).")

    evicted_now = sum(p.graphics_evicted for p in snap.processes)
    partly = "at least " if any(not p.evicted_known for p in snap.processes
                                if p.client_details) else ""
    if previous is not None:
        # Compare each GPU client with itself, so programs that start, exit
        # or are attributed differently cannot change the result.
        before = {(c.gpu, c.client_id): c.evicted for q in previous.processes
                  for c in q.client_details if c.evicted is not None}
        grown = grew = 0
        for q in snap.processes:
            for c in q.client_details:
                old = before.get((c.gpu, c.client_id))
                if c.evicted is not None and old is not None and c.evicted > old:
                    grown += c.evicted - old
                    grew += 1
        if grown >= EVICTED_HIGHLIGHT:
            say("bad", "red", f"Evicted memory of {grew} GPU client(s) present in both "
                f"refreshes grew by {_fmt(grown)}: buffers were moved out of VRAM, or new "
                "ones were placed elsewhere. This can cause stutter.")
    sizes = {d.address: d.vram_total for d in snap.devices}
    for p in snap.processes:                  # requests larger than the card are real
        wanted: dict[str | None, int] = {}
        for c in p.client_details:
            wanted[c.gpu] = wanted.get(c.gpu, 0) + (c.requested or 0)
        for c in p.compute_details:
            wanted[c.gpu] = wanted.get(c.gpu, 0) + c.beyond_client
        for gpu, amount in wanted.items():
            card = sizes.get(gpu) if gpu else (max(sizes.values()) if len(sizes) == 1 else None)
            if card and amount > card:
                where = f"GPU {gpu}: " if len(snap.devices) > 1 and gpu else ""
                say("info", "cyan", f"{where}{p.name} has {_fmt(amount)} of buffers that "
                    f"prefer VRAM on a {_fmt(card)} card, more than fits: the rest is "
                    "placed elsewhere or has no memory allocated yet.")
    for p in snap.processes:                  # ROCm memory without a known location
        hidden = [c for c in p.compute_details if not c.in_vram_known]
        unmapped = sum(c.beyond_client for c in p.compute_details if c.in_vram_known)
        if hidden:
            say("info", "cyan", f"{p.name}: ROCm allocated "
                f"{_fmt(sum(c.kfd_vram for c in hidden))}, but its GPU address space is "
                "not visible, so how much of it is in VRAM is not known.")
        if unmapped >= EVICTED_HIGHLIGHT:
            say("info", "cyan", f"{p.name}: at least {_fmt(unmapped)} of its ROCm "
                "allocations are not mapped to the GPU; the kernel does not report "
                "where they are, so they are not counted as in VRAM.")
    graphics = [p for p in snap.processes
                if p.graphics_in_vram + p.graphics_evicted >= MIB]
    if evicted_now > EVICTED_NOTICE and graphics:
        worst = max(graphics, key=lambda p: p.graphics_evicted)
        fix = ("logging out and back in resets it" if worst.name == "gnome-shell"
               else f"restarting {worst.name} resets it")
        say("warn", "yellow", f"{partly}{_fmt(evicted_now)} of graphics memory is evicted: "
            f"it prefers VRAM but is placed elsewhere, mostly from {worst.name}. It can "
            f"move back when used and space allows; {fix}.")
    for w in snap.warnings:
        say("warn", "yellow", w)
    return lines


def layout_needs(snap: Snapshot, previous: Snapshot | None, width: int,
                 theme: Theme) -> dict[str, int]:
    """Lines the program rows and status messages need for this snapshot."""
    width = max(MIN_WIDTH, min(width, MAX_WIDTH))
    _, rows, _ = _process_table(snap.processes, snap.devices, width, theme)
    return {"rows": len(rows), "status": len(_status_lines(snap, previous, width, theme))}


class StickyHeights:
    """Live view: each part keeps the largest size it needed during the last
    LAYOUT_SETTLE_SECONDS, so programs or messages that come and go do not
    make everything below them jump; it shrinks once the space stays unused."""

    def __init__(self, settle: float = LAYOUT_SETTLE_SECONDS):
        self.settle = settle
        self.seen: deque = deque()            # (time, {part: lines})

    def update(self, needed: dict[str, int], now: float) -> dict[str, int]:
        self.seen.append((now, needed))
        while self.seen and self.seen[0][0] < now - self.settle:
            self.seen.popleft()
        return {part: max(n.get(part, 0) for _, n in self.seen) for part in needed}


def stable_order(processes: list[ProcessUsage],
                 previous_order: list[int]) -> list[ProcessUsage]:
    """Live view: keep last frame's row order; a program only moves above the
    one before it when it is clearly larger (ROW_SWAP_MARGIN), so rows of
    similar size do not swap back and forth on every refresh."""
    size = _row_size
    rank = {pid: i for i, pid in enumerate(previous_order)}
    rows = sorted(processes, key=lambda p: (p.kind != "compute",
                                            rank.get(p.pid, len(rank)), -size(p)))
    swapped = True
    while swapped:                            # ends: each swap fixes a size inversion
        swapped = False
        for i in range(len(rows) - 1):
            a, b = rows[i], rows[i + 1]
            if a.kind == b.kind and size(b) > size(a) * (1 + ROW_SWAP_MARGIN) + MIB:
                rows[i], rows[i + 1] = b, a
                swapped = True
    return rows


def render(snap: Snapshot, theme: Theme, refresh: float | None = None,
           width: int | None = None, height: int | None = None,
           previous: Snapshot | None = None, final: bool = False,
           reserve: dict[str, int] | None = None) -> str:
    """Draw a snapshot as text.

    Every line fits in width columns (measured in terminal columns). If height
    is given (live view), the output is at most height lines: footnotes go
    first, then the smallest programs are folded into one line, then the
    status messages are shortened. final: the frame left on screen when a
    live view ends (no "Ctrl+C to quit" line). reserve: lines to keep for the
    program rows and status messages (from StickyHeights), so the layout stays
    put when content shrinks; padding is the first thing given up when the
    window is too short.
    """
    t, ch = theme, theme.ch
    if width is None:
        width = shutil.get_terminal_size((100, 24)).columns
    width = max(MIN_WIDTH, min(width, MAX_WIDTH))
    stamp = snap.timestamp.strftime("%H:%M:%S") if refresh else ""
    single = len(snap.devices) == 1

    def build_header(compact: bool) -> list[str]:
        lines: list[str] = []
        for dev in snap.devices:
            lines += _gpu_section(dev, snap.compute_by_device.get(dev.address),
                                  width, t, stamp,
                                  snap.averages.get(dev.address), snap.averaged_seconds,
                                  compact)
        if snap.servers:
            lines += _box("MODEL SERVERS", _server_lines(snap.servers, width, t), width, t,
                          color="magenta")
        return lines

    header = build_header(False)
    elapsed = ((snap.timestamp - previous.timestamp).total_seconds()
               if previous is not None else 0.0)
    head, all_rows, foot = _process_table(snap.processes, snap.devices, width, t,
                                          previous, elapsed, show_engine=bool(refresh))
    rows = all_rows
    all_status = _status_lines(snap, previous, width, t)
    status: list[str] | None = all_status
    notes = [t.paint(f" {ch['full']} compute (ROCm)   {ch['desk']} graphics   "
                     f"{ch['evict']} evicted: prefers VRAM, placed elsewhere   "
                     + ("bars: share of the row's GPU" if not single else
                        "bars: share of the card"), "dim")]
    if any(p.in_vram_estimated for p in snap.processes):
        notes.append(t.paint(f" {ch['approx']} estimate: requested minus evicted, at most what "
                             "the card uses; resident counter unusable", "dim"))
    if refresh:
        notes.append(t.paint(" GPU%: shader engine time since the last refresh; n/a for "
                             "ROCm (not reported per program)", "dim"))
    if any(c.link == "guess" for p in snap.processes for c in p.compute_details):
        notes.append(t.paint(" Which GPU client runs ROCm was guessed (several clients, "
                             "Linux 6.15+): compute/graphics split may be off", "dim"))
    if any(p.attribution_guessed for p in snap.processes):
        notes.append(t.paint(" * held jointly by several processes; credited to the "
                             "oldest (a guess)", "dim"))
    notes.append(t.paint(" Shared window images count for each program, so the sum "
                         "can exceed the card's usage.", "dim"))
    last = ([t.paint(f" Refreshing every {refresh:g} s {ch['dot']} Ctrl+C to quit", "dim")]
            if refresh and not final else [])
    credit = [t.paint(" " + CREDIT_LINE, "dim")] if CREDIT_LINE else []
    gap = [""]
    table = True

    def fold(keep: int) -> list[TableRow]:
        hidden = [row for row in all_rows[keep:] if row[1] is not None]
        if not hidden:
            return all_rows[:keep]
        unit = "programs" if single else "rows"   # several GPUs: a row is program + GPU
        return all_rows[:keep] + [(t.paint(
            f" {ch['ellipsis']} {len(hidden)} {'more ' if keep else ''}{unit}, "
            f"{_fmt(sum(row[2] for row in hidden))} in VRAM", "dim"), None, 0)]

    def total_lines() -> int:
        n = len(header) + len(notes) + len(last) + len(credit)
        if table:
            n += len(gap) + len(head) + len(rows) + pad_rows + len(foot)
        if status is not None:
            n += len(gap) + len(status) + pad_status + 2
        return n

    pad_rows = pad_status = 0
    if height:
        wanted = reserve or {}
        pad_rows = max(0, max(PROGRAM_ROWS_MIN, wanted.get("rows", 0)) - len(all_rows))
        pad_status = max(0, wanted.get("status", 0) - len(all_status))
    if height:   # reduce, least important first; boxes are only ever dropped whole
        for _ in range(2):                    # padding goes first: status, then rows
            excess = total_lines() - height
            if excess > 0 and pad_status:
                pad_status = max(0, pad_status - excess)
            elif excess > 0 and pad_rows:
                pad_rows = max(0, pad_rows - excess)
        while notes and total_lines() > height:
            notes.pop()
        if total_lines() > height:
            credit = []                       # content first; it returns when there is room
        if total_lines() > height and len(all_rows) > 1:
            rows = fold(max(1, len(all_rows) - (total_lines() - height) - 1))
        if total_lines() > height and len(all_status) > 1:
            keep = max(1, len(all_status) - (total_lines() - height) - 1)
            status = all_status[:keep] + [t.paint(
                f"{ch['ellipsis']} {len(all_status) - keep} more lines in a taller window",
                "dim")]
        if total_lines() > height:
            gap = []
        if total_lines() > height:
            header = build_header(True)
        if total_lines() > height:
            rows = fold(0)
        if total_lines() > height:
            last = []
        if total_lines() > height:
            status = None
        if total_lines() > height:
            table = False
        hint = ([t.paint(f" Refreshing every {refresh:g} s {ch['dot']} Ctrl+C to quit", "dim")]
                if refresh and not final else [])
        if not last and hint and total_lines() + 1 <= height:
            last = hint                       # a bigger cut freed room for it again
        if not credit and CREDIT_LINE and total_lines() + 1 <= height:
            credit = [t.paint(" " + CREDIT_LINE, "dim")]

    out = list(header)
    if table:
        out += gap + head + [row[0] for row in rows] + [""] * pad_rows + foot
    if status is not None:
        out += gap + _box("STATUS", status + [""] * pad_status, width, t, color="gray")
    out += notes + last + credit
    out = [_fit(line, width, ch["ellipsis"]) for line in out]
    return "\n".join(out[:height] if height else out)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
EPILOG = f"""\
examples:
  sudo python3 %(prog)s              one snapshot
  sudo python3 %(prog)s -r           live view, refreshing every {DEFAULT_REFRESH:g} seconds
  sudo python3 %(prog)s -r 5         live view, refreshing every 5 seconds
  sudo python3 %(prog)s --json       machine-readable output (sizes in bytes)

environment:
  NO_COLOR      set to any value to disable colours
  OLLAMA_HOST   Ollama address (default 127.0.0.1:11434)

exit status:
  {EXIT_OK}  success
  {EXIT_ERROR}  runtime error
  {EXIT_USAGE}  invalid command-line arguments
  {EXIT_UNSUPPORTED}  unsupported system (not Linux, or no AMD GPU using amdgpu)
""" + (f"\n{CREDIT_LINE}\n" if CREDIT_LINE else "")


def _refresh_seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value < MIN_REFRESH:
        raise argparse.ArgumentTypeError(
            f"must be a number of seconds, at least {MIN_REFRESH:g}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{PROGRAM}.py",
        description=f"{PROGRAM} {__version__}: see what is using your AMD GPUs on Linux "
                    "(memory per program, load, clocks, temperatures, power, model servers).",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-r", "--refresh", nargs="?", const=DEFAULT_REFRESH,
        type=_refresh_seconds, metavar="SECONDS",
        help=f"live view, refreshing every SECONDS "
             f"(default {DEFAULT_REFRESH:g}, minimum {MIN_REFRESH:g})")
    parser.add_argument(
        "--json", action="store_true",
        help="print JSON instead of the table (with -r: one line per refresh)")
    parser.add_argument(
        "-V", "--version", action="version",
        version=f"{PROGRAM} {__version__}" + (f"\n{CREDIT_LINE}" if CREDIT_LINE else ""))
    return parser


def _live(gpus: list[PciGpu], args: argparse.Namespace, theme: Theme) -> None:
    """Refresh until interrupted. On a terminal, frames are drawn on the
    alternate screen, in place without clearing first, and within the window's
    width and height; on exit the normal screen returns with the last frame
    printed on it in full, so it stays visible."""
    screen = sys.stdout.isatty() and not args.json
    previous: Snapshot | None = None
    order: list[int] = []
    history = History()
    sticky = StickyHeights()
    if screen:
        sys.stdout.write(LIVE_START)
    try:
        while True:
            snap = collect(gpus)
            history.apply(snap, time.monotonic())
            if args.json:
                print(json.dumps(snap.to_dict()), flush=True)
            else:
                snap.processes = stable_order(snap.processes, order)
                order = [p.pid for p in snap.processes]
                size = shutil.get_terminal_size((100, 24))
                reserve = sticky.update(layout_needs(snap, previous, size.columns, theme),
                                        time.monotonic())
                frame = render(snap, theme, refresh=args.refresh, width=size.columns,
                               height=size.lines if screen else None,
                               previous=previous, reserve=reserve)
                if screen:   # overwrite in place, clear leftovers, no flicker
                    frame = "\x1b[H" + frame.replace("\n", "\x1b[K\n") + "\x1b[K\x1b[J"
                    sys.stdout.write(frame)
                else:
                    sys.stdout.write(frame + "\n\n")
                sys.stdout.flush()
            previous = snap
            time.sleep(args.refresh)
    finally:
        if screen:
            sys.stdout.write(LIVE_END)
            if previous is not None:          # leave the last frame on the screen,
                columns = shutil.get_terminal_size((100, 24)).columns   # in full
                sys.stdout.write(render(previous, theme, refresh=args.refresh,
                                        width=columns, final=True) + "\n")
            sys.stdout.flush()


def _exit_on_signal(signum: int, _frame: object) -> None:
    """Turn SIGTERM/SIGHUP into a normal exit, so the terminal is restored."""
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prog = PROGRAM
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _exit_on_signal)
    try:
        gpus = find_supported_gpus()
        theme = Theme.auto()
        if args.refresh is not None:
            _live(gpus, args, theme)
        elif args.json:
            print(json.dumps(collect(gpus).to_dict(), indent=2))
        else:
            print(render(collect(gpus), theme))
        return EXIT_OK
    except UnsupportedSystemError as exc:
        print(f"{prog}: unsupported system: {exc}", file=sys.stderr)
        return EXIT_UNSUPPORTED
    except KeyboardInterrupt:
        if args.refresh is None:
            return 130
        if not args.json and not sys.stdout.isatty():
            print()
        return EXIT_OK
    except Exception as exc:      # last-resort handler: short message, no traceback
        if os.environ.get("VRAM_DEBUG"):
            raise
        print(f"{prog}: error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
