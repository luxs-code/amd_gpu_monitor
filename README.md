AMD GPU Monitor
===============

A Linux command-line tool that shows which programs use the VRAM of AMD GPUs.

FEATURES
--------

- Per-process VRAM usage statistics.
- Live monitoring with a configurable refresh interval.
- JSON output for scripting and further analysis.
- GPU sensor monitoring, including temperature and power information where supported.
- Information about VRAM usage, requested memory, and memory placed outside VRAM where the necessary kernel counters are available.
- Detection and reporting of unsupported systems and invalid readings.

The information available depends on the GPU, Linux kernel, AMDGPU driver, and process permissions.


REQUIREMENTS
------------

- Linux.
- An AMD GPU supported by the amdgpu driver.
- Python 3.


USAGE
-----

Single snapshot:

    sudo python3 amd_gpu_monitor.py

Live monitoring:

    sudo python3 amd_gpu_monitor.py -r

Refresh every five seconds:

    sudo python3 amd_gpu_monitor.py -r 5

JSON output:

    sudo python3 amd_gpu_monitor.py --json

Display the version:

    python3 amd_gpu_monitor.py --version

Display command-line help:

    python3 amd_gpu_monitor.py --help


PERMISSIONS
-----------

Running with sudo is recommended for the most complete process-level VRAM statistics. Without elevated privileges, some information may be unavailable or incomplete.


LIMITATIONS
-----------

The available information depends on the GPU, kernel version, driver, and permissions. Not every GPU or Linux configuration exposes the same statistics.

This tool focuses on AMD GPU monitoring under Linux and does not aim to provide universal hardware monitoring.


VERSION
-------

2.9.0
