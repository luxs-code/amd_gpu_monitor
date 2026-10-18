# AMD GPU Monitor

A Linux command-line tool that shows what is using your AMD GPUs: VRAM per program,
GPU load, clocks, temperatures and power, and the AI model servers running on the machine.
It is a single Python file with no dependencies beyond the standard library.

```text
╭─ GPU 0000:83:00.0 · AMD Radeon RX 7900 XTX · 15:47:13 ───────────────────────────────────────────────╮
│ VRAM ████████████████████████████████████████████████████████████▓▓▓▓░░░░░  22.4 GiB / 24.0 GiB used │
│      █ compute (ROCm) 20.8 GiB  ▓ graphics & driver  1.6 GiB  ░ free  1.6 GiB                        │
│                                                                                                      │
│                  now    1 min    5 min   15 min   averages complete                                  │
│ GPU busy        100%      97%      85%      62%                                                      │
│ Power          314 W    305 W    271 W    198 W   SoC, average sensor · limit 327 W                  │
│ GPU clock   2482 MHz 2470 MHz 2390 MHz 2100 MHz   shader (gfx/compute) clock                         │
│ Mem clock   1249 MHz 1249 MHz 1249 MHz 1102 MHz                                                      │
│ Edge            59°C     58°C     57°C     55°C   limit 100°C                                        │
│ Hotspot         74°C     73°C     70°C     66°C   limit 110°C                                        │
│ Memory          68°C     67°C     66°C     64°C   limit 100°C                                        │
│ VRAM used   22.4 GiB 22.4 GiB 22.4 GiB 18.0 GiB                                                      │
│ GTT used     1.2 GiB  1.2 GiB  1.1 GiB  922 MiB   system memory mapped for the GPU (all uses)        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ MODEL SERVERS ──────────────────────────────────────────────────────────────────────────────────────╮
│ Ollama     127.0.0.1:11434        qwen3:32b · 21.8 GiB total · 21.0 GiB on GPU ( 96%)                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────╯

 PROGRAM                   PID  WHAT IT IS               IN VRAM   EVICTED  GPU%  USAGE
 ──────────────────────────────────────────────────────────────────────────────────────────────────────
 llama-server          2535258  LLM server (llama.cpp)  20.8 GiB         -   n/a  ██████████████████
 gnome-shell            492096  desktop & compositor     665 MiB   125 MiB    6%  ▓▒
 firefox               2452536  web browser               28 MiB         -    1%  ▏
 Xwayland               539042  X11 app support           17 MiB    14 MiB    0%  ▏
 ──────────────────────────────────────────────────────────────────────────────────────────────────────
 graphics memory total                                   710 MiB   139 MiB

╭─ STATUS ─────────────────────────────────────────────────────────────────────────────────────────────╮
│ ⚠ 1.6 GiB VRAM free (7%): little headroom left.                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────╯
 █ compute (ROCm)   ▓ graphics   ▒ evicted: prefers VRAM, placed elsewhere   bars: share of the card
 GPU%: shader engine time since the last refresh; n/a for ROCm (not reported per program)
 Shared window images count for each program, so the sum can exceed the card's usage.
 (c) Luxs 2026 · GitHub: https://github.com/<your-account>/amd_gpu_monitor · Web: https://<your-site>
```

*Example of the live view (`-r`) on one RX 7900 XTX running Ollama.*


## Features

- **VRAM per program**, split into memory in VRAM and memory that is *evicted*
  (prefers VRAM but is currently placed elsewhere). ROCm (compute) and graphics
  memory are shown separately.
- **Several GPUs**: one row per program and GPU, each bar scaled to that GPU.
  GPUs linked by XGMI are taken into account.
- **Live view** with a configurable refresh interval, like `top`:
  - 1, 5 and 15-minute averages for load, power, clocks, temperatures and memory;
  - a per-program **GPU%** column (shader engine time) for graphics programs;
  - a stable layout that does not jump when programs come and go;
  - the last frame stays on screen when you quit.
- **GPU sensors**: load, power (with its limit), shader and memory clocks, and edge,
  hotspot and memory temperatures with their critical limits.
- **Model servers**: Ollama, llama.cpp (`llama-server`, llamafile), vLLM, SGLang,
  LM Studio, KoboldCpp, LocalAI and Text Generation Inference are detected
  automatically, with the models they serve. For Ollama, model sizes and the part
  on the GPU are shown too.
- **Status messages**: low free VRAM, temperatures near their limits, memory being
  evicted, programs requesting more VRAM than the card has.
- **JSON output** for scripts and further analysis.
- **Careful with its data**: kernel counters are checked for plausibility, estimates
  are marked, unknown values are shown as `n/a` instead of zero, and a sleeping
  GPU is not woken up to read its sensors.
- **Detection of unsupported systems** (not Linux, no AMD GPU, wrong driver) with a
  clear message.


## Requirements

- Linux.
- An AMD GPU driven by the `amdgpu` kernel driver.
- Python 3.7 or newer. No other packages are needed.
- Optional: `lspci` (package `pciutils`) for GPU names on cards that do not report one.

The figures available depend on the kernel version. Per-program memory needs a
kernel whose amdgpu driver reports per-client statistics; the "requested" and
"evicted" figures need Linux 6.4 or newer. On older kernels the eviction columns
show `n/a`.


## Installation

Download the single file and run it with Python:

```sh
curl -O https://raw.githubusercontent.com/<your-account>/amd_gpu_monitor/main/amd_gpu_monitor.py
sudo python3 amd_gpu_monitor.py
```

Or clone the repository:

```sh
git clone https://github.com/<your-account>/amd_gpu_monitor.git
cd amd_gpu_monitor
sudo python3 amd_gpu_monitor.py
```


## Usage

| Command | What it does |
|---|---|
| `sudo python3 amd_gpu_monitor.py` | One snapshot |
| `sudo python3 amd_gpu_monitor.py -r` | Live view, refreshing every 2 seconds |
| `sudo python3 amd_gpu_monitor.py -r 5` | Live view, refreshing every 5 seconds (minimum 0.2) |
| `sudo python3 amd_gpu_monitor.py --json` | One snapshot as JSON |
| `sudo python3 amd_gpu_monitor.py -r --json` | One line of JSON per refresh |
| `python3 amd_gpu_monitor.py --version` | Show the version |
| `python3 amd_gpu_monitor.py --help` | Show the command-line help |

In the live view, press **Ctrl+C** to quit.


## Reading the display

**GPU box.** The VRAM bar shows how the card's memory is split between compute
(ROCm), graphics and the driver, and free space. The table below it shows each
sensor now and, in the live view, its averages. Temperatures turn yellow within
10 °C of their critical limit and red at or above it.

**Model servers.** One line per detected server with its address and models, or why
it could not be asked (no reply, refused, unexpected reply, or port not visible
without root).

**Program table.**

| Column | Meaning |
|---|---|
| IN VRAM | Memory of the program that is in VRAM |
| EVICTED | Memory that prefers VRAM but is currently placed elsewhere |
| GPU% | Live view only: shader engine time since the last refresh. Several engines working at once can push it above 100% |
| USAGE | Bar: the program's share of the card (with several GPUs: of the GPU in that row) |

Markers:

| Marker | Meaning |
|---|---|
| `≈` | Estimate: the kernel's exact counter was missing or implausible |
| `*` | Memory held jointly by several processes and credited to the oldest of them, which is a guess |
| `≥` | Total is a minimum, because some programs did not report this value |
| `n/a` | Not available from the kernel |

**Status box.** Messages state what was measured; where a consequence depends on what
programs do next, they say "can".


## JSON output

`--json` prints the same data as one JSON object; with `-r`, one object per line and
refresh. All sizes are in bytes. The main keys are `devices` (sensors and VRAM per
GPU), `processes` (per program, with `parts` per GPU and the raw kernel figures per GPU
client), `servers`, `averages` (live view only) and `warnings`.

Values the kernel does not provide come with a flag. For example, check
`evicted_known` before using `evicted`.


## Permissions

Running with `sudo` gives the complete picture. Without root:

- programs of other users (for example the `ollama` service user) are missing from the
  graphics figures; ROCm memory is still shown;
- the kernel's record of which process uses a shared GPU client cannot be read;
- model servers of other users are listed without their port.


## Environment variables

| Variable | Effect |
|---|---|
| `NO_COLOR` | Set to any value to disable colours |
| `OLLAMA_HOST` | Address where Ollama is looked for if its process is not visible (default `127.0.0.1:11434`) |
| `AMD_GPU_MONITOR_DEBUG` | Set to show a full traceback on errors |
| `PYTHONIOENCODING=ascii` | Draw with plain ASCII characters; characters the terminal cannot show are printed as `?` |

Colours are also switched off automatically when the output goes to a file or pipe.


## Exit status

| Code | Meaning |
|---|---|
| 0 | Success (including quitting the live view with Ctrl+C) |
| 1 | Runtime error |
| 2 | Invalid command-line arguments |
| 3 | Unsupported system: not Linux, or no AMD GPU using the amdgpu driver |
| 128 + N | Stopped by signal N, for example 130 when a snapshot is interrupted with Ctrl+C, or 143 after `kill` |


## Limitations

- **AMD GPUs on Linux only.** NVIDIA and Intel GPUs are detected and reported as
  unsupported.
- **No per-program compute load for ROCm on most consumer GPUs.** ROCm sends its work
  to the GPU through user-mode queues that the kernel does not account per process.
  The whole-GPU load includes it, but the GPU% column shows `n/a` for ROCm programs.
- **Some figures are estimates.** On some kernels the resident-memory counter is
  unreliable; the tool then estimates and marks the value with `≈`.
- **Kernel differences.** The meaning of some kernel counters changed between versions;
  the code documents which counter is used for what, and why.
- **Model servers** are recognised by their process names and asked on the ports they
  listen on. Servers running under other names, or in another network namespace
  (for example inside a container), may not be found.


## License

MIT. See the `SPDX-License-Identifier` line at the top of `amd_gpu_monitor.py`.
For GitHub to show the license, add a `LICENSE` file with the MIT license text.


## Author

(c) Luxs 2026 · GitHub: https://github.com/<your-account>/amd_gpu_monitor · Web: https://<your-site>


## Version

2.10.1. Run `python3 amd_gpu_monitor.py --version` to see the version you have.
