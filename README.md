# PSCKangaroo

GPU-accelerated **Pollard's Kangaroo** for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP) on **secp256k1**.

A fork of [RCKangaroo](https://github.com/RetiredC/RCKangaroo) by [RetiredCoder](https://github.com/RetiredC). Special thanks to RetiredCoder for the SOTA method and GPU kernel architecture — the core algorithm and GPU kernel are his work.

## Purpose

PSCKangaroo is built for one scenario: **long-running ECDLP puzzles on a single GPU with lots of RAM.**

The [Bitcoin Puzzle Transaction](https://bitcointalk.org/index.php?topic=1306983.0) challenges define this scenario — known public keys, defined search ranges, and puzzles that may take years on a single machine. When a run may last months or years, what matters is:

1. **No wasted work.** Crashes, reboots, power outages, kernel updates — they all happen. Without checkpoint/resume, months of computation are lost. PSCKangaroo auto-saves and resumes from where it stopped.

2. **No memory crashes.** RCKangaroo has no RAM limit — it allocates memory until the OS kills it (see [OOM analysis](#oom-analysis) below). PSCKangaroo's `-ramlimit` guarantees stable operation indefinitely.

3. **More entries per GB.** PSCKangaroo uses 16-byte entries vs ~40 bytes in RCKangaroo (32 data + 4 pointer + 4 overhead, per RC's own [RAM formula](https://github.com/RetiredC/RCKangaroo/blob/main/RCKangaroo.cpp#L328)). That's **2.5× more DPs in the same RAM**.

For short puzzles (≤ 80 bits), RCKangaroo is faster — see benchmark below. PSCKangaroo targets the puzzles where RCKangaroo can't run safely for extended periods.

## The Mathematical Reality of Large Puzzles

Before choosing any solver, it's important to understand the scale of what we're attempting.

**Puzzle 135** has a 134-bit search range. The expected number of operations to solve it is ~1.15 × √(2¹³⁴) ≈ **2⁶⁷·² operations** — roughly **1.7 × 10²⁰**.

At 3.1 GKeys/s (RTX 5070), that's:

| Metric | Value |
|---|---|
| Operations needed | ~1.7 × 10²⁰ |
| Time at 3.1 GK/s | **~1,740 years** |
| Time with 10 GPUs | ~174 years |
| Time with 100 GPUs | ~17.4 years |
| Time with 1,000 GPUs | ~1.7 years |

**No single-GPU solver — RCKangaroo, PSCKangaroo, or any other — can reliably solve Puzzle 135 in a human lifetime.** Everyone running these solvers on Puzzle 135 is playing a probabilistic lottery: hoping to find the collision early, long before the expected time. Some will get lucky; most won't.

PSCKangaroo doesn't change this reality. It is one more tool in the toolbox — optimized for crash resilience and memory efficiency on long runs. The mathematical difficulty of large puzzles requires either massive parallelism (hundreds of GPUs), a breakthrough in algorithms, or extraordinary luck.

That said, someone running a single GPU 24/7 with checkpoint protection has a nonzero chance every second. PSCKangaroo ensures that chance is never wasted by a crash.

## Benchmark: PSCKangaroo v59 vs RCKangaroo v3.1

**Hardware:** NVIDIA RTX 5070 (Blackwell, 12 GB VRAM) / AMD Ryzen 7 9800X3D / 123 GB RAM / CUDA 12.9 / Linux

**Puzzle 80** (79-bit range, known private key `ea1a5c66dcc11b5ad180`), 5 runs each:

| Configuration | Median | Mean | Min | Max | Solved |
|---|---|---|---|---|---|
| **RCKangaroo** DP=16 | **301s** | 299s | 120s | 514s | 5/5 |
| **PSC v59** concurrent DP=12 RAM=8GB | **320s** | 463s | 236s | 837s | 5/5 |
| **PSC v59** concurrent DP=14 RAM=20GB | 423s | 536s | 142s | 1223s | 5/5 |
| **PSC v59** concurrent DP=12 RAM=20GB | 491s | 558s | 252s | 817s | 5/5 |

**Observations:**

- RCKangaroo wins by ~6% on median (301s vs 320s) — expected, as SOTA's concurrent walk dynamics are mathematically optimal (K ≈ 1.15).
- PSC's best single run (142s) was faster than RC's best (120s) — high variance is normal for a probabilistic algorithm.
- Both solvers run the same GPU kernel at ~3.1 GKeys/s. The performance difference is purely architectural.
- **For Puzzle 80 and similar short puzzles, RCKangaroo is the better choice** — it solves in minutes with zero setup overhead.

## How RCKangaroo and PSCKangaroo Compare

| Aspect | RCKangaroo v3.1 | PSCKangaroo v59 |
|---|---|---|
| GPU kernel speed | ~3.1 GKeys/s | ~3.1 GKeys/s (same kernel) |
| Effective bytes per DP | **~40 B** | **16 B** (2.5× more entries/GB) |
| RAM management | No limit (grows until OOM) | `-ramlimit` (stable indefinitely) |
| Checkpoint | **`-tames` (pre-generated TAMEs only)** | **Full** (auto-save + Ctrl+C safe) |
| Crash recovery | Loses all WILDs and solve progress | Resumes from last checkpoint |
| Collision dynamics | Concurrent T+W (t² growth, K≈1.15) | Concurrent T+W (t² growth) |
| Minimum DP @120GB | DP≥24 to run for months | **DP=12+** safely |
| Pre-generate TAMEs | Yes (`-tames file -max N`) | Yes (checkpoint system) |
| Ops limit | `-max` (stops solver, no save) | Runs indefinitely |
| Puzzle 80 median | **301s** (faster) | 320s |
| Puzzle 135+ viability | Needs DP≥24, no crash recovery | DP=16, checkpoint every 4h |

**Note on RCKangaroo's `-tames` feature:** RC can pre-generate a TAME table with `-tames file -max N` and reload it across multiple runs. This gives a head start but is not a full checkpoint — WILDs and solve progress are not saved. If the solver crashes during a solve, only the pre-generated tame file survives.

**Note on RCKangaroo's `-max` feature:** RC can limit the number of operations with `-max N`. When reached, the solver stops — but does not save progress (except in tame generation mode). It's a "give up after N ops" limit, not a RAM management tool.

## OOM Analysis

RCKangaroo stores every DP unconditionally. At ~40 bytes per entry with no RAM limit, memory grows linearly until the OS kills the process:

| DP | DPs/s @3.1GK/s | Growth rate | OOM on 120 GB system |
|---|---|---|---|
| 14 | 189,209 | 7.6 MB/s | **~4.4 hours** |
| 16 | 47,302 | 1.9 MB/s | **~17.6 hours** |
| 20 | 2,956 | 118 KB/s | **~11.8 days** |
| 24 | 185 | 7.4 KB/s | **~6.3 months** |

These times are computed from RC's own RAM formula: `(32 + 4 + 4) × total_ops / 2^DP`, verified in `RCKangaroo.cpp` line 328. The `malloc` inside RC's MemPool does not check for allocation failure — OOM results in a segfault with no save.

PSCKangaroo with `-ramlimit 120` runs indefinitely at any DP value. When the table reaches 93% capacity, it freezes (no new TAMEs stored) and continues hunting with 100% WILDs.

## Quick Start

### Recommended: Concurrent mode (v59)

```bash
./psckangaroo -gpu 0 -dp 16 -range 134 \
  -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  -start 4000000000000000000000000000000000 \
  -ramlimit 120 -concurrent 1 -wwbuffer 5 -checkpoint 4
```

This runs TAME and WILD kangaroos simultaneously from second 1 (like RCKangaroo), but with memory protection, checkpoint, and 16-byte compact entries.

> **DP value note:** Lower DP values (12–14) generate more collision candidates per second but increase CPU load and temperature during the HUNT phase. For 24/7 operation, DP=16 is recommended as a balance between collision rate and thermal stability. For short validation runs, DP=12 is fine.

### Resume from checkpoint

```bash
./psckangaroo -gpu 0 -dp 16 -range 134 \
  -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  -start 4000000000000000000000000000000000 \
  -ramlimit 120 -concurrent 1 -wwbuffer 5 -checkpoint 4 \
  -loadwild wild_checkpoint.dat
```

### Multiple GPUs (experimental)

Multi-GPU support is inherited from RCKangaroo but has not been tested with PSCKangaroo's concurrent mode. If you have multiple GPUs and want to try:

```bash
# Two GPUs (GPU 0 and GPU 1)
./psckangaroo -gpu 01 -dp 16 -range 134 \
  -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  -start 4000000000000000000000000000000000 \
  -ramlimit 120 -concurrent 1 -wwbuffer 5 -checkpoint 4
```

Please report results via [GitHub Issues](https://github.com/pscamillo/PSCKangaroo/issues).

### Validation (Puzzle 70, ~30 seconds)

```bash
./psckangaroo -gpu 0 -dp 12 -range 69 -ramlimit 4 -checkpoint 0 -concurrent 1 \
  -pubkey 0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483 \
  -start 200000000000000000
```

Expected result: private key `349b84b6431a6c4ef1`.

### Validation (Puzzle 80, ~5–10 minutes)

```bash
./psckangaroo -gpu 0 -dp 12 -range 79 -ramlimit 8 -checkpoint 0 -concurrent 1 -wwbuffer 5 \
  -pubkey 037e1238f7b1ce757df94faa9a2eb261bf0aeb9f84dbf81212104e78931c2a19dc \
  -start 80000000000000000000
```

Expected result: private key `ea1a5c66dcc11b5ad180`.

## Operating Modes

### Concurrent (v59) — recommended

```
-concurrent 1
```

Runs 33% TAME + 33% WILD1 + 33% WILD2 kangaroos from the first second. The TAME table grows while WILDs hunt against it — collision probability grows quadratically (t²) until the table fills, then automatically switches to 100% WILDs.

This is functionally equivalent to how RCKangaroo operates, but with `-ramlimit`, checkpoint, and 2.5× more entries per GB.

### ALL-TAME (legacy)

```
-allwild 0
```

Phase 1 (TRAP): fills all RAM with TAMEs — no collisions possible during this phase. Phase 2 (HUNT): 100% WILDs check against the frozen table. For Puzzle 80, this took 1h31m (1h26m TRAP + 5 min HUNT), while concurrent mode solved the same puzzle in ~5 minutes. Concurrent mode eliminates this dead time.

### ALL-WILD

```
-allwild 1
```

WILD-WILD collisions only, no TAMEs. Experimental — concurrent mode outperforms in all tested scenarios.

### W-W Buffer (v58)

```
-wwbuffer 5
```

Reserves 5% of RAM for a WILD1-WILD2 cross-collision buffer. Adds a second collision channel on top of TAME-WILD detection. In testing, the W-W buffer produced hundreds of additional collision candidates per run. Can be combined with concurrent or ALL-TAME mode.

## How It Works

### The SOTA Kangaroo Method

Pollard's Kangaroo solves ECDLP by launching pseudo-random walks on the elliptic curve:

- **TAME kangaroos** start from known positions (k·G where k is known)
- **WILD kangaroos** start from positions derived from the target public key

When a TAME and WILD land on the same distinguished point (collision), the private key is computed from the difference in their accumulated walk distances.

**Distinguished Points (DPs)** are positions where the x-coordinate has a specific number of leading zero bits. Only DPs are stored, reducing memory by a factor of 2^DP while preserving collision detection.

The **SOTA method** (by RetiredCoder) uses equivalence classes and the negation map to reduce expected operations from ~2.08√n to ~1.15√n. PSCKangaroo inherits this method and GPU kernel unchanged.

### What PSCKangaroo adds

**Concurrent mode (v59):** Eliminates the sequential TRAP/HUNT separation. TAMEs and WILDs run simultaneously from second 1, giving quadratic (t²) collision probability growth while respecting a RAM limit and saving checkpoints.

**16-byte compact entries:** Each DP is stored in 16 bytes using a 20-bit x-signature + 104-bit truncated distance. When a collision is detected, the exact distance is recovered via an async BSGS search (±2³², typically 200–400ms). This fits 2.5× more entries per GB compared to RCKangaroo's ~40-byte entries.

**W-W buffer (v58):** A small hash table (5% of RAM) stores recent WILDs and cross-checks WILD1 vs WILD2 collisions, adding a second collision channel alongside TAME-WILD detection.

**Checkpoint/Resume:** Auto-saves the TAME table at configurable intervals and on Ctrl+C. A crash after weeks of computation costs at most a few hours of work, not the entire run.

## Command-Line Options

| Option | Description | Default |
|---|---|---|
| `-gpu N` | GPU index(es) — e.g., `0` for one GPU, `01` for two (multi-GPU is experimental) | 0 |
| `-dp N` | Distinguished point bits (6–60) | — |
| `-range N` | Key range in bits (32–170) | — |
| `-pubkey <hex>` | Target compressed public key | — |
| `-start <hex>` | Range start offset | — |
| `-ramlimit N` | RAM limit in GB | — |
| `-concurrent 1` | v59 concurrent mode (recommended) | 0 |
| `-allwild 0/1` | 0 = ALL-TAME, 1 = ALL-WILD | 0 |
| `-wwbuffer N` | W-W buffer: N% of RAM (0–20) | 0 |
| `-checkpoint N` | Auto-save interval in hours (0 = off) | 4 |
| `-savefile <f>` | Checkpoint filename | `wild_checkpoint.dat` |
| `-loadwild <f>` | Load checkpoint and resume | — |
| `-groups N` | Points per batch inversion (8–256) | 24 |
| `-savedps <f>` | Save evicted DPs to file | — |
| `-waveinterval N` | Minutes between WILD wave renewals (0 = off) | 0 |
| `-rotation 0/1` | Table freeze (0) or rotation (1) | 0 |

## Build

### Linux

```bash
git clone https://github.com/pscamillo/PSCKangaroo.git
cd PSCKangaroo

# Edit Makefile if needed: set GPU_ARCH for your GPU
# Default is sm_120 (Blackwell / RTX 5070)

make clean && make
```

### Windows

Requirements: Visual Studio 2022 + CUDA Toolkit 12.8+ (with VS integration).

1. Clone the repository
2. Open `PSCKangaroo.sln` in Visual Studio
3. Select **Release | x64**
4. Build (F7)

The `.vcxproj` targets CUDA 12.8. If you have a different CUDA version, right-click the project → Build Dependencies → Build Customizations → select your installed CUDA version.

### GPU Architecture

| GPU Series | Linux Makefile | Windows (auto-detected) |
|---|---|---|
| RTX 2060/2070/2080 | `GPU_ARCH="-gencode=arch=compute_75,code=sm_75"` | sm_75 |
| RTX 3060/3070/3080/3090 | `GPU_ARCH="-gencode=arch=compute_86,code=sm_86"` | sm_86 |
| RTX 4060/4070/4080/4090 | `GPU_ARCH="-gencode=arch=compute_89,code=sm_89"` | sm_89 |
| RTX 5070/5080/5090 | `GPU_ARCH="-gencode=arch=compute_120,code=sm_120"` (default) | sm_120 |

**CUDA 13.0 note:** CUDA 13.0+ dropped support for Maxwell, Pascal, and Volta architectures. If you use CUDA 13.0+, you need at least Turing (RTX 2060 / sm_75). For older GPUs (GTX 1060/1070/1080), use CUDA 12.x.

## Requirements

- **GPU:** NVIDIA with Compute Capability ≥ 7.5 (Turing or newer) for CUDA 13.0+, or ≥ 6.0 (Pascal or newer) for CUDA 12.x
- **CUDA Toolkit:** 12.0+ (tested on 12.9)
- **RAM:** 8 GB minimum, 128 GB recommended for large puzzles
- **OS:** Linux (Ubuntu 22.04+) or Windows

## Performance

| GPU | Speed | Notes |
|---|---|---|
| RTX 5070 Ti (Blackwell) | ~4.5 GKeys/s | Confirmed by user (Windows) |
| RTX 5070 (Blackwell) | ~3.0 GKeys/s | Tested (Linux) |
| RTX 5070 × 2 (multi-GPU) | ~5.8 GKeys/s | Confirmed by user (Linux, experimental) |
| RTX 2070 (Turing) | ~1.2 GKeys/s | Confirmed by user (Windows) |
| GTX 1060 (Pascal) | ~0.19 GKeys/s | Confirmed by user (Linux) |

## Design History

### What was tried and removed

Earlier versions included GPU-side endomorphism (canonical form), a "cheap second point" (P−J), and extended distinguished points (XDP). These were removed after feedback from kTimesG on BitcoinTalk, who correctly identified that:

- Canonical X form doesn't produce useful collisions below 254-bit ranges
- Cheap second point generates DPs that no kangaroo converges toward
- XDP is functionally equivalent to lowering the DP parameter

### The ALL-TAME lesson

The original PSCKangaroo architecture used sequential TRAP/HUNT phases — fill all RAM with TAMEs first, then hunt with WILDs. While more TAMEs increase collision probability per-check, the TRAP phase is dead time with zero collision probability. For Puzzle 80, ALL-TAME took 1h31m (1h26m TRAP + 5 min HUNT). Concurrent mode solved the same puzzle in ~5 minutes by hunting from second 1. This analysis led to the v59 concurrent mode.

## Changelog

### v59 (current — concurrent mode)
- **Concurrent mode:** 33% TAME + 67% WILD from second 1, quadratic (t²) collision growth
- New CLI flag: `-concurrent 1`
- Auto-transitions to 100% WILDs when table reaches 93% capacity
- Stats display shows TAMEs growing + WILDs hunting simultaneously

### v58 (W-W buffer)
- Small hash table (5–10% of RAM) for WILD1-WILD2 cross-collision detection
- New CLI flag: `-wwbuffer N`

### v57 (kTimesG cleanup)
- Removed GPU endomorphism canonical form, cheap second point, XDP, GPU Bloom filter

### v56C
- Ultra-compact 16-byte DP entries (+56% capacity vs 25-byte format)
- Async BSGS resolver (4 threads, precomputed baby table)

### v53
- ALL-TAME mode, table freeze

### Earlier versions
- Dual hash table, uniform jumps, shard-locked reads

## Credits

- **[RetiredCoder (RC)](https://github.com/RetiredC)** — Original RCKangaroo, SOTA method, GPU kernel, batch Montgomery inversion, PTX assembly. The core algorithm is entirely his work.
- **[JeanLucPons](https://github.com/JeanLucPons)** — Foundational Kangaroo/VanitySearch/BSGS implementations.
- **kTimesG** — Critical feedback on endomorphism/cheap point/XDP that led to the v57 cleanup.
- **pscamillo** — Concurrent mode (v59), W-W buffer (v58), ALL-TAME mode, 16-byte compact entries, async BSGS resolver, checkpoint system, table freeze, `-ramlimit`.

## Related work

- [mr_blackwell](https://github.com/pscamillo/mr_blackwell) — Native Miller-Rabin GPU kernel for Blackwell SM 12.0. Same author, related domain (modular arithmetic on GPU), independent project.

## License

GPLv3 — see [LICENSE](LICENSE). Derivative work of [RCKangaroo](https://github.com/RetiredC/RCKangaroo).

## Disclaimer

This software is provided for educational and research purposes. Solving Bitcoin puzzle challenges is a probabilistic endeavor — there is no guarantee of success regardless of hardware or software used. Use responsibly and in accordance with applicable laws.

## Supporting development

If this project is useful to you, consider **starring the repository** — it helps visibility and signals to others that the work matters.

If you want to support continued development directly:

**BTC (bech32):** `bc1q62axpzhteksv4yhqhfzvmeqqq98gcvzfak0etk`

Hardware, electricity, and tooling costs are real, though partially offset by solar power. Contributions of code, benchmarks, bug reports, and documentation are also welcome — open an issue or pull request.
