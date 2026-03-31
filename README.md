# PSCKangaroo

GPU-accelerated **Pollard's Kangaroo** algorithm for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP) on **secp256k1**.

A fork of [RCKangaroo](https://github.com/RetiredC/RCKangaroo) by [RetiredCoder](https://github.com/RetiredC), with bug fixes, new modes, and optimizations.

## What is this?

PSCKangaroo solves the ECDLP: given a public key `P = k*G` and a known range `[start, start + 2^range]`, it finds the private key `k` using Pollard's Kangaroo (lambda) method with GPU acceleration.

This is primarily aimed at the [Bitcoin Puzzle Transaction](https://bitcointalk.org/index.php?topic=1306983.0) challenges, where public keys are known and the search range is defined.

## Features

### Core Algorithm
- **SOTA method** (State of the Art) — Equivalence Classes + Negation Map, achieving the theoretical optimum K ≈ 1.15
- **SOTA+ Cheap Second Point** — observes `P - J` for free during `P + J` computation, effectively doubling DP generation rate
- **3x Endomorphism** — exploits secp256k1's cube root of unity (β) to search 3 equivalent points per kangaroo step

### Optimizations
- **XDP (Extended Distinguished Points)** — threshold-based DP detection accepts multiple patterns instead of just zero, multiplying DP rate by 8x (configurable: 1x, 2x, 4x, 8x, 16x)
- **Ultra-compact 16-byte DP entries** — 56% more entries fit in RAM compared to the original 25-byte format
- **Async BSGS resolver** — multi-threaded Baby-Step Giant-Step resolves truncated-distance collisions in background
- **Dual hash table** — separate tables for WILD1/WILD2 (or TAME/WILD) with cross-collision detection
- **Table freeze** — tables become read-only when full, preventing false-positive explosion from overwrites
- **Uniform jumps** — benchmarked 19% faster than stratified jump tables for large puzzles
- **Batch Montgomery inversion** — amortizes modular inversions across groups of point additions
- **PTX inline assembly** — critical 256-bit arithmetic uses hand-tuned PTX instructions

### Modes
- **ALL-TAME** (default, recommended for 130+ bits) — fills all RAM with TAME points, then hunts with 100% WILDs. Maximizes T-W collision probability.
- **ALL-WILD** — dual table with WILD-WILD collision detection only. No TRAP phase.
- **HYBRID** — preload a checkpoint, then hunt with combined T-W and W-W detection.

### Reliability
- **Checkpoints** — auto-save at configurable intervals + safe exit on Ctrl+C (format RCKDT5C)
- **Savedps** — evicted DPs saved to disk for offline cross-checking
- **Shard-locked reads** — eliminates torn-read false positives in concurrent hash table access

## Design Rationale

The original RCKangaroo was designed as a proof of concept focused on the GPU kernel. PSCKangaroo's modifications were driven by a specific hardware setup: a single **RTX 5070** (12 GB VRAM) paired with **128 GB of system RAM**.

The key insight was that with a large RAM-to-GPU ratio, the optimal strategy changes. Instead of running balanced TAME/WILD kangaroos (where half the RAM stores TAMEs and the other half stores WILDs), it's better to exploit the full RAM capacity:

- **ALL-TAME mode** fills the entire 128 GB with TAME distinguished points during Phase 1 (TRAP), then switches to 100% WILD kangaroos on the GPU during Phase 2 (HUNT). The WILDs are not stored — they are checked against the massive TAME table and discarded. This doubles the number of TAMEs compared to a balanced split, which directly doubles the T-W collision probability per WILD step.

- **ALL-WILD mode** takes the opposite approach for scenarios where WILD-WILD collisions are preferred, using dual hash tables that fill the entire RAM.

- **Ultra-compact 16-byte DP format** was implemented specifically to squeeze 56% more entries into the same RAM footprint. On a 128 GB system, this translates to billions of additional stored points.

- **Table freeze** prevents the hash tables from rotating entries once full, which would cause an explosion of false positives on long runs — important when a single machine may run for weeks or months.

These choices reflect the reality that most individual hunters run a single GPU with as much RAM as they can afford, rather than multi-GPU clusters. The architecture is optimized for that profile.

## Bug Fixes Over Original RCKangaroo

These bugs were found and fixed through systematic auditing:

1. **Wrong BETA2/LAMBDA/LAMBDA2 constants** — the endomorphism constants were incorrect, causing missed collisions. Verified against `bitcoin-core/secp256k1` and `noble-secp256k1`. This fix alone yielded ~2x speedup on Puzzle 79 (consistent with the theoretical √3 endomorphism gain).

2. **MulLambdaModN sign-extension bug** — 64-bit intermediate values were sign-extended incorrectly, corrupting scalar multiplication results for certain input patterns.

3. **GPU Bloom filter byte-index error** — bit addressing in the Bloom filter used wrong byte offsets, causing both false positives and false negatives.

All fixes were validated by successfully solving **Puzzle 79** (known answer) and cross-referencing against reference implementations.

## Requirements

- **GPU**: NVIDIA with Compute Capability ≥ 6.0 (Pascal or newer)
  - Tested on: RTX 5070 (Blackwell, sm_120)
  - Should work on: RTX 3000/4000/5000 series (adjust `GPU_ARCH` in Makefile)
- **CUDA Toolkit**: 12.0 or newer
- **RAM**: 16 GB minimum, 128 GB recommended for large puzzles
- **OS**: Linux (Ubuntu 22.04+ recommended) or Windows

## Build

```bash
# Clone the repository
git clone https://github.com/user/PSCKangaroo.git
cd PSCKangaroo

# Edit Makefile if needed: set GPU_ARCH for your GPU
# Default is sm_120 (Blackwell / RTX 5070)

# Build
make clean && make
```

### GPU Architecture Reference

| GPU Series | Architecture | Makefile Setting |
|---|---|---|
| RTX 3060/3070/3080/3090 | Ampere | `GPU_ARCH="-gencode=arch=compute_86,code=sm_86"` |
| RTX 4060/4070/4080/4090 | Ada Lovelace | `GPU_ARCH="-gencode=arch=compute_89,code=sm_89"` |
| RTX 5070/5080/5090 | Blackwell | `GPU_ARCH="-gencode=arch=compute_120,code=sm_120"` (default) |

### Build Options

```bash
# For RTX 4090:
make GPU_ARCH="-gencode=arch=compute_89,code=sm_89"

# Larger hash table (for 128GB+ RAM systems):
make V45_TABLE_BITS=33

# Disable cheap second point (for A/B benchmarking):
make USE_CHEAP_POINT=0
```

## Usage

### Basic (Puzzle #135 example)

```bash
./psckangaroo \
  -gpu 0 \
  -dp 20 \
  -range 135 \
  -ramlimit 115 \
  -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  -start 4000000000000000000000000000000000
```

Auto-checkpoint is enabled by default (every 4 hours). To disable: `-checkpoint 0`.

### Resume from checkpoint

```bash
./psckangaroo \
  -gpu 0 \
  -dp 20 \
  -range 135 \
  -ramlimit 115 \
  -loadwild wild_checkpoint.dat \
  -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  -start 4000000000000000000000000000000000
```

### ALL-WILD mode (alternative strategy)

```bash
./psckangaroo \
  -gpu 0 \
  -dp 20 \
  -range 135 \
  -ramlimit 115 \
  -allwild 1 \
  -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  -start 4000000000000000000000000000000000
```

### Command-Line Options

| Option | Description | Default |
|---|---|---|
| `-gpu N` | GPU index | 0 |
| `-dp N` | Distinguished point bits (6–60) | — |
| `-range N` | Key range in bits (32–170) | — |
| `-pubkey <hex>` | Target compressed public key | — |
| `-start <hex>` | Range start offset | — |
| `-ramlimit N` | RAM limit in GB | — |
| `-allwild 0/1` | 0 = ALL-TAME (default), 1 = ALL-WILD | 0 |
| `-groups N` | Points per batch inversion (8–256) | 24 |
| `-checkpoint N` | Auto-save interval in hours (0 = off) | 4 |
| `-savefile <f>` | Checkpoint filename | `wild_checkpoint.dat` |
| `-loadwild <f>` | Load checkpoint and resume | — |
| `-savedps <f>` | Save evicted DPs to file | — |
| `-waveinterval N` | Minutes between WILD wave renewals (0 = off) | 0 |
| `-resonant 0/1` | Resonant WILD spawning | 0 |
| `-rotation 0/1` | Table freeze (0) or rotation (1) | 0 |

## Performance

Approximate speeds (Puzzle 135, DP 20, ALL-TAME mode):

| GPU | Speed | Notes |
|---|---|---|
| RTX 5070 (Blackwell) | ~2.0 GKeys/s | sm_120, tested |
| RTX 4090 (Ada) | ~3.5 GKeys/s | sm_89, estimated |
| RTX 3090 (Ampere) | ~1.5 GKeys/s | sm_86, estimated |

Actual performance depends on DP value, RAM configuration, and kernel tuning parameters.

## Validation

You can validate PSCKangaroo by solving already-solved Bitcoin puzzles. Puzzle #70 (69-bit range) is a good quick test — it should solve in about 2 minutes on a modern GPU:

```bash
./psckangaroo -gpu 0 -dp 14 -range 69 -ramlimit 4 -checkpoint 0 \
  -pubkey 0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483 \
  -start 200000000000000000
```

Expected output: the program fills the TAME table (~30 seconds), switches to HUNT mode, and finds the key via BSGS collision resolution. The known private key for Puzzle #70 is `349b84b6431a6c4ef1`.

For a longer test, Puzzle #80 (79-bit range) takes roughly 30 minutes:

```bash
./psckangaroo -gpu 0 -dp 14 -range 79 -ramlimit 60 -checkpoint 0 \
  -pubkey 037e1238f7b1ce757df94faa9a2eb261bf0aeb9f84dbf81212104e78931c2a19dc \
  -start 80000000000000000000
```

**Note:** During the HUNT phase, hash false positives (FP) are expected and counted in the stats. These are harmless x_sig truncation collisions that fail verification — not bugs. The BSGS async resolver handles real collisions correctly.

## How It Works

The Pollard's Kangaroo algorithm solves ECDLP by launching two types of pseudo-random walks on the elliptic curve:

1. **TAME kangaroos** start from known positions (k*G where k is known)
2. **WILD kangaroos** start from positions derived from the target public key

When a TAME and WILD kangaroo land on the same point (collision), the private key can be computed from the difference of their accumulated distances.

**Distinguished Points (DPs)** are positions where the x-coordinate has a specific number of leading zero bits. Only DPs are stored, reducing memory requirements by a factor of 2^DP while preserving collision detection capability.

The **SOTA method** (by RetiredCoder) uses equivalence classes and the negation map to reduce the expected number of steps from ~2.08√n to ~1.15√n, where n is the range size.

## Changelog

### v56C (current)
- Ultra-compact 16-byte DP entries (+56% RAM capacity)
- Async BSGS resolver (4 threads, queue depth 4096, precomputed baby table)
- Checkpoint format RCKDT5C

### v55
- Fixed endomorphism constants: BETA2, LAMBDA, LAMBDA2 (verified against bitcoin-core/secp256k1)
- Fixed MulLambdaModN sign-extension bug
- NormDistForLambda corrected

### v54
- SOTA+ Cheap Second Point: observes P-J during P+J for ~2x DP rate
- Validated by solving Puzzle 79 (known answer)

### v53
- ALL-TAME mode: dedicates all RAM to TAMEs for maximum T-W probability
- Table freeze: prevents false-positive explosion on long runs

### v52
- Reverted to GroupCnt=24 (GroupCnt=200 caused catastrophic register spill)
- Reverted occupancy to 1 block/SM (256 regs/thread optimal)

### Earlier versions
- XDP (Extended Distinguished Points) with configurable multiplier
- Dual hash table with cross-type collision detection
- GPU Bloom filter (byte-index bug fixed)
- Uniform jumps (19% faster than stratified, benchmarked)
- Shard-locked reads for concurrent access safety

## Credits

- **[RetiredCoder (RC)](https://github.com/RetiredC)** — Original RCKangaroo, SOTA method, SOTA+ Cheap Second Point theory, GPU kernel architecture, batch Montgomery inversion. The vast majority of the code and all core algorithmic innovations are his work.
- **[JeanLucPons](https://github.com/JeanLucPons)** — Foundational Kangaroo/VanitySearch/BSGS implementations that inspired the ecosystem.
- **PSC** — Bug fixes (endomorphism constants, sign-extension, Bloom filter), ALL-TAME mode, XDP, ultra-compact DP format, async BSGS resolver, table freeze, checkpoint system, and various integrations.

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file.

This is a derivative work of [RCKangaroo](https://github.com/RetiredC/RCKangaroo) which is also GPLv3.

## Disclaimer

This software is provided for educational and research purposes. Use responsibly and in accordance with applicable laws. The authors assume no liability for any use of this software.
