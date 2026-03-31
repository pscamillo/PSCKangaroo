// defs.h
// PSCKangaroo — A fork of RCKangaroo by RetiredCoder
// Original: https://github.com/RetiredC/RCKangaroo
// License: GPLv3
//
// Key changes from original RCKangaroo:
//   v56C: Ultra-compact 16-byte entries (+56% capacity) + Async BSGS resolver
//   v55:  Fixed endomorphism constants (beta², lambda, lambda²)
//   v54:  SOTA+ Cheap Second Point (2x DP generation rate)
//   v53:  ALL-TAME mode (all RAM for TAMEs, WILDs check-only)
//   v52:  Reverted occupancy/group tuning, fixed register spill
//   v48:  Corrected W-W collision handling + Hybrid preload mode
//   v47:  Tame-Wild mode fix (TRAP/HUNT with dual table)
//   v45:  GPU Occupancy tuning + Large Table support
//   XDP:  Extended Distinguished Points (threshold-based, configurable multiplier)

// =============================================================================
// V46 STRATIFIED JUMP CONFIGURATION
// =============================================================================
// Instead of uniform jump sizes, split the jump table into:
//   - Standard entries (80%): normal scale ~2^(Range/2+3)
//   - Explorer entries (20%): larger scale for long-range coverage
//
// The jump selection (x[0] % JMP_CNT) is deterministic per point,
// so two kangaroos at the same point still follow the same path.
// This preserves collision detection correctness.
//
// V46_EXPLORER_PCT:   % of JMP_CNT entries that are explorers (0-40)
// V46_EXPLORER_SHIFT: extra bits added to explorer jump scale (1-6)
//   shift=2 → 4x bigger jumps, shift=3 → 8x bigger
// V46_ENABLE: 1 to enable stratified jumps, 0 for v45 behavior
// =============================================================================
#ifndef V46_ENABLE
#define V46_ENABLE 1
#endif

#ifndef V46_EXPLORER_PCT
#define V46_EXPLORER_PCT 20
#endif

#ifndef V46_EXPLORER_SHIFT
#define V46_EXPLORER_SHIFT 2
#endif

#pragma once

// =============================================================================
// XDP CONFIGURATION (Extended Distinguished Points)
// =============================================================================
// XDP_MULT: Multiplier for DP rate (must be power of 2: 1, 2, 4, 8, 16)
//
// RECOMMENDATIONS:
// - DP 30-34: XDP_MULT = 4  (4x more DPs)
// - DP 35-38: XDP_MULT = 8  (8x more DPs) [RECOMMENDED FOR PUZZLE 135]
// - DP 39+:   XDP_MULT = 16 (16x more DPs)
//
// With low DP (<28), XDP is not needed (already enough DPs)
// With high DP (>40), consider XDP 16x
// =============================================================================

#ifndef XDP_MULT
#define XDP_MULT 8   // Default: 8x more DPs (recommended for DP 35-38)
#endif

// Validation: XDP_MULT must be a power of 2
#if (XDP_MULT != 1) && (XDP_MULT != 2) && (XDP_MULT != 4) && (XDP_MULT != 8) && (XDP_MULT != 16)
#error "XDP_MULT must be 1, 2, 4, 8, or 16"
#endif

// =============================================================================
// XDP MACRO - Distinguished Point check with threshold
// =============================================================================
// Replaces: (x3 & dp_mask) == 0     (accepts only pattern 0)
// With:       (x3 >> shift) < XDP_MULT (accepts patterns 0 to XDP_MULT-1)
//
// Mathematically equivalent to accepting XDP_MULT patterns instead of 1,
// increasing DP rate by XDP_MULT times.
// =============================================================================

#define IS_DP_XDP(x3, dp_bits) \
    (((x3) >> (64 - (dp_bits))) < XDP_MULT) 

// =============================================================================
// V45 CONFIGURATION - GPU OCCUPANCY & TABLE SIZE
// =============================================================================
// V45_OCCUPANCY: Target blocks per SM (1 = original, 2 = double occupancy)
//   - Value 1: 256 threads/SM = 12.5% occupancy (v44 default)
//   - Value 2: 512 threads/SM = 25% occupancy (v45 target)
//   - Value 3: 768 threads/SM = 37.5% occupancy (experimental)
//   Higher occupancy hides memory latency better but may spill registers.
//   RTX 5070 Blackwell: 65536 regs/SM, 256 threads × 2 = 128 regs/thread
// =============================================================================
#ifndef V45_OCCUPANCY
#define V45_OCCUPANCY 1   // v52: Reverted to 1 (256 regs/thread vs 128 with occ=2)
#endif

// V45_TABLE_BITS: Hash table size = 2^V45_TABLE_BITS entries per sub-table
//   - 32 = 4.29B entries × 2 tables ≈ 22 GB/table (v44 default)
//   - 33 = 8.59B entries × 2 tables ≈ 44 GB/table (for 128GB+ RAM systems)
//   Total RAM ≈ 2 × 2^V45_TABLE_BITS × 25 bytes
#ifndef V45_TABLE_BITS
#define V45_TABLE_BITS 32
#endif

// =============================================================================
// V45 STEP CONFIGURATION  
// =============================================================================
// STEP_CNT tuning: More steps per kernel = less launch overhead
// but more time between DP checks. Original: 1000
#ifndef V45_STEP_CNT
#define V45_STEP_CNT 1000
#endif

#pragma warning(disable : 4996)

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef short i16;
typedef unsigned char u8;
typedef char i8;



#define MAX_GPU_CNT			32

//must be divisible by MD_LEN
#define STEP_CNT			V45_STEP_CNT

#define JMP_CNT				512

//use different options for cards older than RTX 40xx
#ifdef __CUDA_ARCH__
	#if __CUDA_ARCH__ < 890
		#define OLD_GPU
	#endif
	#ifdef OLD_GPU
		#define BLOCK_SIZE			512
		//can be 8, 16, 24, 32, 40, 48, 56, 64
		#define PNT_GROUP_CNT		64	
	#else
		#define BLOCK_SIZE			256
		// v45: PNT_GROUP_CNT configurable at compile time
		// v52: Reverted to 24 (original). GroupCnt=200 caused 12KB register spill/thread
		//       and 6MB spill per SM — catastrophic L1 cache thrashing.
		//       GroupCnt=24: 512B spill fits in L1 (128KB). Speed: 2.7→3.1+ GKeys/s
		#ifndef V45_PNT_GROUP_CNT
		#define V45_PNT_GROUP_CNT	24
		#endif
		#define PNT_GROUP_CNT		V45_PNT_GROUP_CNT
	#endif
#else //CPU, fake values
	#define BLOCK_SIZE			512
	#ifndef V45_PNT_GROUP_CNT
	#define V45_PNT_GROUP_CNT	24
	#endif
	#define PNT_GROUP_CNT		V45_PNT_GROUP_CNT
#endif

// kang type
#define TAME				0  // Tame kangs
#define WILD1				1  // Wild kangs1 
#define WILD2				2  // Wild kangs2

#define GPU_DP_SIZE			48
// v52: Reduced from 16M to 1M. With GroupCnt=24, each kernel produces ~2K DPs (XDP 8x).
// 1M entries = ~450 kernel calls worth. Saves 720 MB GPU RAM.
// v53: Increased from 1M to 3M - canonical DP check generates ~3x more DPs
// v54 CHEAP: Increased from 3M to 5M - cheap second point generates ~2x more DPs
#define MAX_DP_CNT			(5 * 1024 * 1024)

#define JMP_MASK			(JMP_CNT-1)

#define DPTABLE_MAX_CNT		16

// v52: Reduced from 32M to 2M. With GroupCnt=24 + XDP 8x, ~2K DPs/kernel.
// 2M buffer = ~900 kernel calls. CPU buffer: 96 MB vs 1.5 GB before.
#define MAX_CNT_LIST		(2 * 1024 * 1024)

#define DP_FLAG				0x8000
#define INV_FLAG			0x4000
#define JMP2_FLAG			0x2000
#define CHEAP_FLAG			0x1000

// =============================================================================
// SOTA+ CHEAP SECOND POINT
// =============================================================================
// When computing P + J, we can also cheaply compute P - J because the
// modular inverse of (xP - xJ) is already available from batch inversion.
// Cost: ~1 AddModP + 1 MulModP + 1 SqrModP + 2 SubModP per group
// Benefit: ~2x DP generation rate → effective ~15-20% speedup
// The cheap point does NOT alter the kangaroo trajectory.
// It's a free "observation" at an opposite-direction position.
// =============================================================================
#ifndef USE_CHEAP_POINT
#define USE_CHEAP_POINT 1
#endif

#define MD_LEN				10

//#define DEBUG_MODE

//gpu kernel parameters
struct TKparams
{
	u64* Kangs;
	u32 KangCnt;
	u32 BlockCnt;
	u32 BlockSize;
	u32 GroupCnt;
	u64* L2;
	u64 DP;
	u32* DPs_out;
	u64* Jumps1; //x(32b), y(32b), d(32b)
	u64* Jumps2; //x(32b), y(32b), d(32b)
	u64* Jumps3; //x(32b), y(32b), d(32b)
	u64* JumpsList; //list of all performed jumps, grouped by warp(32) every 8 groups (from PNT_GROUP_CNT). Each jump is 2 bytes: 10bit jump index + flags: INV_FLAG, DP_FLAG, JMP2_FLAG
	u32* DPTable;
	
	// CRÍTICO: Alterado para u64 para suportar -groups 80 sem corromper memória
	u64* L1S2; 
	
	u64* LastPnts;
	u64* LoopTable;
	u32* dbg_buf;
	u32* LoopedKangs;
	bool IsGenMode; //tames generation mode
	bool AllWildsMode; //HUNT optimization: all kangaroos are WILDs
	
	// GPU-Side Bloom Filter
	u8* GpuBloom;           // Bloom filter in GPU memory (2GB)
	u64 GpuBloomSize;       // Size in bits
	bool UseGpuBloom;       // Enable GPU-side Bloom checking during HUNT

	u32 KernelA_LDS_Size;
	u32 KernelB_LDS_Size;
	u32 KernelC_LDS_Size;	
};
