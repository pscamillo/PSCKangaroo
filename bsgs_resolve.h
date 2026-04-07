// bsgs_resolve.h
// Baby-Step Giant-Step resolver for v56C truncated distance collisions
//
// v56C-OPT: THREE key optimizations:
//   1. PRECOMPUTED baby table — built once at startup (~300ms), reused forever.
//      Saves ~50ms per BSGS_ResolveKey call (was rebuilt every call).
//   2. TARGETED lambda — uses endo_t/endo_w to compute exact lambda exponent,
//      reducing from 18 BSGS calls to 2 (targeted) + 4 (fallback) = 6 max.
//   3. Multi-thread safe — baby table is read-only after init, naturally thread-safe.
//
// Algorithm:
// 1. Compute diff = target - k_approx*G (this equals delta*G)
// 2. Baby steps: precompute i*G for i=0..sqrt(range), store x-coords in hash table
// 3. Giant steps: check diff +/- j*sqrt(range)*G against baby table
// 4. If match: delta = i + j*sqrt(range), so k = k_approx + delta

#pragma once

#include <unordered_map>
#include <cstring>
#include <cstdio>
#include <chrono>
#include "Ec.h"

// Configuration
#define BSGS_TRUNC_BITS     32          // bits truncated from distance
#define BSGS_HALF_BITS      17          // ceil((TRUNC_BITS+1)/2) 
#define BSGS_BABY_STEPS     (1 << BSGS_HALF_BITS)  // 131072
#define BSGS_MAX_GIANTS     ((1 << (BSGS_TRUNC_BITS + 1 - BSGS_HALF_BITS)) + 2)

// Forward declaration (defined in RCKangaroo_hunt_v2.cpp)
void ApplyLambda(EcInt& d, int exp);

// =============================================================================
// PRECOMPUTED BSGS CONTEXT
// Built once at startup, shared read-only across all BSGS threads.
// =============================================================================
struct BSGSContext {
    std::unordered_map<u64, u32> baby_table;
    EcPoint G;
    EcPoint giant_step;     // BABY_STEPS * G
    EcPoint neg_giant;      // -(BABY_STEPS * G)
    u32 max_giants;
    bool initialized;
    
    BSGSContext() : initialized(false), max_giants(0) {}
    
    void Init(Ec& ec) {
        if (initialized) return;
        
        auto t_start = std::chrono::steady_clock::now();
        printf("Precomputing BSGS baby table (%d entries)...\n", BSGS_BABY_STEPS);
        
        // Compute G (generator)
        EcInt one;
        memset(one.data, 0, sizeof(one.data));
        one.data[0] = 1;
        G = ec.MultiplyG(one);
        
        // Compute 2*G for safe start
        EcInt two;
        memset(two.data, 0, sizeof(two.data));
        two.data[0] = 2;
        EcPoint twoG = ec.MultiplyG(two);
        
        // Build baby table: i*G for i=1..BABY_STEPS
        baby_table.reserve(BSGS_BABY_STEPS + 16);
        EcPoint baby = G;  // 1*G
        baby_table[baby.x.data[0]] = 1;
        
        baby = twoG;  // 2*G
        baby_table[baby.x.data[0]] = 2;
        
        for (u32 i = 3; i <= BSGS_BABY_STEPS; i++) {
            baby = ec.AddPoints(baby, G);  // i*G
            baby_table[baby.x.data[0]] = i;
        }
        
        // Compute giant step = BABY_STEPS * G
        EcInt gs_dist;
        memset(gs_dist.data, 0, sizeof(gs_dist.data));
        gs_dist.data[0] = BSGS_BABY_STEPS;
        giant_step = ec.MultiplyG(gs_dist);
        neg_giant = giant_step;
        neg_giant.y.NegModP();
        
        max_giants = BSGS_MAX_GIANTS;
        
        auto t_end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        printf("BSGS baby table ready: %d entries, giant range +/-2^%d, built in %lld ms\n", 
               BSGS_BABY_STEPS, BSGS_TRUNC_BITS, (long long)ms);
        
        initialized = true;
    }
};

// Global context -- initialized once, read by all threads
static BSGSContext g_bsgs;

static void InitBSGS(Ec& ec) {
    g_bsgs.Init(ec);
}

// =============================================================================
// BSGS RESOLVE -- uses precomputed baby table (thread-safe, read-only)
// =============================================================================
static bool BSGS_ResolveKey(Ec& ec, EcInt k_approx, EcPoint& target, EcInt& k_out) {
    // Step 0: Quick check -- is k_approx already exact?
    EcPoint P_base = ec.MultiplyG(k_approx);
    if (P_base.x.IsEqual(target.x)) {
        k_out = k_approx;
        return true;
    }
    
    // Step 1: Compute diff = target - P_base = delta * G
    EcPoint neg_base = P_base;
    neg_base.y.NegModP();
    EcPoint diff = ec.AddPoints(target, neg_base);
    
    // Step 2: Search positive direction using precomputed baby table
    // For j=0..max: compute diff - j*giant_step, look up in baby table
    // If diff - j*GS = i*G then delta = i + j*BABY_STEPS
    u32 max_giants = g_bsgs.max_giants;
    
    EcPoint search = diff;  // j=0
    for (u32 j = 0; j <= max_giants; j++) {
        auto it = g_bsgs.baby_table.find(search.x.data[0]);
        if (it != g_bsgs.baby_table.end()) {
            // Candidate: delta = i + j * BABY_STEPS
            u64 delta = (u64)it->second + (u64)j * (u64)BSGS_BABY_STEPS;
            
            EcInt k_test = k_approx;
            EcInt delta_int;
            memset(delta_int.data, 0, sizeof(delta_int.data));
            delta_int.data[0] = delta;
            k_test.Add(delta_int);
            
            EcPoint P_test = ec.MultiplyG(k_test);
            if (P_test.x.IsEqual(target.x)) {
                k_out = k_test;
                return true;
            }
        }
        if (j < max_giants) {
            search = ec.AddPoints(search, g_bsgs.neg_giant);  // diff - (j+1)*GS
        }
    }
    
    // Step 3: Search negative direction
    // For j=1..max: compute diff + j*giant_step, look up in baby table
    search = ec.AddPoints(diff, g_bsgs.giant_step);  // diff + 1*GS
    for (u32 j = 1; j <= max_giants; j++) {
        auto it = g_bsgs.baby_table.find(search.x.data[0]);
        if (it != g_bsgs.baby_table.end()) {
            // Candidate: delta = i - j * BABY_STEPS (could be negative)
            i64 delta = (i64)it->second - (i64)j * (i64)BSGS_BABY_STEPS;
            
            EcInt k_test = k_approx;
            if (delta >= 0) {
                EcInt d;
                memset(d.data, 0, sizeof(d.data));
                d.data[0] = (u64)delta;
                k_test.Add(d);
            } else {
                EcInt d;
                memset(d.data, 0, sizeof(d.data));
                d.data[0] = (u64)(-delta);
                k_test.Sub(d);
            }
            
            EcPoint P_test = ec.MultiplyG(k_test);
            if (P_test.x.IsEqual(target.x)) {
                k_out = k_test;
                return true;
            }
        }
        if (j < max_giants) {
            search = ec.AddPoints(search, g_bsgs.giant_step);  // diff + (j+1)*GS
        }
    }
    
    return false;  // Not found -- hash false positive
}

// =============================================================================
// COLLISION_BSGS -- TARGETED LAMBDA (6 calls max instead of 18)
//
// Math: canonical collision means phi^(endo_t)(P_tame) = +/-phi^(endo_w)(P_wild)
// This gives: k = lambda^adj * d_t +/- d_w +/- HalfRange
// where adj = (endo_t - endo_w + 3) % 3
//
// tryFormula handles +/-HalfRange and +/-negation (2 BSGS calls).
// We try targeted adj first, then fallback to other 2 values = 6 max.
// For REAL collisions: resolves in 2 calls (targeted hit).
// For FPs: exhausts all 6 and returns false.
// =============================================================================
static bool Collision_BSGS(Ec& ec, EcPoint& pntToSolve, 
                           EcInt t, int TameType, int endo_t,
                           EcInt w, int WildType, int endo_w,
                           EcInt& HalfRange, EcInt& privKeyOut,
                           bool show_debug = false) {
    
    // =========================================================================
    // WILD-WILD collision: k = HalfRange + (w2 - w1) / 2
    // Both distances are truncated, error bounded by ±2^31 → within BSGS range
    // =========================================================================
    if (TameType != 0 && WildType != 0) {
        // Identify WILD1 and WILD2
        EcInt dist_w1, dist_w2;
        if (TameType == 1 && WildType == 2) {
            dist_w1 = t; dist_w2 = w;
        } else {
            dist_w1 = w; dist_w2 = t;
        }
        
        // Try both orderings: (w2-w1)/2 and (w1-w2)/2
        for (int sign = 0; sign < 2; sign++) {
            EcInt diff;
            if (sign == 0) { diff = dist_w2; diff.Sub(dist_w1); }
            else           { diff = dist_w1; diff.Sub(dist_w2); }
            
            bool neg = (diff.data[4] >> 63) != 0;
            EcInt halfDiff = diff;
            if (neg) halfDiff.Neg();
            halfDiff.ShiftRight(1);
            
            // v57c FIX: k = HalfRange + (w2-w1)/2 — NO gStart!
            // Same as T-W formulas. PntA = (k-HalfRange)*G, distances are relative to k.
            EcInt k_approx = HalfRange;
            if (neg) k_approx.Sub(halfDiff);
            else     k_approx.Add(halfDiff);
            
            EcInt k_found;
            if (BSGS_ResolveKey(ec, k_approx, pntToSolve, k_found)) {
                privKeyOut = k_found;
                return true;
            }
        }
        return false;
    }
    
    // =========================================================================
    // TAME-WILD collision: original path
    // =========================================================================
    int wildType = (TameType == 0) ? WildType : TameType;
    bool isWild2 = (wildType == 2);
    
    // Ensure t is TAME, w is WILD
    EcInt t_use = t, w_use = w;
    int endo_t_use = endo_t, endo_w_use = endo_w;
    if (TameType != 0) {
        t_use = w; w_use = t;
        endo_t_use = endo_w; endo_w_use = endo_t;
    }
    
    // Lambda for each formula variant (handles WILD1/WILD2 + negation)
    auto tryFormula = [&](EcInt t_adj, EcInt w_adj, bool wild2) -> bool {
        EcInt k_approx;
        
        if (wild2) {
            // k = HalfRange + w - t
            k_approx = HalfRange;
            k_approx.Add(w_adj);
            k_approx.Sub(t_adj);
        } else {
            // k = t - w + HalfRange
            k_approx = t_adj;
            k_approx.Sub(w_adj);
            k_approx.Add(HalfRange);
        }
        
        EcInt k_found;
        if (BSGS_ResolveKey(ec, k_approx, pntToSolve, k_found)) {
            privKeyOut = k_found;
            return true;
        }
        
        // Try negation variant
        if (wild2) {
            // k = HalfRange + w + t
            k_approx = HalfRange;
            k_approx.Add(w_adj);
            k_approx.Add(t_adj);
        } else {
            // k = HalfRange - t - w
            k_approx = HalfRange;
            k_approx.Sub(t_adj);
            k_approx.Sub(w_adj);
        }
        
        if (BSGS_ResolveKey(ec, k_approx, pntToSolve, k_found)) {
            privKeyOut = k_found;
            return true;
        }
        
        return false;
    };
    
    // =========================================================================
    // TARGETED LAMBDA: compute exact adjustment from endo values
    // adj = (endo_t - endo_w + 3) % 3
    // For real collisions, this is the CORRECT formula -- resolves in 2 calls.
    // =========================================================================
    int adj_targeted = (endo_t_use - endo_w_use + 3) % 3;
    
    {
        EcInt t_adj = t_use;
        if (adj_targeted > 0) ApplyLambda(t_adj, adj_targeted);
        if (tryFormula(t_adj, w_use, isWild2)) return true;
    }
    
    // =========================================================================
    // FALLBACK: try remaining 2 lambda values (handles edge cases / FP exhaust)
    // For FPs: all 3 fail -> 6 total BSGS calls (vs 18 before = 3x faster)
    // =========================================================================
    for (int adj = 0; adj < 3; adj++) {
        if (adj == adj_targeted) continue;  // already tried
        
        EcInt t_adj = t_use;
        if (adj > 0) ApplyLambda(t_adj, adj);
        if (tryFormula(t_adj, w_use, isWild2)) return true;
    }
    
    return false;
}
