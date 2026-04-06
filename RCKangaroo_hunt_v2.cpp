// PSCKangaroo — GPU-accelerated Pollard's Kangaroo for secp256k1 ECDLP
// Fork of RCKangaroo by RetiredCoder (RC)
// Original: https://github.com/RetiredC/RCKangaroo
// License: GPLv3
//
// Modes:
// ALL-TAME (default): Fill RAM with TAMEs, then hunt with 100% WILDs (T-W collisions)
// ALL-WILD (-allwild 1): Dual table W-W collisions only
// HYBRID: Preload checkpoint + live hunt

#include <iostream>
#include <vector>
#include <signal.h>
#include <cmath>
#include <ctime>   // v38.5 CHECKPOINT: for time()

#ifndef _WIN32
#include <sched.h>  // For sched_yield
#endif

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"

#include <atomic>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <deque>

#include "TameStore.h"
#include "bsgs_resolve.h"  // v56C: BSGS for truncated distance resolution
TameStore gTameStore;

// Multi-threading for DP processing
#define NUM_WORKER_THREADS 12

// Thread pool structures
struct WorkChunk {
    u8* data;
    int start;
    int end;
};

std::atomic<bool> gWorkersRunning{false};
std::mutex gWorkMutex;
std::condition_variable gWorkCV;
std::vector<WorkChunk> gWorkQueue;
std::atomic<int> gPendingChunks{0};
std::vector<std::thread> gWorkerThreads;

// ============================================================================
// v56C ASYNC BSGS QUEUE — MULTI-THREADED
// Workers push collision candidates; dedicated threads resolve via BSGS.
// Queue has max size — oldest entries evicted (real collisions reappear).
// v56C-OPT: 8 threads + queue 4096 + precomputed baby table
// ============================================================================
#define BSGS_QUEUE_MAX 4096  // Queue depth (was 64 — dropped 99% of candidates!)
#define BSGS_NUM_THREADS 4   // Parallel BSGS resolver threads

struct BSGSCandidate {
    EcInt t, w;
    int Type1, Type2;
    int endo_1, endo_2;
    u8 dist1_raw[24];
    u8 dist2_raw[24];
};

std::mutex gBSGSMutex;
std::condition_variable gBSGSCV;
std::deque<BSGSCandidate> gBSGSQueue;
std::thread gBSGSThreads[BSGS_NUM_THREADS];
std::atomic<bool> gBSGSRunning{false};
std::atomic<u64> gBSGSProcessed{0};
std::atomic<u64> gBSGSDropped{0};

// Forward declarations (implementations after global declarations)
void BSGSConsumerThread();
void PushBSGSCandidate(const EcInt& t, const EcInt& w, int Type1, int Type2,
                        int endo_1, int endo_2, const u8* d1, const u8* d2);
void StartBSGSThread();
void StopBSGSThread();

// Macros for kang type
#define GET_KANG_TYPE(t) ((t) & 0x03)
#define GET_ENDO_TRANSFORM(t) (((t) >> 4) & 0x03)

// Forward declarations
extern volatile bool gSolved;
extern EcPoint gPntToSolve;
extern volatile bool gTrapPhase;
extern bool gAllWildMode;  // v37: ALL-WILD mode (defined later)
extern EcInt Int_HalfRange;  // v38: Forward declaration for WorkerThread
extern EcPoint Pnt_HalfRange; // v39.4.1: Forward declaration for collision verification
extern Ec ec;                // v38: Forward declaration for WorkerThread
extern EcInt gPrivKey;       // v38: Forward declaration for WorkerThread
extern u32 gRange;           // v38: Forward declaration for WorkerThread
extern EcInt gStart;         // v39.4.1: Forward declaration for WILD-WILD resolution
extern EcInt g_N;            // v55: For NormDistForLambda
bool Collision_SOTA_Endo(EcPoint& pnt, EcInt t, int TameType, int endo_t, EcInt w, int WildType, int endo_w);

// v52 FIX: CPU-side canonical X comparison for endomorphism-aware W-W verification
// Beta constants for secp256k1: beta^3 = 1 mod p
static EcInt g_BetaCPU, g_Beta2CPU;
static bool g_BetaInit = false;

static void InitBetaCPU() {
    if (g_BetaInit) return;
    memset(&g_BetaCPU, 0, sizeof(EcInt));
    g_BetaCPU.data[0] = 0xC1396C28719501EEull;
    g_BetaCPU.data[1] = 0x9CF0497512F58995ull;
    g_BetaCPU.data[2] = 0x6E64479EAC3434E9ull;
    g_BetaCPU.data[3] = 0x7AE96A2B657C0710ull;
    memset(&g_Beta2CPU, 0, sizeof(EcInt));
    // v55 FIX: BETA2 was wrong! Corrected to actual beta^2 mod p
    g_Beta2CPU.data[0] = 0x3EC693D68E6AFA40ull;
    g_Beta2CPU.data[1] = 0x630FB68AED0A766Aull;
    g_Beta2CPU.data[2] = 0x919BB86153CBCB16ull;
    g_Beta2CPU.data[3] = 0x851695D49A83F8EFull;
    g_BetaInit = true;
}

// Compare canonical X: check if x1 and x2 are in the same beta-orbit
// i.e., x1 == beta^i * x2 for some i in {0, 1, 2}
static bool CanonicalXMatch(EcInt& x1, EcInt& x2) {
    // Fast path: raw X match (same endo variant, ~1/3 of collisions)
    if (x1.IsEqual(x2)) return true;
    
    InitBetaCPU();
    
    // Check x1 == beta * x2
    EcInt bx2 = g_BetaCPU; bx2.MulModP(x2);
    if (x1.IsEqual(bx2)) return true;
    
    // Check x1 == beta² * x2
    EcInt b2x2 = g_Beta2CPU; b2x2.MulModP(x2);
    if (x1.IsEqual(b2x2)) return true;
    
    return false;
}

// Statistics (atomic for thread safety)
std::atomic<u64> gTamesDPs{0};
std::atomic<u64> gWildsDPs{0};
std::atomic<u64> gWildCollisions{0};
std::atomic<u64> gHuntChecks{0};
std::atomic<u64> gFalsePositives{0};  // v40.1: Silent FP counter

// v39.4 FIX: Mutex to prevent multiple threads processing same collision
std::mutex gCollisionResolveMutex;
std::atomic<bool> gCollisionBeingResolved{false};

// Worker thread function - processes TRAP and HUNT
void WorkerThread(int thread_id) {
    while (gWorkersRunning.load()) {
        WorkChunk chunk = {nullptr, 0, 0};
        bool hasWork = false;
        
        {
            std::unique_lock<std::mutex> lock(gWorkMutex);
            if (gWorkCV.wait_for(lock, std::chrono::microseconds(100), []{
                return !gWorkQueue.empty() || !gWorkersRunning.load();
            })) {
                if (!gWorkQueue.empty()) {
                    chunk = gWorkQueue.back();
                    gWorkQueue.pop_back();
                    hasWork = true;
                }
            }
        }
        
        if (hasWork && chunk.data) {
            for (int i = chunk.start; i < chunk.end && !gSolved; i++) {
                u8* p = chunk.data + i * GPU_DP_SIZE;
                u8 x_bytes[16];
                u8 dist_bytes[24];
                
                memcpy(x_bytes, p, 12);
                memset(x_bytes + 12, 0, 4);
                memcpy(dist_bytes, p + 16, 22);
                
                u8 type_info = p[40];
                u8 kang_type = GET_KANG_TYPE(type_info);
                u8 endo_transform = GET_ENDO_TRANSFORM(type_info);
                
                if (gTrapPhase) {
                    // TRAP PHASE: Store DPs into tables
                    // v51: Real TAME mode — GPU generates TAMEs (IsGenMode=true, no pubkey offset)
                    //       Force type=TAME but PRESERVE endo_transform in high bits
                    dist_bytes[22] = (endo_transform << 4) | TAME;  // v51: type=TAME + endo
                    dist_bytes[23] = 0;
                    
                    gTameStore.AddTame(x_bytes, dist_bytes);
                    gTamesDPs++;
                }
                else if (kang_type == WILD1 || kang_type == WILD2) {
                    // HUNT PHASE: Check WILD against stored entries
                    gHuntChecks++;
                    dist_bytes[22] = type_info;
                    dist_bytes[23] = 0;
                    
                    // v52: Use correct function for each mode:
                    //   allwild: CheckWildOnly stores WILD1→table[0], WILD2→table[1]
                    //   TAME/HUNT: CheckWild stores all WILDs→table[1] (protects TAMEs in table[0])
                    int result;
                    if (gAllWildMode) {
                        result = gTameStore.CheckWildOnly(x_bytes, dist_bytes);
                    } else {
                        result = gTameStore.CheckWild(x_bytes, dist_bytes);
                    }
                    
                    if (result == COLLISION_TAME_WILD || result == COLLISION_WILD_WILD) {
                        gWildCollisions++;
                        
                        // v39.4 FIX: Only ONE thread should process each collision
                        // Try to acquire the resolution lock - if another thread is already resolving, skip
                        bool expected = false;
                        if (!gCollisionBeingResolved.compare_exchange_strong(expected, true)) {
                            // Another thread is already resolving this collision
                            continue;
                        }
                        
                        // We have exclusive access to resolve this collision
                        std::lock_guard<std::mutex> lock(gCollisionResolveMutex);
                        
                        // Double-check the collision is still valid (wasn't cleared by another thread)
                        if (!gTameStore.HasCollision()) {
                            gCollisionBeingResolved.store(false);
                            continue;
                        }
                        
                        u8 dist1_data[24], dist2_data[24];
                        int coll_type;
                        if (!gTameStore.GetCollisionData(dist1_data, dist2_data, &coll_type)) {
                             gCollisionBeingResolved.store(false);
                             continue;
                        }
                        
                        // Clear collision IMMEDIATELY to prevent other threads from seeing it
                        gTameStore.ClearCollision();
                        
                        EcInt t, w;
                        memset(t.data, 0, sizeof(t.data));
                        memset(w.data, 0, sizeof(w.data));
                        memcpy(t.data, dist1_data, 22);
                        memcpy(w.data, dist2_data, 22);
                        
                        // Keep unsigned copies for alternative calculations
                        EcInt t_unsigned = t;
                        EcInt w_unsigned = w;
                        
                        // Sign extension - kangaroo distances can be negative if they walked "backwards"
                        // (more subtractions than additions in jumps)
                        if (dist1_data[21] & 0x80) memset(((u8*)t.data) + 22, 0xFF, 18);
                        if (dist2_data[21] & 0x80) memset(((u8*)w.data) + 22, 0xFF, 18);
                        
                        u8 type1_info = dist1_data[22];
                        u8 type2_info = dist2_data[22];
                        int Type1 = GET_KANG_TYPE(type1_info);
                        int Type2 = GET_KANG_TYPE(type2_info);
                        int endo_1 = GET_ENDO_TRANSFORM(type1_info);
                        int endo_2 = GET_ENDO_TRANSFORM(type2_info);
                        
                        // v51.1: Label based on actual types, not table index
                        const char* collision_str = (Type1 == TAME || Type2 == TAME) ? "TAME-WILD" : "WILD-WILD";
                        
                        // Debug output disabled for clean output. Uncomment to enable:
                        // static std::atomic<int> gDebugCollisionCount{0};
                        // bool show_debug = (gDebugCollisionCount.fetch_add(1) < 5);
                        bool show_debug = false;
                        
                        // v38 FIX: Use LOCAL variable instead of global gPrivKey to avoid race condition!
                        EcInt localPrivKey;
                        bool resolved = false;
                        
                        // TAME-WILD collision math:
                        // TAME: starts at G, walks distance t → point = t*G
                        // WILD1 (type=1): starts at PntA = (k-HalfRange)*G, walks w → point = (k-HalfRange+w)*G
                        // WILD2 (type=2): starts at PntB = -(k-HalfRange)*G = (HalfRange-k)*G, walks w → point = (HalfRange-k+w)*G
                        //
                        // For WILD1 collision: t = k - HalfRange + w  →  k = t - w + HalfRange
                        // For WILD2 collision: t = HalfRange - k + w  →  k = HalfRange + w - t  (DIFFERENT!)
                        //
                        // Also consider negation (same X, opposite Y):
                        // For WILD1: -t = k - HalfRange + w  →  k = -t - w + HalfRange
                        // For WILD2: -t = HalfRange - k + w  →  k = HalfRange + w + t
                        
                        if (Type1 == TAME) {
                            EcInt diff;
                            
                            if (Type2 == WILD1) {
                                // WILD1: k = t - w + HalfRange
                                if(show_debug) printf("[DEBUG] WILD1 collision: trying k = t - w + HalfRange\n");
                                localPrivKey = t;
                                localPrivKey.Sub(w);
                                localPrivKey.Add(Int_HalfRange);
                                
                                if(show_debug) {
                                    printf("[DEBUG] Calculated k: ");
                                    for(int j=31; j>=0; j--) printf("%02X", ((u8*)localPrivKey.data)[j]);
                                    printf("\n");
                                    fflush(stdout);
                                }
                                
                                EcPoint P = ec.MultiplyG(localPrivKey);
                                
                                if(show_debug) {
                                    printf("[DEBUG] Calculated P.x: ");
                                    for(int j=31; j>=0; j--) printf("%02X", ((u8*)P.x.data)[j]);
                                    printf("\n[DEBUG] Target PntToSolve.x: ");
                                    for(int j=31; j>=0; j--) printf("%02X", ((u8*)gPntToSolve.x.data)[j]);
                                    printf("\n");
                                    fflush(stdout);
                                }
                                
                                if (P.IsEqual(gPntToSolve)) {
                                    gPrivKey = localPrivKey;
                                    resolved = true;
                                } else {
                                    // Try negation: k = HalfRange - t - w
                                    if(show_debug) printf("[DEBUG] Trying k = HalfRange - t - w\n");
                                    localPrivKey = Int_HalfRange;
                                    localPrivKey.Sub(t);
                                    localPrivKey.Sub(w);
                                    
                                    P = ec.MultiplyG(localPrivKey);
                                    if (P.IsEqual(gPntToSolve)) {
                                        gPrivKey = localPrivKey;
                                        resolved = true;
                                    }
                                }
                            } else {
                                // WILD2: k = HalfRange + w - t
                                if(show_debug) printf("[DEBUG] WILD2 collision: trying k = HalfRange + w - t\n");
                                localPrivKey = Int_HalfRange;
                                localPrivKey.Add(w);
                                localPrivKey.Sub(t);
                                
                                if(show_debug) {
                                    printf("[DEBUG] Calculated k: ");
                                    for(int j=31; j>=0; j--) printf("%02X", ((u8*)localPrivKey.data)[j]);
                                    printf("\n");
                                    fflush(stdout);
                                }
                                
                                EcPoint P = ec.MultiplyG(localPrivKey);
                                
                                if(show_debug) {
                                    printf("[DEBUG] Calculated P.x: ");
                                    for(int j=31; j>=0; j--) printf("%02X", ((u8*)P.x.data)[j]);
                                    printf("\n[DEBUG] Target PntToSolve.x: ");
                                    for(int j=31; j>=0; j--) printf("%02X", ((u8*)gPntToSolve.x.data)[j]);
                                    printf("\n");
                                    fflush(stdout);
                                }
                                
                                if (P.IsEqual(gPntToSolve)) {
                                    gPrivKey = localPrivKey;
                                    resolved = true;
                                } else {
                                    // Try: k = HalfRange + w + t
                                    if(show_debug) printf("[DEBUG] Trying k = HalfRange + w + t\n");
                                    localPrivKey = Int_HalfRange;
                                    localPrivKey.Add(w);
                                    localPrivKey.Add(t);
                                    
                                    if(show_debug) {
                                        printf("[DEBUG] Calculated k: ");
                                        for(int j=31; j>=0; j--) printf("%02X", ((u8*)localPrivKey.data)[j]);
                                        printf("\n");
                                        fflush(stdout);
                                    }
                                    
                                    P = ec.MultiplyG(localPrivKey);
                                    
                                    if(show_debug) {
                                        printf("[DEBUG] Calculated P.x: ");
                                        for(int j=31; j>=0; j--) printf("%02X", ((u8*)P.x.data)[j]);
                                        printf("\n");
                                        fflush(stdout);
                                    }
                                    
                                    if (P.IsEqual(gPntToSolve)) {
                                        gPrivKey = localPrivKey;
                                        resolved = true;
                                    }
                                }
                            }
                            
                            // If still not resolved, try all 4 original combinations as fallback
                            if (!resolved) {
                                if(show_debug) printf("[DEBUG] Type-specific failed, trying generic combinations...\n");
                                
                                // First try with UNSIGNED distances (no sign extension)
                                // This is often correct because distances accumulate from 0
                                
                                // Try: k = t_unsigned - w_unsigned + HalfRange
                                localPrivKey = t_unsigned;
                                localPrivKey.Sub(w_unsigned);
                                localPrivKey.Add(Int_HalfRange);
                                EcPoint P = ec.MultiplyG(localPrivKey);
                                if (P.IsEqual(gPntToSolve)) { 
                                    gPrivKey = localPrivKey; 
                                    resolved = true; 
                                    if(show_debug) printf("[DEBUG] Resolved with UNSIGNED: k = t - w + HalfRange!\n");
                                }
                                
                                if (!resolved) {
                                    // Try: k = w_unsigned - t_unsigned + HalfRange  
                                    localPrivKey = w_unsigned;
                                    localPrivKey.Sub(t_unsigned);
                                    localPrivKey.Add(Int_HalfRange);
                                    P = ec.MultiplyG(localPrivKey);
                                    if (P.IsEqual(gPntToSolve)) { 
                                        gPrivKey = localPrivKey; 
                                        resolved = true;
                                        if(show_debug) printf("[DEBUG] Resolved with UNSIGNED: k = w - t + HalfRange!\n");
                                    }
                                }
                                
                                if (!resolved) {
                                    // Try: k = t_unsigned - w_unsigned - HalfRange
                                    localPrivKey = t_unsigned;
                                    localPrivKey.Sub(w_unsigned);
                                    localPrivKey.Sub(Int_HalfRange);
                                    P = ec.MultiplyG(localPrivKey);
                                    if (P.IsEqual(gPntToSolve)) { 
                                        gPrivKey = localPrivKey; 
                                        resolved = true;
                                        if(show_debug) printf("[DEBUG] Resolved with UNSIGNED: k = t - w - HalfRange!\n");
                                    }
                                }
                                
                                if (!resolved) {
                                    // Try: k = w_unsigned - t_unsigned - HalfRange
                                    localPrivKey = w_unsigned;
                                    localPrivKey.Sub(t_unsigned);
                                    localPrivKey.Sub(Int_HalfRange);
                                    P = ec.MultiplyG(localPrivKey);
                                    if (P.IsEqual(gPntToSolve)) { 
                                        gPrivKey = localPrivKey; 
                                        resolved = true;
                                        if(show_debug) printf("[DEBUG] Resolved with UNSIGNED: k = w - t - HalfRange!\n");
                                    }
                                }
                                
                                if (!resolved) {
                                    // Try original with sign-extended w
                                    // k = t - w + HalfRange
                                    localPrivKey = t;
                                    localPrivKey.Sub(w);
                                    EcInt sv = localPrivKey;
                                    localPrivKey.Add(Int_HalfRange);
                                    P = ec.MultiplyG(localPrivKey);
                                    if (P.IsEqual(gPntToSolve)) { gPrivKey = localPrivKey; resolved = true; }
                                    
                                    if (!resolved) {
                                        // k = t - w - HalfRange
                                        localPrivKey = sv;
                                        localPrivKey.Sub(Int_HalfRange);
                                        P = ec.MultiplyG(localPrivKey);
                                        if (P.IsEqual(gPntToSolve)) { gPrivKey = localPrivKey; resolved = true; }
                                    }
                                    
                                    if (!resolved) {
                                        // k = w - t + HalfRange
                                        localPrivKey = w;
                                        localPrivKey.Sub(t);
                                        localPrivKey.Add(Int_HalfRange);
                                        P = ec.MultiplyG(localPrivKey);
                                        if (P.IsEqual(gPntToSolve)) { gPrivKey = localPrivKey; resolved = true; }
                                    }
                                    
                                    if (!resolved) {
                                        // k = w - t - HalfRange
                                        localPrivKey = w;
                                        localPrivKey.Sub(t);
                                        localPrivKey.Sub(Int_HalfRange);
                                        P = ec.MultiplyG(localPrivKey);
                                        if (P.IsEqual(gPntToSolve)) { gPrivKey = localPrivKey; resolved = true; }
                                    }
                                }
                            }
                        } else {
                            // WILD-WILD collision: WILD1 vs WILD2
                            // WILD1 starts at: (k - HalfRange) * G, walks dist_w1
                            // WILD2 starts at: (HalfRange - k) * G, walks dist_w2
                            // At collision: k - HalfRange + w1 = HalfRange - k + w2
                            // Therefore: k = HalfRange + (w2 - w1) / 2
                            //
                            // CRITICAL: Must identify which distance is WILD1 and which is WILD2!
                            
                            EcInt dist_w1, dist_w2;
                            
                            // Identify WILD1 and WILD2 distances correctly
                            if (Type1 == WILD1 && Type2 == WILD2) {
                                dist_w1 = t_unsigned;  // t is WILD1
                                dist_w2 = w_unsigned;  // w is WILD2
                            } else if (Type1 == WILD2 && Type2 == WILD1) {
                                dist_w1 = w_unsigned;  // w is WILD1  
                                dist_w2 = t_unsigned;  // t is WILD2
                            } else {
                                // Same type collision (shouldn't happen in normal operation)
                                // Try both orderings
                                dist_w1 = t_unsigned;
                                dist_w2 = w_unsigned;
                            }
                            
                            // v40.1: Verify collision is REAL FIRST (silently skip FPs)
                            {
                                EcPoint P1 = ec.MultiplyG(dist_w1);
                                EcPoint P2 = ec.MultiplyG(dist_w2);
                                
                                EcPoint NegHalfRange = Pnt_HalfRange;
                                NegHalfRange.y.NegModP();
                                
                                EcPoint PntA = Ec::AddPoints(gPntToSolve, NegHalfRange);
                                
                                P1 = Ec::AddPoints(P1, PntA);
                                
                                EcPoint PntB = PntA;
                                PntB.y.NegModP();
                                
                                P2 = Ec::AddPoints(P2, PntB);
                                
                                // v52 FIX: Use canonical X comparison (accounts for endomorphism)
                                // GPU stores DPs by canonical X = min(x, beta*x, beta²*x)
                                // Two kangaroos at endomorphism-related points have same canonical X
                                // but different raw X (differ by factor beta or beta²)
                                bool realCollision = CanonicalXMatch(P1.x, P2.x);
                                
                                if (!realCollision) {
                                    // W-W hash false positive (expected, silenced for clean output)
                                    gFalsePositives.fetch_add(1, std::memory_order_relaxed);
                                    gCollisionBeingResolved.store(false);
                                    continue;
                                }
                                
                                // REAL COLLISION - now print debug info
                                if(show_debug) printf("\n[DEBUG] *** REAL WILD-WILD COLLISION DETECTED! ***\n");
                                if(show_debug) printf("[DEBUG] Type1=%d, Type2=%d\n", Type1, Type2);
                            }
                            
                            // Formula: k = gStart + HalfRange + (dist_w2 - dist_w1) / 2
                            EcInt diff;
                            diff = dist_w2;
                            diff.Sub(dist_w1);
                            
                            if(show_debug) {
                                printf("[DEBUG] dist_w1: ");
                                for(int j=21; j>=0; j--) printf("%02X", ((u8*)dist_w1.data)[j]);
                                printf("\n[DEBUG] dist_w2: ");
                                for(int j=21; j>=0; j--) printf("%02X", ((u8*)dist_w2.data)[j]);
                                printf("\n[DEBUG] diff (w2-w1): ");
                                for(int j=21; j>=0; j--) printf("%02X", ((u8*)diff.data)[j]);
                                printf("\n");
                                fflush(stdout);
                            }
                            
                            // Check sign and handle accordingly
                            bool diffNegative = (diff.data[4] >> 63) != 0;
                            EcInt halfDiff = diff;
                            if (diffNegative) halfDiff.Neg();
                            halfDiff.ShiftRight(1);
                            
                            // v39.4 FIX: Formula is k = Start + HalfRange + (w2 - w1)/2
                            // Must include gStart when -start parameter is used!
                            localPrivKey = gStart;           // Start with gStart
                            localPrivKey.Add(Int_HalfRange); // Add HalfRange
                            if (diffNegative) {
                                localPrivKey.Sub(halfDiff);
                            } else {
                                localPrivKey.Add(halfDiff);
                            }
                            
                            if(show_debug) {
                                printf("[DEBUG] Trying k = Start + HalfRange %c halfDiff: ", diffNegative ? '-' : '+');
                                for(int j=31; j>=0; j--) printf("%02X", ((u8*)localPrivKey.data)[j]);
                                printf("\n");
                                fflush(stdout);
                            }
                            
                            EcPoint P = ec.MultiplyG(localPrivKey);
                            if (P.IsEqual(gPntToSolve)) {
                                gPrivKey = localPrivKey;
                                resolved = true;
                                if(show_debug) printf("[DEBUG] RESOLVED: k = Start + HalfRange %c |w2-w1|/2\n", diffNegative ? '-' : '+');
                            } else {
                                // Try opposite sign
                                localPrivKey = gStart;
                                localPrivKey.Add(Int_HalfRange);
                                if (diffNegative) {
                                    localPrivKey.Add(halfDiff);
                                } else {
                                    localPrivKey.Sub(halfDiff);
                                }
                                
                                if(show_debug) {
                                    printf("[DEBUG] Trying k = Start + HalfRange %c halfDiff: ", diffNegative ? '+' : '-');
                                    for(int j=31; j>=0; j--) printf("%02X", ((u8*)localPrivKey.data)[j]);
                                    printf("\n");
                                    fflush(stdout);
                                }
                                
                                P = ec.MultiplyG(localPrivKey);
                                if (P.IsEqual(gPntToSolve)) {
                                    gPrivKey = localPrivKey;
                                    resolved = true;
                                    if(show_debug) printf("[DEBUG] RESOLVED: k = Start + HalfRange %c |w2-w1|/2\n", diffNegative ? '+' : '-');
                                }
                            }
                            
                            // If still not resolved, try swapping WILD1/WILD2 interpretation
                            if (!resolved) {
                                if(show_debug) printf("[DEBUG] Trying swapped WILD1/WILD2 interpretation...\n");
                                diff = dist_w1;
                                diff.Sub(dist_w2);
                                
                                diffNegative = (diff.data[4] >> 63) != 0;
                                halfDiff = diff;
                                if (diffNegative) halfDiff.Neg();
                                halfDiff.ShiftRight(1);
                                
                                localPrivKey = gStart;
                                localPrivKey.Add(Int_HalfRange);
                                if (diffNegative) {
                                    localPrivKey.Sub(halfDiff);
                                } else {
                                    localPrivKey.Add(halfDiff);
                                }
                                
                                P = ec.MultiplyG(localPrivKey);
                                if (P.IsEqual(gPntToSolve)) {
                                    gPrivKey = localPrivKey;
                                    resolved = true;
                                } else {
                                    localPrivKey = gStart;
                                    localPrivKey.Add(Int_HalfRange);
                                    if (diffNegative) {
                                        localPrivKey.Add(halfDiff);
                                    } else {
                                        localPrivKey.Sub(halfDiff);
                                    }
                                    P = ec.MultiplyG(localPrivKey);
                                    if (P.IsEqual(gPntToSolve)) {
                                        gPrivKey = localPrivKey;
                                        resolved = true;
                                    }
                                }
                            }
                        }
                        
                        if (!resolved) {
                            // v52.1: Only try endo combinations when at least one endo != 0
                            // When both endos are 0, lambda adjustments cannot help — it's a hash FP.
                            // Also adds timeout guard to prevent worker thread hang.
                            if (endo_1 != 0 || endo_2 != 0) {
                                if(show_debug) printf("[DEBUG] Basic failed, trying endo combinations (endo1=%d, endo2=%d)...\n", endo_1, endo_2);
                                if(show_debug) fflush(stdout);
                                
                                // Timeout guard: measure resolution time
                                auto resolve_start = std::chrono::steady_clock::now();
                                resolved = Collision_SOTA_Endo(gPntToSolve, t, Type1, endo_1, w, Type2, endo_2);
                                auto resolve_end = std::chrono::steady_clock::now();
                                auto resolve_ms = std::chrono::duration_cast<std::chrono::milliseconds>(resolve_end - resolve_start).count();
                                if (resolve_ms > 100) {
                                    if(show_debug) printf("[DEBUG] WARNING: Collision_SOTA_Endo took %lld ms!\n", (long long)resolve_ms);
                                    if(show_debug) fflush(stdout);
                                }
                            } else {
                                if(show_debug) printf("[DEBUG] Both endos=0, skipping lambda combinations (hash FP)\n");
                                if(show_debug) fflush(stdout);
                            }
                        }
                        
                        // v56C: Push to async BSGS queue (non-blocking)
                        // Workers are never stalled by 400ms BSGS resolution.
                        // Real collisions reappear indefinitely; FPs are one-shot.
                        if (!resolved) {
                            PushBSGSCandidate(t, w, Type1, Type2, endo_1, endo_2,
                                            dist1_data, dist2_data);
                            // Don't set resolved — worker continues immediately
                        }

                        if (resolved) {
                            // v38: Verify k is within expected range [0, 2^Range)
                            // If k is outside this range, the collision is mathematically valid
                            // but doesn't solve THIS puzzle
                            EcInt maxRange;
                            maxRange.Set(1);
                            maxRange.ShiftLeft(gRange);  // 2^Range
                            
                            // Check if k is negative (bit 319 set) or >= 2^Range
                            bool isNegative = (gPrivKey.data[4] >> 63) != 0;
                            bool inRange = !isNegative && gPrivKey.IsLessThanU(maxRange);
                            
                            if (!inRange) {
                                if(show_debug) printf("[DEBUG] WARNING: k is outside expected range! Collision valid but doesn't solve puzzle.\n");
                                char kHex[100];
                                gPrivKey.GetHexStr(kHex);
                                if(show_debug) printf("[DEBUG] k = %s (isNegative=%d)\n", kHex, isNegative);
                                resolved = false;
                                gCollisionBeingResolved.store(false);
                                continue;
                            }
                            
                            gSolved = true;
                            gCollisionBeingResolved.store(false);
                            printf("\r\n");
                            printf("################################################################################\r\n");
                            printf("###                                                                          ###\r\n");
                            printf("###              *** PRIVATE KEY FOUND! ***                                  ###\r\n");
                            printf("###                                                                          ###\r\n");
                            printf("################################################################################\r\n");
                            printf("\r\n");
                            printf("  Collision type: %s\r\n", collision_str);
                            printf("  Dist1: ");
                            for(int j=21; j>=0; j--) printf("%02X", dist1_data[j]);
                            printf(" (type=%d, endo=%d)\r\n", Type1, endo_1);
                            printf("  Dist2: ");
                            for(int j=21; j>=0; j--) printf("%02X", dist2_data[j]);
                            printf(" (type=%d, endo=%d)\r\n", Type2, endo_2);
                            printf("\r\n");
                            printf("################################################################################\r\n");
                            if(show_debug) fflush(stdout);
                            break;
                        } 
                        else {
                            // v51.1: Silent false positive — x_sig hash collision, not a real EC collision
                            static int fp_resolve_count = 0;
                            fp_resolve_count++;
                            // Hash false positives are expected (x_sig truncation collisions).
                            // Silenced for clean output. Uncomment for debugging:
                            // printf("[RESOLVE-FAIL #%d] %s hash FP (type=%d/%d endo=%d/%d)\n",
                            //        fp_resolve_count, collision_str, Type1, Type2, endo_1, endo_2);
                            gFalsePositives.fetch_add(1, std::memory_order_relaxed);
                        }
                        
                        gCollisionBeingResolved.store(false);
                    }
                }
            }
            gPendingChunks--;
        }
    }
}
void StartWorkerThreads() {
    if (gWorkersRunning.load()) return;
    gWorkersRunning = true;
    
    for (int i = 0; i < NUM_WORKER_THREADS; i++) {
        gWorkerThreads.emplace_back(WorkerThread, i);
    }
    printf("Started %d worker threads for LOCK-FREE processing\r\n", NUM_WORKER_THREADS);
    
    // v56C: Start async BSGS resolver thread
    StartBSGSThread();
}

void StopWorkerThreads() {
    gWorkersRunning = false;
    gWorkCV.notify_all();
    for (auto& t : gWorkerThreads) {
        if (t.joinable()) t.join();
    }
    gWorkerThreads.clear();
    
    // v56C: Stop async BSGS resolver
    StopBSGSThread();
}

// Global TameStore already declared in TameStore.h as extern

// Signal handling
volatile bool gSaveAndExit = false;

// Hunt Mode state
volatile bool gTrapPhase = true;       // true = generating TAMES, false = hunting with WILDS
volatile bool gHuntPhase = false;      // true = hunt mode active
// gHuntChecks moved to thread pool section (atomic)
volatile u64 gHuntHits = 0;            // Bloom hits in hunt phase
double gRAMLimitGB = 0.0;
u64 gHuntStartTime = 0;

// Smart Wave System
int gWaveIntervalMin = 0;              // Minutes between wave renewals (0 = disabled)
u64 gLastWaveTime = 0;                 // Timestamp of last wave launch

// Configurable kangaroo count
int gGroupCnt = 24;                    // Groups per block (default 24, can be 8-256)

// v34: Resonant WILD spawning
bool gResonantSpawn = false;           // Use TAME-based resonant spawning (default: off)

// v37: WILD-WILD collision support
bool gWildWildEnabled = true;          // Enable WILD-WILD collision detection (default: on)
bool gAllWildMode = false;            // Default: ALL-TAME (fill RAM with TAMEs, hunt with WILDs)
bool gNoRotation = true;              // v52: Freeze tables when full (no rotation = no FP explosion)

// v38.5: CHECKPOINT system
char gCheckpointFile[1024] = "wild_checkpoint.dat";  // Default checkpoint filename
char gLoadCheckpointFile[1024] = "";                 // File to load on startup (empty = don't load)
int gCheckpointIntervalHours = 4;                    // Auto-save interval in hours (0 = disabled)
time_t gLastCheckpointTime = 0;                      // Timestamp of last checkpoint
volatile bool gCheckpointRequested = false;          // Request checkpoint from signal handler

// v40: Save discarded DPs for "offline cross-check"
char gDiscardedDPsFile[1024] = "";                   // Empty = disabled
char gPreloadWildsFile[1024] = "";                    // V48: Pre-load wilds into table[1] for hybrid mode
bool gPreloadIsCheckpoint = false;                    // V48: true=checkpoint format, false=savedps format

// Checkpoint signal handler for Ctrl+C
void CheckpointSignalHandler(int sig) {
    if (gSolved) {
        _exit(0);  // Already solved, just exit
    }
    
    if (gCheckpointRequested) {
        printf("\n\nForce exit requested. Exiting immediately.\n");
        fflush(stdout);
        _exit(0);  // Force clean exit, no cleanup (threads may be stuck)
    }
    
    printf("\n\n*** Ctrl+C detected! Shutting down... ***\n");
    printf("    (Press Ctrl+C again to force exit immediately)\n\n");
    fflush(stdout);
    gCheckpointRequested = true;
    gSaveAndExit = true;
    gSolved = true;  // v56C: Stop all workers immediately (they check gSolved)
}

// Collision type codes (match TameStore.h)
#define COLLISION_NONE          0
#define COLLISION_BLOOM_HIT     1
#define COLLISION_TAME_WILD     2
#define COLLISION_WILD_WILD     3

// Force all kangaroos to be TAME or WILD
volatile bool gForceTameMode = false;
volatile bool gForceWildMode = false;

double GetRAMUsageGB() {
    FILE* fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0.0;
    
    u64 total = 0, available = 0;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "MemTotal:", 9) == 0) {
            sscanf(line + 9, "%llu", &total);
        } else if (strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line + 13, "%llu", &available);
        }
    }
    fclose(fp);
    return (double)(total - available) / (1024.0 * 1024.0);
}

void SignalHandler(int sig) {
    if (sig == SIGTERM || sig == SIGINT) {
        printf("\n*** Signal %d received - exiting... ***\n", sig);
        gSaveAndExit = true;
        gSolved = true;  // v56C: Stop workers immediately
    }
}

EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
volatile long ThrCnt;
volatile bool gSolved;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;

CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
EcPoint gPntToSolve;
EcInt gPrivKey;

volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
u64 PntTotalOps;
bool IsBench;

u32 gDP;
u32 gRange;
EcInt gStart;
bool gStartSet;
EcPoint gPubKey;
u8 gGPUs_Mask[MAX_GPU_CNT];
char gTamesFileName[1024];
double gMax;
bool gGenMode;
bool gIsOpsLimit;

// ============================================================================
// v56C ASYNC BSGS — IMPLEMENTATION (after globals)
// ============================================================================

void BSGSConsumerThread() {
    while ((gBSGSRunning.load() || !gBSGSQueue.empty()) && !gSaveAndExit && !gSolved) {
        BSGSCandidate candidate;
        bool hasWork = false;
        
        {
            std::unique_lock<std::mutex> lock(gBSGSMutex);
            gBSGSCV.wait_for(lock, std::chrono::milliseconds(100), [&] {
                return !gBSGSQueue.empty() || !gBSGSRunning.load();
            });
            
            if (!gBSGSQueue.empty()) {
                candidate = gBSGSQueue.front();
                gBSGSQueue.pop_front();
                hasWork = true;
            }
        }
        
        if (!hasWork || gSaveAndExit || gSolved) continue;
        
        auto bsgs_start = std::chrono::steady_clock::now();
        
        EcInt bsgs_key;
        if (Collision_BSGS(ec, gPntToSolve, candidate.t, candidate.Type1, candidate.endo_1,
                          candidate.w, candidate.Type2, candidate.endo_2,
                          Int_HalfRange, bsgs_key, false)) {
            
            EcInt maxRange;
            maxRange.Set(1);
            maxRange.ShiftLeft(gRange);
            bool isNegative = (bsgs_key.data[4] >> 63) != 0;
            bool inRange = !isNegative && bsgs_key.IsLessThanU(maxRange);
            
            if (inRange) {
                gPrivKey = bsgs_key;
                gSolved = true;
                
                auto bsgs_end = std::chrono::steady_clock::now();
                auto bsgs_ms = std::chrono::duration_cast<std::chrono::milliseconds>(bsgs_end - bsgs_start).count();
                
                printf("\r\n[BSGS] Key resolved via BSGS in %lld ms!\r\n", (long long)bsgs_ms);
                printf("\r\n");
                printf("================================================================================\r\n");
                const char* cstr = (candidate.Type1 == 0 || candidate.Type2 == 0) ? "TAME-WILD" : "WILD-WILD";
                printf("*** %s COLLISION RESOLVED VIA BSGS - KEY FOUND! ***\r\n", cstr);
                printf("Dist1: ");
                for(int j=21; j>=0; j--) printf("%02X", candidate.dist1_raw[j]);
                printf(" (type=%d, endo=%d)\r\n", candidate.Type1, candidate.endo_1);
                printf("Dist2: ");
                for(int j=21; j>=0; j--) printf("%02X", candidate.dist2_raw[j]);
                printf(" (type=%d, endo=%d)\r\n", candidate.Type2, candidate.endo_2);
                printf("================================================================================\r\n");
                fflush(stdout);
            }
        }
        gBSGSProcessed.fetch_add(1, std::memory_order_relaxed);
    }
}

void PushBSGSCandidate(const EcInt& t, const EcInt& w, int Type1, int Type2,
                        int endo_1, int endo_2, const u8* d1, const u8* d2) {
    BSGSCandidate c;
    c.t = t; c.w = w;
    c.Type1 = Type1; c.Type2 = Type2;
    c.endo_1 = endo_1; c.endo_2 = endo_2;
    memcpy(c.dist1_raw, d1, 24);
    memcpy(c.dist2_raw, d2, 24);
    
    {
        std::lock_guard<std::mutex> lock(gBSGSMutex);
        if ((int)gBSGSQueue.size() >= BSGS_QUEUE_MAX) {
            gBSGSQueue.pop_front();
            gBSGSDropped.fetch_add(1, std::memory_order_relaxed);
        }
        gBSGSQueue.push_back(c);
    }
    gBSGSCV.notify_one();
}

void StartBSGSThread() {
    if (gBSGSRunning.load()) return;
    
    // v56C-OPT: Precompute BSGS baby table (once, ~300ms)
    InitBSGS(ec);
    
    gBSGSRunning = true;
    for (int i = 0; i < BSGS_NUM_THREADS; i++) {
        gBSGSThreads[i] = std::thread(BSGSConsumerThread);
    }
    printf("BSGS async resolver: STARTED (%d threads, queue max=%d, precomputed baby table)\r\n", 
           BSGS_NUM_THREADS, BSGS_QUEUE_MAX);
}

void StopBSGSThread() {
    gBSGSRunning = false;
    gBSGSCV.notify_all();
    for (int i = 0; i < BSGS_NUM_THREADS; i++) {
        if (gBSGSThreads[i].joinable()) gBSGSThreads[i].join();
    }
    u64 processed = gBSGSProcessed.load();
    u64 dropped = gBSGSDropped.load();
    if (processed > 0 || dropped > 0) {
        printf("BSGS async resolver: processed %llu, dropped %llu (%d threads)\r\n",
               (unsigned long long)processed, (unsigned long long)dropped, BSGS_NUM_THREADS);
    }
}

// ============================================================================

#pragma pack(push, 1)
struct DBRec
{
    u8 x[12];
    u8 d[22];
    u8 type;
};
#pragma pack(pop)

void InitGpus()
{
    GpuCnt = 0;
    int gcnt = 0;
    cudaGetDeviceCount(&gcnt);
    if (gcnt > MAX_GPU_CNT)
        gcnt = MAX_GPU_CNT;

    if (!gcnt)
        return;

    int drv, rt;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    char drvver[100];
    sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

    printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);
    cudaError_t cudaStatus;
    for (int i = 0; i < gcnt; i++)
    {
        cudaStatus = cudaSetDevice(i);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaSetDevice for gpu %d failed!\r\n", i);
            continue;
        }

        if (!gGPUs_Mask[i])
            continue;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, PCI %d, L2 size: %d KB\r\n", 
               i, deviceProp.name, ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f, 
               deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor, 
               deviceProp.pciBusID, deviceProp.l2CacheSize / 1024);
        
        if (deviceProp.major < 6)
        {
            printf("GPU %d - not supported, skip\r\n", i);
            continue;
        }

        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

        GpuKangs[GpuCnt] = new RCGpuKang();
        GpuKangs[GpuCnt]->CudaIndex = i;
        GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
        GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
        GpuKangs[GpuCnt]->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
        GpuKangs[GpuCnt]->GroupCnt = gGroupCnt;  // Use configurable group count
        GpuCnt++;
    }
    printf("GroupCnt: %d, Kangaroos per GPU: %d (%.1fx default)\r\n", 
           gGroupCnt, 256 * gGroupCnt * 48, (double)gGroupCnt / 24.0);
    printf("Total GPUs for work: %d\r\n", GpuCnt);
}

#ifdef _WIN32
u32 __stdcall kang_thr_proc(void* data)
{
    RCGpuKang* Kang = (RCGpuKang*)data;
    Kang->Execute();
    InterlockedDecrement(&ThrCnt);
    return 0;
}
#else
void* kang_thr_proc(void* data)
{
    RCGpuKang* Kang = (RCGpuKang*)data;
    Kang->Execute();
    __sync_fetch_and_sub(&ThrCnt, 1);
    return 0;
}
#endif

// Track overflow for rate-limiting messages
static u64 last_overflow_msg = 0;
static u64 overflow_count = 0;

// Forward declaration
void CheckNewPoints();

// Processing thread for HUNT phase
volatile bool gProcessingRunning = false;

#ifdef _WIN32
unsigned int __stdcall processing_thread(void* param)
#else
void* processing_thread(void* param)
#endif
{
    while (gProcessingRunning && !gSolved) {
        CheckNewPoints();
        // Minimal sleep to prevent CPU spin
        if (!gSolved) {
#ifdef _WIN32
            Sleep(0);  // Yield CPU
#else
            sched_yield();
#endif
        }
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt)
{
    csAddPoints.Enter();
    if (PntIndex + pnt_cnt >= MAX_CNT_LIST)
    {
        csAddPoints.Leave();
        overflow_count++;
        // Rate limit overflow messages to once per second
        u64 now = GetTickCount64();
        if (now - last_overflow_msg > 1000) {
            printf("DPs buffer overflow (%llu times), some points lost\r\n", overflow_count);
            last_overflow_msg = now;
        }
        return;
    }
    memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
    PntIndex += pnt_cnt;
    PntTotalOps += ops_cnt;
    csAddPoints.Leave();
}

// Basic collision resolution - handles TameType properly
bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg)
{
    // v38 FIX: Use local variable to avoid race condition!
    EcInt localKey;
    
    if (IsNeg)
        t.Neg();
    
    if (TameType == TAME)
    {
        // TAME-WILD collision
        // TAME: starts from G, distance t → point = t*G
        // WILD1: starts from PntA = (k - HalfRange)*G, distance w → point = (k - HalfRange + w)*G
        // WILD2: starts from PntB = (HalfRange - k)*G, distance w → point = (HalfRange - k + w)*G
        //
        // If WILD1 collision: t = k - HalfRange + w → k = t - w + HalfRange
        // If WILD2 collision: t = HalfRange - k + w → k = HalfRange + w - t
        // Both formulas are tried below (+ negation variants)
        
        localKey = t;
        localKey.Sub(w);
        EcInt sv = localKey;
        
        // Try WILD1 formula: k = t - w + HalfRange
        localKey.Add(Int_HalfRange);
        EcPoint P = ec.MultiplyG(localKey);
        if (P.IsEqual(pnt)) {
            gPrivKey = localKey;
            return true;
        }
        
        // Try WILD2 formula: k = t - w - HalfRange
        localKey = sv;
        localKey.Sub(Int_HalfRange);
        P = ec.MultiplyG(localKey);
        if (P.IsEqual(pnt)) {
            gPrivKey = localKey;
            return true;
        }
        
        // Try negations
        localKey = sv;
        localKey.Neg();
        EcInt sv2 = localKey;
        
        localKey.Add(Int_HalfRange);
        P = ec.MultiplyG(localKey);
        if (P.IsEqual(pnt)) {
            gPrivKey = localKey;
            return true;
        }
        
        localKey = sv2;
        localKey.Sub(Int_HalfRange);
        P = ec.MultiplyG(localKey);
        if (P.IsEqual(pnt)) {
            gPrivKey = localKey;
            return true;
        }
        return false;
    }
    else
    {
        // WILD-WILD collision (different types: WILD1-WILD2)
        // WILD1 starts at: (k - HalfRange)*G, walks w1 → point = (k - HalfRange + w1)*G
        // WILD2 starts at: (HalfRange - k)*G, walks w2 → point = (HalfRange - k + w2)*G
        // At collision: k - HalfRange + w1 = HalfRange - k + w2
        // 2k = 2*HalfRange + w2 - w1
        // k = HalfRange + (w2 - w1)/2
        //
        // v41.1 FIX: Must include gStart! k_absolute = gStart + k_relative
        // k = gStart + HalfRange + (w2 - w1)/2
        // OR equivalently: k = gStart + HalfRange ± |t - w|/2
        
        localKey = t;
        localKey.Sub(w);
        if (localKey.data[4] >> 63)
            localKey.Neg();
        localKey.ShiftRight(1);
        EcInt sv = localKey;
        
        // v41.1 FIX: k = gStart + HalfRange + |t-w|/2
        localKey = gStart;
        localKey.Add(sv);
        localKey.Add(Int_HalfRange);
        EcPoint P = ec.MultiplyG(localKey);
        if (P.IsEqual(pnt)) {
            gPrivKey = localKey;
            return true;
        }
        
        // Try opposite sign: k = gStart + HalfRange - |t-w|/2
        localKey = gStart;
        localKey.Sub(sv);
        localKey.Add(Int_HalfRange);
        P = ec.MultiplyG(localKey);
        if (P.IsEqual(pnt)) {
            gPrivKey = localKey;
            return true;
        }
        
        // Also try without gStart as fallback (in case gStart==0)
        localKey = sv;
        localKey.Add(Int_HalfRange);
        P = ec.MultiplyG(localKey);
        if (P.IsEqual(pnt)) {
            gPrivKey = localKey;
            return true;
        }
        localKey = sv;
        localKey.Neg();
        localKey.Add(Int_HalfRange);
        P = ec.MultiplyG(localKey);
        if (P.IsEqual(pnt)) {
            gPrivKey = localKey;
            return true;
        }
        return false;
    }
}

// v55 FIX: Normalize signed distance to [0, n) before lambda multiplication.
// MulLambdaModN treats data[0..3] as unsigned 256-bit, but sign-extended
// negative distances have data[0..3] ≈ 2^256 + d (NOT n + d).
// This causes an error of 2^256 * lambda mod n in the result.
static void NormDistForLambda(EcInt& v) {
    // Check if negative: data[4] has bits set, or data[3] has high bit set
    if (v.data[4] != 0 || (v.data[3] >> 63)) {
        v.Neg();       // v = |v_signed| (positive, small ~176 bits)
        // Compute n - |v_signed| = v_signed mod n
        EcInt tmp;
        tmp.Assign(g_N);
        tmp.Sub(v);
        v = tmp;
        v.data[4] = 0; // Clear 5th limb
    }
}

// Apply lambda^power to distance, with proper sign normalization
void ApplyLambda(EcInt& v, int power) {
    if (power == 0) return;
    NormDistForLambda(v);
    if (power == 1) v.MulLambdaModN();
    else if (power == 2) v.MulLambda2ModN();
}

// Collision with endomorphism - try all lambda combinations
bool Collision_SOTA_Endo(EcPoint& pnt, EcInt t_orig, int TameType, int endo_t, 
                         EcInt w_orig, int WildType, int endo_w)
{
    // First try without any adjustment
    if (Collision_SOTA(pnt, t_orig, TameType, w_orig, WildType, false))
        return true;
    if (Collision_SOTA(pnt, t_orig, TameType, w_orig, WildType, true))
        return true;
    
    // If endo transforms are different, try adjusting distances by lambda
    if (endo_t != endo_w)
    {
        // v55: Try both possible lambda exponents on t
        for (int exp = 1; exp <= 2; exp++)
        {
            EcInt t_adj = t_orig;
            ApplyLambda(t_adj, exp);
            
            if (Collision_SOTA(pnt, t_adj, TameType, w_orig, WildType, false))
                return true;
            if (Collision_SOTA(pnt, t_adj, TameType, w_orig, WildType, true))
                return true;
        }
        
        // Try adjusting w instead
        for (int exp = 1; exp <= 2; exp++)
        {
            EcInt w_adj = w_orig;
            ApplyLambda(w_adj, exp);
            
            if (Collision_SOTA(pnt, t_orig, TameType, w_adj, WildType, false))
                return true;
            if (Collision_SOTA(pnt, t_orig, TameType, w_adj, WildType, true))
                return true;
        }
    }
    
    // Try all other combinations (brute force for robustness)
    // v52.1: SKIP when both endos are 0 — lambda adjustments cannot help
    if (endo_t == 0 && endo_w == 0)
        return false;
    
    for (int try_t = 0; try_t < 3; try_t++)
    {
        EcInt t_try = t_orig;
        ApplyLambda(t_try, try_t);
        
        for (int try_w = 0; try_w < 3; try_w++)
        {
            if (try_t == 0 && try_w == 0) continue; // Already tried
            
            EcInt w_try = w_orig;
            ApplyLambda(w_try, try_w);
            
            if (Collision_SOTA(pnt, t_try, TameType, w_try, WildType, false))
                return true;
            if (Collision_SOTA(pnt, t_try, TameType, w_try, WildType, true))
                return true;
        }
    }
    
    return false;
}

// GET_KANG_TYPE and GET_ENDO_TRANSFORM defined above in thread pool section
// Statistics moved to thread pool section (atomic versions)

void CheckNewPoints()
{
    csAddPoints.Enter();
    if (!PntIndex)
    {
        csAddPoints.Leave();
        return;
    }

    int cnt = PntIndex;
    
    // v51.1 FIX: Use pre-allocated pPntList2 instead of malloc()
    // Old code did malloc(1.5GB) per call — fails when RAM is tight (120GB tables + 11.5GB GPU)
    // causing permanent buffer overflow deadlock.
    // pPntList2 is pre-allocated at startup and safe to reuse here because
    // we wait for all workers to finish (gPendingChunks==0) before returning.
    u8* localBuffer = pPntList2;
    memcpy(localBuffer, pPntList, GPU_DP_SIZE * cnt);
    PntIndex = 0;
    csAddPoints.Leave();

    // Use thread pool for both TRAP and HUNT
    // v52: With GroupCnt=24, typical batch = ~2K DPs. Workers add overhead for small batches.
    // Only use workers when batch > 5000 DPs (reduces mutex/condvar overhead)
    if (gWorkersRunning.load() && cnt > 5000) {
        int num_chunks = NUM_WORKER_THREADS;
        int chunk_size = cnt / num_chunks;
        if (chunk_size < 50) {
            num_chunks = std::max(1, cnt / 50);
            chunk_size = cnt / num_chunks;
        }
        
        {
            std::lock_guard<std::mutex> lock(gWorkMutex);
            for (int c = 0; c < num_chunks; c++) {
                WorkChunk chunk;
                chunk.data = localBuffer;  // Use local buffer
                chunk.start = c * chunk_size;
                chunk.end = (c == num_chunks - 1) ? cnt : (c + 1) * chunk_size;
                gWorkQueue.push_back(chunk);
                gPendingChunks++;
            }
        }
        gWorkCV.notify_all();
        
        // Wait for completion (MUST complete before reusing pPntList2!)
        while (gPendingChunks.load() > 0 && !gSolved) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        // v51.1: No free() — localBuffer is pre-allocated pPntList2
    }
    else {
        // Fallback single-threaded (uses local buffer)
        for (int i = 0; i < cnt && !gSolved; i++)
        {
            u8* p = localBuffer + i * GPU_DP_SIZE;
            u8 x_bytes[16];
            u8 dist_bytes[24];
            
            memcpy(x_bytes, p, 12);
            memset(x_bytes + 12, 0, 4);
            memcpy(dist_bytes, p + 16, 22);
            
            u8 type_info = p[40];  
            u8 kang_type = GET_KANG_TYPE(type_info);
            u8 endo_transform = GET_ENDO_TRANSFORM(type_info);
            
            if (gTrapPhase) {
                // v51: Store real TAMEs in table[0]
                dist_bytes[22] = (endo_transform << 4) | TAME;
                dist_bytes[23] = 0;
                
                gTameStore.AddTame(x_bytes, dist_bytes);
                gTamesDPs++;
            }
            else if (kang_type == WILD1 || kang_type == WILD2) {
                gHuntChecks++;
                dist_bytes[22] = type_info;
                dist_bytes[23] = 0;
                
                // v52: Use correct function for each mode
                int result;
                if (gAllWildMode) {
                    result = gTameStore.CheckWildOnly(x_bytes, dist_bytes);
                } else {
                    result = gTameStore.CheckWild(x_bytes, dist_bytes);
                }
                
                if (result == COLLISION_TAME_WILD || result == COLLISION_WILD_WILD) {
                    gWildCollisions++;
                    
                    const char* collision_str; // set after types are extracted
                    
                    u8 dist1_data[24], dist2_data[24];
                    int coll_type;
                    gTameStore.GetCollisionData(dist1_data, dist2_data, &coll_type);
                    
                    EcInt t, w;
                    memset(t.data, 0, sizeof(t.data));
                    memset(w.data, 0, sizeof(w.data));
                    memcpy(t.data, dist1_data, 22);
                    memcpy(w.data, dist2_data, 22);
                    
                    if (dist1_data[21] & 0x80) memset(((u8*)t.data) + 22, 0xFF, 18);
                    if (dist2_data[21] & 0x80) memset(((u8*)w.data) + 22, 0xFF, 18);
                    
                    u8 type1_info = dist1_data[22];
                    u8 type2_info = dist2_data[22];
                    int Type1 = GET_KANG_TYPE(type1_info);
                    int Type2 = GET_KANG_TYPE(type2_info);
                    int endo_1 = GET_ENDO_TRANSFORM(type1_info);
                    int endo_2 = GET_ENDO_TRANSFORM(type2_info);
                    collision_str = (Type1 == TAME || Type2 == TAME) ? "TAME-WILD" : "WILD-WILD";
                    
                    if (Collision_SOTA_Endo(gPntToSolve, t, Type1, endo_1, w, Type2, endo_2)) {
                        gSolved = true;
                        printf("\r\n");
                        printf("================================================================================\r\n");
                        printf("*** %s COLLISION RESOLVED - KEY FOUND! ***\r\n", collision_str);
                        printf("Dist1: "); 
                        for(int j=21; j>=0; j--) printf("%02X", dist1_data[j]);
                        printf(" (type=%d, endo=%d)\r\n", Type1, endo_1);
                        printf("Dist2: ");
                        for(int j=21; j>=0; j--) printf("%02X", dist2_data[j]);
                        printf(" (type=%d, endo=%d)\r\n", Type2, endo_2);
                        printf("================================================================================\r\n");
                        fflush(stdout);
                        return;
                    }
                    // v56C: BSGS fallback (async)
                    {
                        EcInt t2, w2;
                        memset(t2.data, 0, sizeof(t2.data));
                        memset(w2.data, 0, sizeof(w2.data));
                        memcpy(t2.data, dist1_data, 22);
                        memcpy(w2.data, dist2_data, 22);
                        if (dist1_data[21] & 0x80) memset(((u8*)t2.data) + 22, 0xFF, 18);
                        if (dist2_data[21] & 0x80) memset(((u8*)w2.data) + 22, 0xFF, 18);
                        PushBSGSCandidate(t2, w2, Type1, Type2, endo_1, endo_2,
                                        dist1_data, dist2_data);
                    }
                    gTameStore.ClearCollision();
                }
            }
        }
        // v51.1: No free() — localBuffer is pre-allocated pPntList2
    }
}

void ShowStats(u64 tm_start, double exp_ops, double dp_val)
{
    int speed = GpuKangs[0]->GetStatsSpeed();
    for (int i = 1; i < GpuCnt; i++)
        speed += GpuKangs[i]->GetStatsSpeed();

    u64 sec = (GetTickCount64() - tm_start) / 1000;
    u64 days = sec / (3600 * 24);
    int hours = (int)(sec - days * (3600 * 24)) / 3600;
    int min = (int)(sec - days * (3600 * 24) - hours * 3600) / 60;
    
    // ALL-WILD mode: Always show HUNT stats (no TRAP phase)
    if (gAllWildMode) {
        u64 hunt_sec = sec > 0 ? sec : 1;
        double checks_per_sec = (double)gHuntChecks.load() / hunt_sec;
        u64 wave = GpuKangs[0]->GetWaveNumber();
        u64 ww_coll = gTameStore.GetWildWildCollisions();
        u64 wild_stored = gTameStore.GetWildCount();
        u64 duplicates = gTameStore.GetDuplicatePoints();
        double wild_load = gTameStore.GetWildLoadFactor();
        u64 fp_count = gFalsePositives.load();  // v40.1
        
        // Calculate expected DPs/s for display
        double expected_dps = (double)speed * 1000000.0 / pow(2.0, gDP);
        
        printf("  Speed: %.2f GKeys/s | Time: %llud %02dh %02dm | Wave #%llu\r\n", 
               speed / 1000.0, days, hours, min, wave);
        printf("  Checks: %lluM | W-W: %llu | FP: %llu | Dup: %llu | %.1fK checks/s\r\n",
               gHuntChecks.load() / 1000000, ww_coll, fp_count, duplicates, checks_per_sec / 1000.0);
        
        // Smart display: show absolute value when small, M when large
        if (wild_stored < 1000000) {
            printf("  WILDs stored: %llu / %.0fM (%.4f%% load) [~%.2f DPs/s]\r\n",
                   wild_stored, gTameStore.GetWildTableSize() / 1000000.0, wild_load,
                   expected_dps);
        } else {
            const char* freeze_tag = gTameStore.IsAnyFrozen() ? " FROZEN" : "";
            printf("  W1: %.0fM  W2: %.0fM  Total: %.0fM / %.0fM (%.2f%% load)%s\r\n",
                   gTameStore.GetWild1Count() / 1000000.0,
                   gTameStore.GetWild2Count() / 1000000.0,
                   wild_stored / 1000000.0, gTameStore.GetWildTableSize() / 1000000.0, wild_load,
                   freeze_tag);
        }
        
        // v45: Show savedps disk usage
        u64 disk_bytes = gTameStore.GetDiscardedBytes();
        if (disk_bytes > 0) {
            u64 disk_dps = gTameStore.GetDiscardedSaved();
            printf("  Disk: %.2f GB (%lluM DPs saved for offline cross-check)\r\n",
                   (double)disk_bytes / (1024.0*1024*1024),
                   (unsigned long long)(disk_dps / 1000000));
        }
        return;
    }
    
    // Normal mode: TRAP/HUNT phases
    const char* phase = gTrapPhase ? "TRAP" : "HUNT";
    u64 w1_count = gTameStore.GetWild1Count();
    u64 w2_count = gTameStore.GetWild2Count();
    u64 total_stored = w1_count + w2_count;
    u64 per_table_size = gTameStore.GetPerTableSize();
    u64 total_capacity = per_table_size * 2;
    
    double total_load = (total_capacity > 0) ? (double)total_stored * 100.0 / (double)total_capacity : 0;
    
    printf("%s: Speed: %.2f GKeys/s | DPs: %lluM | Time: %llud %02dh %02dm\r\n", 
           phase, speed / 1000.0, total_stored / 1000000,
           days, hours, min);
    
    if (gTrapPhase) {
        // v53: Show TAME count in table[0]
        printf("  TAMEs: %lluM / %lluM (%.1f%%) - target 93%%\r\n",
               w1_count / 1000000, per_table_size / 1000000,
               (per_table_size > 0) ? (double)w1_count * 100.0 / (double)per_table_size : 0);
        printf("  DPs stored: %lluM (all TAMEs → table[0])\r\n",
               gTamesDPs.load() / 1000000);
    } else {
        u64 hunt_sec = (GetTickCount64() - gHuntStartTime) / 1000;
        double checks_per_sec = (hunt_sec > 0) ? (double)gHuntChecks.load() / hunt_sec : 0;
        double fp_rate = gTameStore.GetFalsePositiveRate();
        u64 wave = GpuKangs[0]->GetWaveNumber();
        u64 tw_coll = gTameStore.GetTameWildCollisions();
        u64 ww_coll = gTameStore.GetWildWildCollisions();
        u64 wild_stored = gTameStore.GetWildCount();
        u64 duplicates = gTameStore.GetDuplicatePoints();
        
        printf("  HUNT Wave #%llu: Checks: %lluM, T-W: %llu, W-W: %llu, FP: %llu, Dup: %llu, %.1fK/s\r\n",
               wave, gHuntChecks.load() / 1000000, tw_coll, ww_coll, 
               gFalsePositives.load(), duplicates, checks_per_sec / 1000.0);
        
        if (gTameStore.IsAllTameMode()) {
            // v53: Show TAME table stats
            u64 tame_count = gTameStore.GetWild1Count();
            u64 tame_size = gTameStore.GetPerTableSize();
            printf("  TAMEs: %lluM (%.1f%% of %.1f GB) - check-only mode, no WILD storage\r\n",
                   tame_count / 1000000,
                   (tame_size > 0) ? (double)tame_count * 100.0 / (double)tame_size : 0,
                   (double)(tame_size * sizeof(WildEntryCompact)) / (1024.0*1024*1024));
            // v56C-OPT: Show BSGS queue stats with thread count
            u64 bsgs_done = gBSGSProcessed.load();
            u64 bsgs_drop = gBSGSDropped.load();
            if (bsgs_done > 0 || bsgs_drop > 0) {
                std::lock_guard<std::mutex> lk(gBSGSMutex);
                printf("  BSGS [%d thr]: %zu pending, %llu processed, %llu dropped\r\n",
                       BSGS_NUM_THREADS, gBSGSQueue.size(), 
                       (unsigned long long)bsgs_done, (unsigned long long)bsgs_drop);
            }
        } else if (gWildWildEnabled && wild_stored > 0) {
            double wild_load = gTameStore.GetWildLoadFactor();
            printf("  WILDs stored: %lluM (%.1f%% load) - for WILD-WILD detection\r\n",
                   wild_stored / 1000000, wild_load);
        }
    }
}

bool SolvePoint(EcPoint PntToSolve, int Range, int DP, EcInt* pk_res)
{
    if ((Range < 32) || (Range > 180))
    {
        printf("Unsupported Range value (%d)!\r\n", Range);
        return false;
    }
    if ((DP < 6) || (DP > 60)) 
    {
        printf("Unsupported DP value (%d)!\r\n", DP);
        return false;
    }

    // Initialize TameStore
    double store_ram = gRAMLimitGB - 3.0;  // Leave 3GB margin for GPU + system
    if (store_ram < 4.0) store_ram = 4.0;  // Minimum 4GB for TameStore
    
    bool init_ok;
    if (gAllWildMode) {
        // WILD-ONLY mode: All RAM goes to WILD table for maximum W-W collisions
        init_ok = gTameStore.InitWildOnly((size_t)store_ram);
    } else {
        // v53: ALL-TAME mode: All RAM goes to TAME table for maximum T-W probability
        init_ok = gTameStore.InitAllTame((size_t)store_ram);
    }
    
    if (!init_ok) {
        printf("Failed to initialize TameStore!\r\n");
        return false;
    }
    
    // v52: Set rotation policy
    gTameStore.SetNoRotation(gNoRotation);
    if (gNoRotation) {
        printf("Table freeze: ENABLED (tables become read-only when full, no FP explosion)\n");
    }
    
    // v38.5: Load checkpoint if specified
    if (gLoadCheckpointFile[0] != 0) {
        if (gTameStore.LoadCheckpoint(gLoadCheckpointFile)) {
            printf("Checkpoint loaded! Continuing from previous state.\r\n\r\n");
        } else {
            printf("WARNING: Failed to load checkpoint. Starting fresh.\r\n\r\n");
        }
    }
    
    // v38.5: Initialize checkpoint timer
    gLastCheckpointTime = time(NULL);
    if (gCheckpointIntervalHours > 0) {
        printf("Auto-checkpoint enabled: Every %d hours to '%s'\r\n", gCheckpointIntervalHours, gCheckpointFile);
        printf("Press Ctrl+C to save checkpoint and exit safely.\r\n\r\n");
    }
    
    // v40: Open discarded DPs file if specified
    if (gDiscardedDPsFile[0] != 0) {
        if (gTameStore.OpenDiscardedFile(gDiscardedDPsFile)) {
            printf("Discarded DPs will be saved to: %s\r\n\r\n", gDiscardedDPsFile);
        }
    }

    printf("\r\nSolving point: Range %d bits, DP %d\r\n", Range, DP);
    double ops = 1.15 * pow(2.0, Range / 2.0);
    double dp_val = (double)(1ull << DP);
    
    if (gAllWildMode) {
        printf("\r\nStrategy: ALL-WILD (dual table, W-W collisions)\r\n");
        printf("  Table 1 (WILD1): %llu entries\r\n", 
               (unsigned long long)gTameStore.GetPerTableSize());
        printf("  Table 2 (WILD2): %llu entries\r\n", 
               (unsigned long long)gTameStore.GetPerTableSize());
        printf("  Total capacity:  %llu entries\r\n", 
               (unsigned long long)gTameStore.GetWildTableSize());
        printf("  Skipping TRAP phase — direct WILD-WILD hunt.\r\n");
    } else {
        if (gPreloadWildsFile[0] != 0) {
            printf("\r\nStrategy: HYBRID TRAP/HUNT\r\n");
            printf("  Phase 1 (TRAP):    Fill %.0f GB with TAMEs\r\n", gRAMLimitGB);
            printf("  Phase 1.5:         Preload WILDs from: %s\r\n", gPreloadWildsFile);
            printf("  Phase 2 (HUNT):    T-W + W-W collision detection\r\n");
        } else {
            printf("\r\nStrategy: ALL-TAME TRAP/HUNT\r\n");
            printf("  Phase 1 (TRAP):  Fill ALL %.0f GB with TAMEs (target 93%%)\r\n", gRAMLimitGB);
            printf("  Phase 2 (HUNT):  100%% WILDs check against TAMEs (no WILD storage)\r\n");
            printf("  Advantage:       2x more TAMEs = 2x T-W collision probability\r\n");
        }
    }
    printf("\r\n");
    
    gIsOpsLimit = false;
    gTrapPhase = !gAllWildMode;  // Skip TRAP if ALL-WILD mode
    gHuntPhase = gAllWildMode;   // Start in HUNT if ALL-WILD mode

    u64 total_kangs = GpuKangs[0]->CalcKangCnt();
    for (int i = 1; i < GpuCnt; i++)
        total_kangs += GpuKangs[i]->CalcKangCnt();

    SetRndSeed(0);
    PntTotalOps = 0;
    PntIndex = 0;

    // Prepare jumps
    EcInt minjump, t;

#if V46_ENABLE
    // =========================================================================
    // V46 STRATIFIED JUMP TABLE
    // =========================================================================
    // Split jmp1 into standard + explorer entries.
    // Standard: scale = 2^(Range/2 + 3)     [covers local area]
    // Explorer: scale = 2^(Range/2 + 3 + SHIFT) [covers wider area]
    //
    // The jump selection (x[0] % JMP_CNT) is hash-based and deterministic,
    // so collision detection is preserved. Two kangaroos at the same point
    // will always pick the same jump entry → same path forward.
    // =========================================================================
    
    int explorer_count = (JMP_CNT * V46_EXPLORER_PCT) / 100;
    int standard_count = JMP_CNT - explorer_count;
    int base_shift = Range / 2 + 3;
    int explorer_shift = base_shift + V46_EXPLORER_SHIFT;
    
    printf("\n=== Stratified Jump Table ===\n");
    printf("  Standard entries: %d (scale 2^%d)\n", standard_count, base_shift);
    printf("  Explorer entries: %d (scale 2^%d = %dx bigger)\n", 
           explorer_count, explorer_shift, 1 << V46_EXPLORER_SHIFT);
    printf("  Explorer fraction: %d%%\n", V46_EXPLORER_PCT);
    
    // Standard entries (first standard_count entries)
    minjump.Set(1);
    minjump.ShiftLeft(base_shift);
    for (int i = 0; i < standard_count; i++)
    {
        EcJumps1[i].dist = minjump;
        t.RndMax(minjump);
        EcJumps1[i].dist.Add(t);
        EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        EcJumps1[i].p = ec.MultiplyG(EcJumps1[i].dist);
    }
    
    // Explorer entries (last explorer_count entries)
    EcInt minjump_explorer;
    minjump_explorer.Set(1);
    minjump_explorer.ShiftLeft(explorer_shift);
    for (int i = standard_count; i < JMP_CNT; i++)
    {
        EcJumps1[i].dist = minjump_explorer;
        t.RndMax(minjump_explorer);
        EcJumps1[i].dist.Add(t);
        EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        EcJumps1[i].p = ec.MultiplyG(EcJumps1[i].dist);
    }
    
    // Calculate theoretical mean step size
    // Mean ≈ (standard_count * 1.5 * 2^base + explorer_count * 1.5 * 2^explorer) / JMP_CNT
    double mean_factor = ((double)standard_count * 1.0 + (double)explorer_count * (1 << V46_EXPLORER_SHIFT)) / JMP_CNT;
    printf("  Mean step increase vs uniform: %.2fx\n", mean_factor);
    printf("================================\n\n");
    
#else
    // Original v45 uniform jump table
    minjump.Set(1);
    minjump.ShiftLeft(Range / 2 + 3);
    for (int i = 0; i < JMP_CNT; i++)
    {
        EcJumps1[i].dist = minjump;
        t.RndMax(minjump);
        EcJumps1[i].dist.Add(t);
        EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        EcJumps1[i].p = ec.MultiplyG(EcJumps1[i].dist);
    }
#endif

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10);
    for (int i = 0; i < JMP_CNT; i++)
    {
        EcJumps2[i].dist = minjump;
        t.RndMax(minjump);
        EcJumps2[i].dist.Add(t);
        EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        EcJumps2[i].p = ec.MultiplyG(EcJumps2[i].dist);
    }

    minjump.Set(1);
    minjump.ShiftLeft(Range - 10 - 2);
    for (int i = 0; i < JMP_CNT; i++)
    {
        EcJumps3[i].dist = minjump;
        t.RndMax(minjump);
        EcJumps3[i].dist.Add(t);
        EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE;
        EcJumps3[i].p = ec.MultiplyG(EcJumps3[i].dist);
    }
    SetRndSeed(GetTickCount64());

    Int_HalfRange.Set(1);
    Int_HalfRange.ShiftLeft(Range - 1);
    Pnt_HalfRange = ec.MultiplyG(Int_HalfRange);
    Pnt_NegHalfRange = Pnt_HalfRange;
    Pnt_NegHalfRange.y.NegModP();
    Int_TameOffset.Set(1);
    Int_TameOffset.ShiftLeft(Range - 1);
    EcInt tt;
    tt.Set(1);
    tt.ShiftLeft(Range - 5);
    Int_TameOffset.Sub(tt);
    gPntToSolve = PntToSolve;

    // Enable TAME-only mode for TRAP phase
    gGenMode = true;  // This tells GPU to generate only TAMES

    // Prepare GPUs
    for (int i = 0; i < GpuCnt; i++)
        if (!GpuKangs[i]->Prepare(PntToSolve, Range, DP, EcJumps1, EcJumps2, EcJumps3))
        {
            GpuKangs[i]->Failed = true;
            printf("GPU %d Prepare failed\r\n", GpuKangs[i]->CudaIndex);
        }
        else
        {
            // v34: Configure resonant spawning
            GpuKangs[i]->SetResonantSpawn(gResonantSpawn);
        }

    u64 tm0 = GetTickCount64();
    
    if (gAllWildMode) {
        // ALL-WILD MODE: Skip TRAP entirely, go directly to HUNT
        printf("ALL-WILD MODE: Starting directly with WILDs...\r\n");
        
        // Initialize GPUs in HUNT mode from the start
        gGenMode = false;
        gHuntStartTime = GetTickCount64();  // Set hunt start time for stats
        
        // CRITICAL: Disable resonant spawning in ALL-WILD mode (no TAMEs to sample from!)
        for (int i = 0; i < GpuCnt; i++) {
            GpuKangs[i]->SetResonantSpawn(false);
        }
        
        for (int i = 0; i < GpuCnt; i++) {
            if (!GpuKangs[i]->ReinitForHunt()) {
                printf("ERROR: Failed to initialize GPU %d for ALL-WILD mode!\r\n", i);
            }
        }
        
        printf("GPU Bloom SKIPPED - ALL-WILD mode uses WILD-WILD collisions only!\r\n");
        printf("All WILDs Mode: 100%% kangaroos hunting!\r\n");
        printf("Smart Waves disabled (use -waveinterval N to enable)\r\n");
        printf("GPUs ready for ALL-WILD HUNT!\r\n\r\n");
        
        // Start worker threads
        StartWorkerThreads();
        
        // Start dedicated processing thread
        gProcessingRunning = true;
        gLastWaveTime = GetTickCount64();
#ifdef _WIN32
        HANDLE proc_thread = (HANDLE)_beginthreadex(NULL, 0, processing_thread, NULL, 0, NULL);
#else
        pthread_t proc_thread;
        pthread_create(&proc_thread, NULL, processing_thread, NULL);
#endif
        
    } else {
        printf("GPUs started in TRAP mode (filling table[0] with real TAMEs)...\r\n");
        if (gResonantSpawn) {
            printf("Resonant WILD spawning: ENABLED (using TAME distance statistics)\r\n");
        } else {
            printf("Resonant WILD spawning: DISABLED (random spawning)\r\n");
        }
        
        // Start worker threads for lock-free parallel processing
        StartWorkerThreads();
    }

#ifdef _WIN32
    HANDLE thr_handles[MAX_GPU_CNT];
#else
    pthread_t thr_handles[MAX_GPU_CNT];
#endif

    u32 ThreadID;
    gSolved = false;
    ThrCnt = GpuCnt;
    
    for (int i = 0; i < GpuCnt; i++)
    {
#ifdef _WIN32
        thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, (void*)GpuKangs[i], 0, &ThreadID);
#else
        pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]);
#endif
    }

    u64 tm_stats = GetTickCount64();
    u64 tm_ram_check = GetTickCount64();
    
    while (!gSolved)
    {
        // During TRAP: process in main loop
        // During HUNT: dedicated thread handles processing
        if (gTrapPhase) {
            CheckNewPoints();
        }
        Sleep(1);
        
        if (GetTickCount64() - tm_stats > 10 * 1000)
        {
            ShowStats(tm0, ops, dp_val);
            tm_stats = GetTickCount64();
        }

        // Check for phase transition based on table fill level
        if (gTrapPhase && (GetTickCount64() - tm_ram_check > 1000))
        {
            // v51: Check fill level of table[0] (TAMEs)
            u64 w1_count = gTameStore.GetWild1Count();  // table[0] count (TAMEs)
            u64 per_table = gTameStore.GetPerTableSize();
            
            double w1_load = (per_table > 0) ? (double)w1_count / (double)per_table : 0;
            
            // v53: Fill to 93% OR switch when table freezes (hash probing limit reached)
            bool table_is_frozen = gTameStore.IsTableFrozen(0);
            bool should_switch = (w1_load >= 0.93) || table_is_frozen;
            const char* switch_reason = table_is_frozen ? 
                "Table[0] FROZEN (probing limit reached)" : "Table[0] TAMEs 93% full";
            
            if (should_switch)
            {
                // TRANSITION: TRAP -> HUNT
                
                printf("\r\n");
                printf("================================================================================\r\n");
                printf("*** SWITCHING TO HUNT PHASE ***\r\n");
                printf("Reason: %s\r\n", switch_reason);
                printf("  Table[0] (TAMEs): %lluM / %lluM (%.1f%%)\r\n",
                       w1_count / 1000000, per_table / 1000000, w1_load * 100.0);
                printf("  Total TAMEs: %lluM\r\n", w1_count / 1000000);
                printf("TRAP stats: TAMEs stored: %llu in table[0]\r\n", gTamesDPs.load());
                printf("Now hunting with 100%% WILDs against TAMEs (tame-wild collision)...\r\n");
                printf("================================================================================\r\n");
                printf("\r\n");
                
                gTrapPhase = false;
                gHuntPhase = true;
                gHuntStartTime = GetTickCount64();
                gHuntChecks.store(0);
                gHuntHits = 0;
                gWildsDPs.store(0);  // Reset for HUNT phase counting
                gWildCollisions.store(0);
                
                // Switch to WILD mode - CRITICAL: reinitialize GPUs for HUNT!
                gGenMode = false;
                
                // V48 HYBRID: Pre-load wilds into table[1] before starting HUNT
                if (gPreloadWildsFile[0] != 0) {
                    u64 preloaded = 0;
                    if (gPreloadIsCheckpoint) {
                        preloaded = gTameStore.LoadWildsFromCheckpoint(gPreloadWildsFile);
                    } else {
                        preloaded = gTameStore.LoadWildsFromDescartados(gPreloadWildsFile);
                    }
                    if (preloaded > 0) {
                        printf("HYBRID MODE: Checking new WILDs against:\n");
                        printf("  - Table[0]: %lluM TAMEs (from TRAP)\n", 
                               (unsigned long long)(gTameStore.GetWild1Count() / 1000000));
                        printf("  - Table[1]: %lluM pre-loaded WILDs\n",
                               (unsigned long long)(preloaded / 1000000));
                    }
                }
                
                printf("Reinitializing kangaroos for HUNT mode (100%% WILDs - Wave #1)...\r\n");
                for (int i = 0; i < GpuCnt; i++) {
                    if (!GpuKangs[i]->ReinitForHunt()) {
                        printf("ERROR: Failed to reinitialize GPU %d for HUNT!\r\n", i);
                    }
                }
                
                printf("All WILDs Mode: 100%% kangaroos hunting!\r\n");
                if (gWaveIntervalMin > 0) {
                    printf("Smart Waves enabled: New wave every %d minutes\r\n", gWaveIntervalMin);
                } else {
                    printf("Smart Waves disabled (use -waveinterval N to enable)\r\n");
                }
                printf("GPUs ready for HUNT!\r\n\r\n");
                
                // Start worker threads for multi-threaded processing
                StartWorkerThreads();
                
                // Start dedicated processing thread for faster DP handling
                gProcessingRunning = true;
                gLastWaveTime = GetTickCount64();  // Track wave time
#ifdef _WIN32
                HANDLE proc_thread = (HANDLE)_beginthreadex(NULL, 0, processing_thread, NULL, 0, NULL);
#else
                pthread_t proc_thread;
                pthread_create(&proc_thread, NULL, processing_thread, NULL);
#endif
            }
            tm_ram_check = GetTickCount64();
        }
        
        // SMART WAVE RENEWAL: Launch new wave periodically during HUNT
        if (gHuntPhase && gWaveIntervalMin > 0)
        {
            u64 wave_interval_ms = (u64)gWaveIntervalMin * 60 * 1000;
            if (GetTickCount64() - gLastWaveTime > wave_interval_ms)
            {
                u64 old_wave = GpuKangs[0]->GetWaveNumber();
                printf("\r\n*** Launching new WILD wave (Wave #%llu -> #%llu) ***\r\n",
                       old_wave, old_wave + 1);
                
                // Reinitialize all GPUs with new wave positions
                for (int i = 0; i < GpuCnt; i++) {
                    GpuKangs[i]->ReinitForHunt();
                }
                
                gLastWaveTime = GetTickCount64();
                printf("Wave #%llu deployed - covering new regions!\r\n\r\n",
                       GpuKangs[0]->GetWaveNumber());
            }
        }
        
        // v38.5: Auto-checkpoint (only in HUNT phase, not TRAP)
        if (gCheckpointIntervalHours > 0 && !gTrapPhase) {
            time_t current_time = time(NULL);
            double elapsed_hours = difftime(current_time, gLastCheckpointTime) / 3600.0;
            if (elapsed_hours >= gCheckpointIntervalHours) {
                printf("\n*** AUTO-CHECKPOINT (every %d hours) ***\n", gCheckpointIntervalHours);
                if (gTameStore.SaveCheckpoint(gCheckpointFile)) {
                    gLastCheckpointTime = current_time;
                }
                // v45: Also flush savedps to disk for safety
                if (gDiscardedDPsFile[0] != 0) {
                    gTameStore.FlushDiscardedFile();
                    printf("Savedps flushed (%.2f GB on disk)\n",
                           (double)gTameStore.GetDiscardedBytes() / (1024.0*1024*1024));
                }
            }
        }

        if (gSaveAndExit)
        {
            // v56C: Stop GPU first to prevent buffer overflow flood
            for (int i = 0; i < GpuCnt; i++)
                GpuKangs[i]->Stop();
            Sleep(200);  // Let GPU threads wind down
            
            // v55: Respect -checkpoint 0 (never save, not even on Ctrl+C)
            if (!gTrapPhase) {
                // Export RAM tables to savedps BEFORE saving checkpoint
                if (gDiscardedDPsFile[0] != 0) {
                    printf("\n*** Exporting RAM to savedps for complete offline cross-check coverage ***\n");
                    u64 exported = gTameStore.ExportRAMToSavedDPs();
                    if (exported > 0) {
                        printf(" %llu entries exported. Total savedps: %.2f GB\n",
                               (unsigned long long)exported,
                               (double)gTameStore.GetDiscardedBytes() / (1024.0*1024*1024));
                    }
                    gTameStore.CloseDiscardedFile();
                }
                
                // v55: Only save checkpoint if -checkpoint > 0
                if (gCheckpointIntervalHours > 0) {
                    printf("\nSaving checkpoint before exit...\n");
                    if (gTameStore.SaveCheckpoint(gCheckpointFile)) {
                        printf("Checkpoint saved successfully.\n");
                    } else {
                        printf("WARNING: Checkpoint save failed.\n");
                    }
                } else {
                    printf("\nExiting cleanly (checkpoint disabled).\n");
                }
            } else {
                printf("\nExiting during TRAP phase.\n");
            }
            gIsOpsLimit = true;
            printf("Exit completed.\r\n");
            break;
        }
    }

    // Stop processing thread
    gProcessingRunning = false;
    Sleep(100);  // Give thread time to exit
    
    // Stop worker thread pool
    StopWorkerThreads();
    
    printf("Stopping...\r\n");
    for (int i = 0; i < GpuCnt; i++)
        GpuKangs[i]->Stop();
    while (ThrCnt)
        Sleep(10);
    for (int i = 0; i < GpuCnt; i++)
    {
#ifdef _WIN32
        CloseHandle(thr_handles[i]);
#else
        pthread_join(thr_handles[i], NULL);
#endif
    }

    if (!gSolved) {
        printf("\r\nFinal Stats:\r\n");
        gTameStore.PrintStats();
    }
    
    if (gSolved && !gSaveAndExit) {
        *pk_res = gPrivKey;
        return true;
    }
    
    return false;
}

bool ParseCommandLine(int argc, char* argv[])
{
    int ci = 1;
    while (ci < argc)
    {
        char* argument = argv[ci];
        ci++;
        if (strcmp(argument, "-gpu") == 0)
        {
            if (ci >= argc) return false;
            char* gpus = argv[ci];
            ci++;
            memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
            for (int i = 0; i < (int)strlen(gpus); i++)
            {
                if ((gpus[i] < '0') || (gpus[i] > '9')) return false;
                gGPUs_Mask[gpus[i] - '0'] = 1;
            }
        }
        else if (strcmp(argument, "-dp") == 0)
        {
            int val = atoi(argv[ci]);
            ci++;
            if ((val < 6) || (val > 60)) return false;
            gDP = val;
        }
        else if (strcmp(argument, "-range") == 0)
        {
            int val = atoi(argv[ci]);
            ci++;
            if ((val < 32) || (val > 170)) return false;
            gRange = val;
        }
        else if (strcmp(argument, "-start") == 0)
        {    
            if (!gStart.SetHexStr(argv[ci])) return false;
            ci++;
            gStartSet = true;
        }
        else if (strcmp(argument, "-pubkey") == 0)
        {
            if (!gPubKey.SetHexStr(argv[ci])) return false;
            ci++;
        }
        else if (strcmp(argument, "-ramlimit") == 0)
        {
            double val = atof(argv[ci]);
            ci++;
            if (val < 1.0) return false;
            gRAMLimitGB = val;
        }
        else if (strcmp(argument, "-waveinterval") == 0)
        {
            int val = atoi(argv[ci]);
            ci++;
            if (val < 0 || val > 1440) return false;  // 0 to 24 hours
            gWaveIntervalMin = val;
        }
        else if (strcmp(argument, "-groups") == 0)
        {
            int val = atoi(argv[ci]);
            ci++;
            if (val < 8 || val > 256) {
                printf("Error: -groups must be 8-256\r\n");
                return false;
            }
            gGroupCnt = val;
        }
        else if (strcmp(argument, "-resonant") == 0)
        {
            int val = atoi(argv[ci]);
            ci++;
            gResonantSpawn = (val != 0);
        }
        else if (strcmp(argument, "-wildwild") == 0)
        {
            int val = atoi(argv[ci]);
            ci++;
            gWildWildEnabled = (val != 0);
        }
        else if (strcmp(argument, "-allwild") == 0)
        {
            int val = atoi(argv[ci]);
            ci++;
            gAllWildMode = (val != 0);
            if (gAllWildMode) {
                gWildWildEnabled = true;  // All-wild requires wild-wild detection
            }
        }
        else if (strcmp(argument, "-checkpoint") == 0)
        {
            gCheckpointIntervalHours = atoi(argv[ci]);
            ci++;
        }
        else if (strcmp(argument, "-rotation") == 0)
        {
            int val = atoi(argv[ci]);
            ci++;
            gNoRotation = (val == 0);  // -rotation 0 = freeze (default), -rotation 1 = old rotation
            if (!gNoRotation) {
                printf("WARNING: Rotation enabled - may cause FP explosion on long runs!\n");
            }
        }
        else if (strcmp(argument, "-savefile") == 0)
        {
            strncpy(gCheckpointFile, argv[ci], sizeof(gCheckpointFile) - 1);
            ci++;
        }
        else if (strcmp(argument, "-loadwild") == 0)
        {
            strncpy(gLoadCheckpointFile, argv[ci], sizeof(gLoadCheckpointFile) - 1);
            ci++;
        }
        else if (strcmp(argument, "-savedps") == 0)
        {
            strncpy(gDiscardedDPsFile, argv[ci], sizeof(gDiscardedDPsFile) - 1);
            ci++;
        }
        else if (strcmp(argument, "-preloadwilds") == 0)
        {
            strncpy(gPreloadWildsFile, argv[ci], sizeof(gPreloadWildsFile) - 1);
            gPreloadIsCheckpoint = false;  // savedps format
            ci++;
        }
        else if (strcmp(argument, "-preloadckpt") == 0)
        {
            strncpy(gPreloadWildsFile, argv[ci], sizeof(gPreloadWildsFile) - 1);
            gPreloadIsCheckpoint = true;   // checkpoint RCKDT43 format
            ci++;
        }
        else
        {
            printf("Unknown option: %s\r\n", argument);
            return false;
        }
    }
    
    if (!gPubKey.x.IsZero() && (!gStartSet || !gRange || !gDP))
    {
        printf("Error: need -dp, -range and -start\r\n");
        return false;
    }
    
    if (gRAMLimitGB == 0) {
        printf("Error: -ramlimit required\r\n");
        return false;
    }
    
    return true;
}

int main(int argc, char* argv[])
{
    printf("\n");
    printf("================================================================================\n");
    printf("\n");
    printf("\033[1;33m 🦘 PSCKangaroo\033[0m — GPU-accelerated Pollard's Kangaroo for secp256k1 ECDLP\n");
    printf("\n");
    printf("\033[1;36m  Fork of RCKangaroo by RetiredCoder (RC)\033[0m\n");
    printf("  Original: https://github.com/RetiredC/RCKangaroo\n");
    printf("  Special thanks to RC for SOTA!\n");
    printf("  License:  GPLv3\n");
    printf("\n");
    printf("================================================================================\n");
    printf("\n");
    printf("  Core algorithm:  SOTA (Equivalence Classes + Negation Map, K~1.15)\n");
    printf("  Optimizations:   SOTA+ | ALL-TAME mode | 16-byte compact entries | Async BSGS\n");
    printf("                   Ultra-compact 16-byte DPs | Async BSGS resolver\n");
    printf("                   Dual hash table | Table freeze | Uniform jumps\n");
    printf("  Modes:           ALL-TAME (recommended 130+ bits) | ALL-WILD | TRAP/HUNT\n");
    printf("  Checkpoint:      Auto-save + Ctrl+C safe exit (format RCKDT5C)\n");
    printf("\n");

#ifdef _WIN32
    printf("  Platform: Windows\n\n");
#else
    printf("  Platform: Linux\n\n");
#endif

    InitEc();
    gDP = 0;
    gRange = 0;
    gStartSet = false;
    gTamesFileName[0] = 0;
    gMax = 0.0;
    gGenMode = false;
    gIsOpsLimit = false;
    gRAMLimitGB = 0.0;
    memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));
    
    signal(SIGTERM, SignalHandler);
    signal(SIGINT, CheckpointSignalHandler);  // Save checkpoint on Ctrl+C
    
    if (!ParseCommandLine(argc, argv))
    {
        printf("Usage:\n");
        printf("  ./psckangaroo -gpu 0 -dp 20 -range 135 -ramlimit 100 \\\n");
        printf("    -pubkey 02... -start 4000...\n");
        printf("\n");
        printf("Required:\n");
        printf("  -gpu N           GPU index (0, 1, ...)\n");
        printf("  -dp N            Distinguished point bits (6-60)\n");
        printf("  -range N         Key range in bits (32-170)\n");
        printf("  -pubkey <hex>    Target public key (compressed 02/03...)\n");
        printf("  -start <hex>     Range start offset\n");
        printf("  -ramlimit N      RAM limit in GB\n");
        printf("\n");
        printf("Optional:\n");
        printf("  -allwild 0       ALL-TAME mode (default, recommended for 130+ bits)\n");
        printf("  -allwild 1       ALL-WILD mode (W-W collisions only)\n");
        printf("  -groups N        Points per batch inversion (default: 24, max: 256)\n");
        printf("  -checkpoint N    Auto-save interval in hours (default: 4, 0=off)\n");
        printf("  -savefile <f>    Checkpoint filename\n");
        printf("  -loadwild <f>    Load checkpoint and resume\n");
        printf("  -savedps <f>     Save evicted DPs to file\n");
        printf("  -waveinterval N  Minutes between WILD wave renewals (default: 0=off)\n");
        printf("  -rotation 0/1    Table freeze (0=freeze, default) or rotation (1)\n");
        printf("\n");
        printf("Example (Puzzle #135):\n");
        printf("  ./psckangaroo -gpu 0 -dp 20 -range 135 -ramlimit 115 \\\n");
        printf("    -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \\\n");
        printf("    -start 4000000000000000000000000000000000\n");
        printf("\n");
        return 1;
    }

    InitGpus();

    if (!GpuCnt)
    {
        printf("No GPUs found!\r\n");
        return 1;
    }

    pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    TotalOps = 0;
    TotalSolved = 0;
    gTotalErrors = 0;
    IsBench = gPubKey.x.IsZero();

    if (!IsBench)
    {
        EcPoint PntToSolve, PntOfs;
        EcInt pk_found;

        PntToSolve = gPubKey;
        if (!gStart.IsZero())
        {
            PntOfs = ec.MultiplyG(gStart);
            PntOfs.y.NegModP();
            PntToSolve = ec.AddPoints(PntToSolve, PntOfs);
        }

        char sx[100], sy[100];
        gPubKey.x.GetHexStr(sx);
        gPubKey.y.GetHexStr(sy);
        printf("Target:\r\nX: %s\r\nY: %s\r\n", sx, sy);
        gStart.GetHexStr(sx);
        printf("Offset: %s\r\n\r\n", sx);

        if (SolvePoint(PntToSolve, gRange, gDP, &pk_found))
        {
            pk_found.AddModP(gStart);
            EcPoint tmp = ec.MultiplyG(pk_found);
            if (tmp.IsEqual(gPubKey))
            {
                char s[100];
                pk_found.GetHexStr(s);
                printf("\r\n");
                printf("################################################################################\r\n");
                printf("  PRIVATE KEY: %s\r\n", s);
                printf("################################################################################\r\n");
                printf("\r\n");
                FILE* fp = fopen("RESULTS.TXT", "a");
                if (fp)
                {
                    fprintf(fp, "PRIVATE KEY: %s\n", s);
                    fclose(fp);
                    printf("  Key saved to RESULTS.TXT\r\n\r\n");
                }
            }
            else
            {
                printf("ERROR: Key verification failed!\r\n");
            }
        }
    }
    else
    {
        printf("Benchmark mode not supported in TRAP/HUNT version\r\n");
    }

    for (int i = 0; i < GpuCnt; i++)
        delete GpuKangs[i];
    DeInitEc();
    free(pPntList2);
    free(pPntList);
    
    return 0;
}
