// TameStore.h
// TameStore v56C - ULTRA-COMPACT 16-BYTE ENTRIES (+56% capacity over v54)
// Based on v56A (20-byte compact, sentinel detection)
//
// v56C ULTRA-COMPACT ENTRY FORMAT (16 bytes, no valid array):
//   WildEntryCompact { u64 part1, u64 part2 }
//   part1: dist[32..95] (shifted distance, lower 32 bits truncated)
//   part2: dist[96..135](40b) | endo(2b) | type(2b) | x_sig(20b)
//   Empty slot = sentinel (all zeros). Safe because real distance >> 2^32.
//   Truncation: lower 32 bits of distance discarded. On collision resolution,
//   use BSGS (Baby-Step Giant-Step) in range ±2^32 to recover exact key (~400ms).
//   Saves 9 bytes/entry vs v54 (25→16): +56% capacity.
//
// v44 FEATURES (retained):
// 1. ExportRAMToSavedDPs() - On exit, exports ALL current RAM entries
//    to savedps file so offline cross-check.
// 2. Flush/sync savedps periodically to prevent data loss.
// 3. Stats tracking for disk usage (bytes written to savedps).
//
// RETAINED from v43:
// - Two independent hash tables: table[0]=WILD1, table[1]=WILD2
// - 100% cross-type collision detection
// - Hash-based rotation with direct slot[0] overwrite
// - Quadratic probing during fill phase
// - Shard-based spinlocks (independent per table)
// - Discarded DPs file support
// - Checkpoint save/load (format RCKDT45)

#pragma once

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <mutex>
#include "defs.h"

// ============================================================================
// CONFIGURATION
// ============================================================================

#define TAME_MAX_PROBE      256
#define PRIMARY_TABLE_MAX   (1ULL << V45_TABLE_BITS)
#define NUM_SHARDS          4096
#define SHARD_MASK          (NUM_SHARDS - 1)

#define COLLISION_NONE          0
#define COLLISION_BLOOM_HIT     1
#define COLLISION_TAME_WILD     2
#define COLLISION_WILD_WILD     3

// Spatial diversity stats (cosmetic only)
#define SPATIAL_BUCKETS         65536
#define SPATIAL_BUCKET_BITS     16
#define SPATIAL_BUCKET_MASK     (SPATIAL_BUCKETS - 1)

// Legacy Bloom defines (kept for compatibility, unused in v43)
#define TAME_BLOOM_SIZE     (2ULL * 1024 * 1024 * 1024 * 8)
#define TAME_BLOOM_HASHES   4
#define WILD_BLOOM_SIZE     (4ULL * 1024 * 1024 * 1024 * 8)
#define WILD_BLOOM_HASHES   3

// ============================================================================
// DATA STRUCTURES - v56 COMPACT (20 bytes, no valid array, sentinel=all-zero)
// ============================================================================
// v54: 24 bytes (WildEntryCompact) + 1 byte (valid) = 25 bytes/slot
// v56: 20 bytes (TameEntry20) + 0 bytes (sentinel)  = 20 bytes/slot → +25% capacity
//
// Layout of part3 (u32):
//   bits [0..7]:   dist_high (byte 16 of distance, 8 bits)
//   bits [8..9]:   endo_transform (2 bits, from meta >> 4)
//   bits [10..31]: x_sig (22 bits, lower 22 bits of x_bytes)
//
// Sentinel: entry with part1==0 && part2==0 && part3==0 means EMPTY.
// This is safe because a real entry always has nonzero distance (accumulates from random start).
// ============================================================================

#pragma pack(push, 1)
struct TameEntry {
    u8 x[16];
    u8 dist[24];
};

struct WildEntryCompact {
    u64 part1; // Dist[32..95] (shifted: original lower 32 bits truncated)
    u64 part2; // Dist[96..135](40b) | Endo(2b) | Type(2b) | X_Sig(20b)
};
#pragma pack(pop)

class SpinLock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
public:
    void lock() {
        while (flag.test_and_set(std::memory_order_acquire)) {
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
            #endif
        }
    }
    void unlock() {
        flag.clear(std::memory_order_release);
    }
};

// ============================================================================
// TAME STORE CLASS v56 - COMPACT 20-BYTE + DUAL TABLE + DISK
// ============================================================================

class TameStore {
private:
    // ========================================================================
    // DUAL TABLE STORAGE (v56: no valid arrays - sentinel detection)
    // table[0] = WILD1/TAME entries, table[1] = WILD2 entries
    // ========================================================================
    WildEntryCompact* wild_table[2];
    // v56: No valid arrays — sentinel detection (all-zero = empty)
    SpinLock* wild_shard_locks[2];
    bool circular_mode_active[2];
    bool table_frozen[2];         // v52: freeze table when full (no rotation)
    bool no_rotation;             // v52: disable rotation globally
    bool all_tame_mode;           // v53: ALL RAM for TAMEs, no WILD storage
    
    u64 wild_table_size;      // Same size for both tables
    u64 wild_table_mask;
    u64 wild_slots_per_shard;
    
    // Per-table counters
    std::atomic<u64> wild_count[2];
    std::atomic<u64> table_overwrites[2];
    
    // Spatial bucket tracking (combined stats)
    struct SpatialBucket {
        u64 start_slot;
        u64 slot_count;
        std::atomic<u64> write_index{0};
        std::atomic<u64> fill_count{0};
    };
    SpatialBucket* spatial_buckets;
    
    // Shared statistics
    std::atomic<u64> total_checks{0};
    std::atomic<u64> wild_wild_collisions{0};
    std::atomic<u64> tame_wild_collisions{0};
    std::atomic<u64> duplicate_points{0};
    std::atomic<u64> false_positives{0};
    
    // Legacy stats (API compatibility)
    std::atomic<u64> bloom_checks{0};
    std::atomic<u64> bloom_hits{0};
    std::atomic<u64> spatial_overwrites{0};
    std::atomic<u64> spatial_rotations{0};
    
    // Discarded DPs file
    FILE* discarded_file = nullptr;
    std::atomic<u64> discarded_saved{0};
    std::atomic<u64> discarded_bytes{0};  // v44: track disk usage
    SpinLock discarded_file_lock;
    
    // Collision data
    std::atomic<bool> has_collision{false};
    SpinLock collision_lock;
    u8 collision_x[16];
    u8 collision_dist1[24];
    u8 collision_dist2[24];
    int collision_type;
    
    bool initialized = false;
    bool wild_store_enabled = true;
    
    // Legacy (unused, API compat)
    u8* bloom = nullptr;
    size_t bloom_bytes = 0;
    u8* wild_bloom = nullptr;
    size_t wild_bloom_bytes = 0;
    TameEntry* table = nullptr;
    u8* valid = nullptr;
    u64 table_size = 0;
    u64 table_mask = 0;
    TameEntry* overflow_table = nullptr;
    u8* overflow_valid = nullptr;
    u64 overflow_size = 0;
    u64 overflow_mask = 0;
    SpinLock* shard_locks = nullptr;
    SpinLock* overflow_shard_locks = nullptr;
    u64 slots_per_shard = 0;
    u64 overflow_slots_per_shard = 0;
    std::atomic<u64> tame_count{0};
    std::atomic<u64> overflow_count{0};
    
    // ========================================================================
    // HELPERS
    // ========================================================================
    
    inline u32 GetSpatialBucketIdx(const u8* x_bytes) {
        return ((u32)x_bytes[0] << 8 | (u32)x_bytes[1]) & SPATIAL_BUCKET_MASK;
    }
    
    inline u64 TableHash(const u8* x_bytes) {
        u64 h = *(u64*)x_bytes;
        h ^= h >> 33; h *= 0xFF51AFD7ED558CCD; h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53; h ^= h >> 33;
        return h;
    }
    
    inline int GetShard(u64 slot) { 
        return (slot / wild_slots_per_shard) & SHARD_MASK; 
    }
    
    // ========================================================================
    // PACK/UNPACK - v56C ULTRA-COMPACT (16 bytes, 32-bit distance truncation)
    // Lower 32 bits (4 bytes) of distance are discarded.
    // part1: dist_bytes[4..11] (bits 32..95 of original distance)
    // part2 layout (u64):
    //   bits [0..39]:  dist_bytes[12..16] (bits 96..135 of original distance)
    //   bits [40..41]: endo_transform (2 bits)
    //   bits [42..43]: kang_type (2 bits)
    //   bits [44..63]: x_sig (20 bits)
    // On collision resolution, BSGS searches ±2^32 to recover exact key.
    // ========================================================================
    
    inline void PackWild(WildEntryCompact* entry, const u8* x_bytes, const u8* dist_bytes) {
        // Truncate lower 32 bits: skip dist_bytes[0..3]
        entry->part1 = *(u64*)(dist_bytes + 4);  // bytes 4..11
        
        u64 d_upper = 0;
        memcpy(&d_upper, dist_bytes + 12, 5);    // bytes 12..16 (40 bits)
        
        u8 meta = dist_bytes[22];
        u8 endo = (meta >> 4) & 0x03;
        u8 type = meta & 0x03;
        u32 x_sig = (*(u32*)x_bytes) & 0x000FFFFFU;  // 20 bits
        
        entry->part1 = *(u64*)(dist_bytes + 4);
        entry->part2 = d_upper 
                     | ((u64)endo << 40) 
                     | ((u64)type << 42) 
                     | ((u64)x_sig << 44);
    }
    
    inline bool CheckXSigMatch(WildEntryCompact* entry, const u8* x_bytes, u8* out_dist_bytes) {
        u32 stored_x_sig = (u32)(entry->part2 >> 44) & 0x000FFFFFU;
        u32 incoming_x_sig = (*(u32*)x_bytes) & 0x000FFFFFU;
        
        if (stored_x_sig != incoming_x_sig) return false;
        
        if (out_dist_bytes) {
            memset(out_dist_bytes, 0, 24);  // clear all (lower 4 bytes stay zero = truncated)
            *(u64*)(out_dist_bytes + 4) = entry->part1;  // bytes 4..11
            u64 d_upper = entry->part2 & 0x000000FFFFFFFFFFULL;  // lower 40 bits
            memcpy(out_dist_bytes + 12, &d_upper, 5);            // bytes 12..16
            
            u8 endo = (entry->part2 >> 40) & 0x03;
            u8 type = (entry->part2 >> 42) & 0x03;
            out_dist_bytes[22] = (endo << 4) | type;
            out_dist_bytes[23] = 0;
        }
        return true;
    }
    
    // v56C: Sentinel check — all-zero entry means empty slot
    inline bool IsSlotEmpty(const WildEntryCompact* entry) {
        return entry->part1 == 0 && entry->part2 == 0;
    }
    
    inline void ClearSlot(WildEntryCompact* entry) {
        entry->part1 = 0; entry->part2 = 0;
    }
    
    // Legacy alias
    inline bool CheckWildCollision(WildEntryCompact* entry, const u8* x_bytes, u8* out_dist_bytes) {
        return CheckXSigMatch(entry, x_bytes, out_dist_bytes);
    }
    
    // ========================================================================
    // DISCARDED DPs FILE
    // ========================================================================
    
    inline void SaveDiscardedDP(WildEntryCompact* entry) {
        if (!discarded_file) return;
        discarded_file_lock.lock();
        if (discarded_file) {
            fwrite(entry, sizeof(WildEntryCompact), 1, discarded_file);
            discarded_saved.fetch_add(1, std::memory_order_relaxed);
            discarded_bytes.fetch_add(sizeof(WildEntryCompact), std::memory_order_relaxed);
        }
        discarded_file_lock.unlock();
    }
    
    // ========================================================================
    // REPORT COLLISION (shared helper)
    // ========================================================================
    
    inline int ReportCollision(const u8* x_bytes, const u8* dist_existing, const u8* dist_new) {
        return ReportCollisionTyped(x_bytes, dist_existing, dist_new, COLLISION_WILD_WILD);
    }
    
    inline int ReportCollisionTyped(const u8* x_bytes, const u8* dist_existing, const u8* dist_new, int coll_type) {
        if (coll_type == COLLISION_TAME_WILD)
            tame_wild_collisions.fetch_add(1, std::memory_order_relaxed);
        else
            wild_wild_collisions.fetch_add(1, std::memory_order_relaxed);
        collision_lock.lock();
        has_collision.store(true, std::memory_order_relaxed);
        memcpy(collision_x, x_bytes, 16);
        memcpy(collision_dist1, dist_existing, 24);
        memcpy(collision_dist2, dist_new, 24);
        collision_type = coll_type;
        collision_lock.unlock();
        return coll_type;
    }
    
public:
    TameStore() : spatial_buckets(nullptr) {
        wild_table[0] = wild_table[1] = nullptr;
        wild_shard_locks[0] = wild_shard_locks[1] = nullptr;
        circular_mode_active[0] = circular_mode_active[1] = false;
        table_frozen[0] = table_frozen[1] = false;
        no_rotation = false;
        all_tame_mode = false;
    }
    
    ~TameStore() { Destroy(); }
    
    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    
    bool Init(size_t ram_limit_gb, bool enable_wild_store = true) {
        wild_store_enabled = enable_wild_store;
        return InitWildOnly(ram_limit_gb);
    }
    
    bool InitWildOnly(size_t ram_limit_gb) {
        wild_store_enabled = true;
        all_tame_mode = false;
        
        size_t ram_bytes = ram_limit_gb * 1024ULL * 1024ULL * 1024ULL;
        
        // v43: No Bloom filter - all RAM goes to tables
        wild_bloom_bytes = 0;
        wild_bloom = nullptr;
        bloom_bytes = 0;
        
        if (ram_bytes < 2ULL * 1024 * 1024 * 1024) {
            printf("ERROR: Not enough RAM (need at least 2GB)\n"); 
            return false;
        }

        // Reserve overhead
        size_t bucket_overhead = SPATIAL_BUCKETS * sizeof(SpatialBucket);
        size_t available = ram_bytes - 512ULL * 1024 * 1024 - bucket_overhead;
        
        // v56C: Each entry = 16 bytes (ultra-compact, no valid array)
        // Two tables, each gets half the available memory
        size_t entry_size = sizeof(WildEntryCompact);  // 16 bytes
        size_t per_table_available = available / 2;
        
        wild_table_size = per_table_available / entry_size;
        // v54 PERF: Round down to power of 2 for fast & masking (eliminates 64-bit division)
        // Find highest power of 2 that fits
        u64 po2 = 1;
        while (po2 * 2 <= wild_table_size) po2 *= 2;
        wild_table_size = po2;
        wild_table_mask = wild_table_size - 1;
        // Cap at PRIMARY_TABLE_MAX
        if (wild_table_size > PRIMARY_TABLE_MAX) {
            wild_table_size = PRIMARY_TABLE_MAX;
            wild_table_mask = wild_table_size - 1;
        }
        
        size_t total_table_bytes = 2 * wild_table_size * entry_size;
        
        printf("========================================================================\n");
        printf("TameStore v56C ULTRA-COMPACT ALLWILD + DUAL TABLE + DISK (16 bytes/entry)\n");
        printf("========================================================================\n");
        printf("  Config: Hybrid mode=ENABLED (table[0]=TAME, table[1]=WILD+preload)\n");
        printf("  Config: Stratified=%d, ExplorerPct=%d, ExplorerShift=%d\n", V46_ENABLE, V46_EXPLORER_PCT, V46_EXPLORER_SHIFT);
        printf("  Config: Occupancy=%d, GroupCnt=%d, TableBits=%d\n", V45_OCCUPANCY, V45_PNT_GROUP_CNT, V45_TABLE_BITS);
        printf("  RAM limit: %.1f GB\n", (double)ram_limit_gb);
        printf("  Bloom filter: REMOVED (saves 4GB => more entries)\n");
        printf("  WILD1 table: %llu entries (%.1f GB)\n",
               (unsigned long long)wild_table_size,
               (double)(wild_table_size * entry_size) / (1024.0*1024*1024));
        printf("  WILD2 table: %llu entries (%.1f GB)\n",
               (unsigned long long)wild_table_size,
               (double)(wild_table_size * entry_size) / (1024.0*1024*1024));
        printf("  Total: %llu entries (%.1f GB)\n",
               (unsigned long long)(wild_table_size * 2),
               (double)(total_table_bytes) / (1024.0*1024*1024));
        printf("  Cross-type checks: 100%% (vs ~50%% in unified table)\n");
        printf("  Spatial buckets: %d\n", SPATIAL_BUCKETS);
        printf("========================================================================\n");
        
        // Allocate both tables (v56: no valid arrays)
        for (int t = 0; t < 2; t++) {
            wild_table[t] = (WildEntryCompact*)calloc(wild_table_size, sizeof(WildEntryCompact));
            
            if (!wild_table[t]) {
                printf("ERROR: Memory allocation failed for table %d\n", t);
                Destroy();
                return false;
            }
            
            wild_shard_locks[t] = new SpinLock[NUM_SHARDS];
            circular_mode_active[t] = false;
            table_frozen[t] = false;
            wild_count[t].store(0);
            table_overwrites[t].store(0);
        }
        
        wild_slots_per_shard = wild_table_size / NUM_SHARDS;
        
        // Initialize spatial buckets
        spatial_buckets = new SpatialBucket[SPATIAL_BUCKETS];
        u64 spb = wild_table_size / SPATIAL_BUCKETS;
        
        for (u32 i = 0; i < SPATIAL_BUCKETS; i++) {
            spatial_buckets[i].start_slot = i * spb;
            spatial_buckets[i].slot_count = spb;
            spatial_buckets[i].write_index.store(0);
            spatial_buckets[i].fill_count.store(0);
        }
        
        u64 remainder = wild_table_size - (spb * SPATIAL_BUCKETS);
        if (remainder > 0) {
            spatial_buckets[SPATIAL_BUCKETS - 1].slot_count += remainder;
        }
        
        initialized = true;
        printf("TameStore OPTIMIZED + DUAL TABLE + DISK: Ready!\n");
        return true;
    }
    
    // ========================================================================
    // v53: ALL-TAME MODE - All RAM goes to table[0] for maximum TAME coverage
    // Table[1] is not allocated. WILDs only check against TAMEs, never stored.
    // This doubles the number of TAMEs → 2x T-W collision probability per check.
    // ========================================================================
    bool InitAllTame(size_t ram_limit_gb) {
        wild_store_enabled = false;
        all_tame_mode = true;
        
        size_t ram_bytes = ram_limit_gb * 1024ULL * 1024ULL * 1024ULL;
        
        wild_bloom_bytes = 0;
        wild_bloom = nullptr;
        bloom_bytes = 0;
        
        if (ram_bytes < 2ULL * 1024 * 1024 * 1024) {
            printf("ERROR: Not enough RAM (need at least 2GB)\n"); 
            return false;
        }

        size_t bucket_overhead = SPATIAL_BUCKETS * sizeof(SpatialBucket);
        size_t available = ram_bytes - 512ULL * 1024 * 1024 - bucket_overhead;
        
        // v56C: ALL available memory goes to table[0] (TAMEs)
        // 16 bytes per entry (ultra-compact, no valid array)
        size_t entry_size = sizeof(WildEntryCompact);  // 16 bytes
        wild_table_size = available / entry_size;
        // Align to SPATIAL_BUCKETS (65536) for clean bucket division
        wild_table_size = (wild_table_size / SPATIAL_BUCKETS) * SPATIAL_BUCKETS;
        wild_table_mask = 0;  // Not used — all lookups use % wild_table_size
        
        size_t total_table_bytes = wild_table_size * entry_size;
        
        printf("========================================================================\n");
        printf("TameStore v56C ALL-TAME MODE (ULTRA-COMPACT 16-BYTE)\n");
        printf("========================================================================\n");
        printf("  Config: ALL RAM → TAMEs (no WILD storage, maximum T-W probability)\n");
        printf("  Config: Occupancy=%d, GroupCnt=%d\n", V45_OCCUPANCY, V45_PNT_GROUP_CNT);
        printf("  RAM limit: %.1f GB\n", (double)ram_limit_gb);
        printf("  TAME table: %llu entries (%.1f GB) [16 bytes/entry, 32-bit truncation]\n",
               (unsigned long long)wild_table_size,
               (double)(total_table_bytes) / (1024.0*1024*1024));
        printf("  WILD table: NONE (WILDs check-only, never stored)\n");
        printf("  Advantage: +56%% more TAMEs vs v54 (16 vs 25 bytes/entry)!\n");
        printf("  NOTE: Uses BSGS (~400ms) to resolve truncated distances on collision.\n");
        printf("  Spatial buckets: %d\n", SPATIAL_BUCKETS);
        printf("========================================================================\n");
        
        // Allocate ONLY table[0] (v56: no valid array)
        wild_table[0] = (WildEntryCompact*)calloc(wild_table_size, sizeof(WildEntryCompact));
        if (!wild_table[0]) {
            printf("ERROR: Memory allocation failed for TAME table\n");
            return false;
        }
        wild_shard_locks[0] = new SpinLock[NUM_SHARDS];
        circular_mode_active[0] = false;
        table_frozen[0] = false;
        wild_count[0].store(0);
        table_overwrites[0].store(0);
        
        // Table[1] = not allocated
        wild_table[1] = nullptr;
        wild_shard_locks[1] = nullptr;
        circular_mode_active[1] = false;
        table_frozen[1] = true;  // Mark as "frozen" so nothing tries to write
        wild_count[1].store(0);
        table_overwrites[1].store(0);
        
        wild_slots_per_shard = wild_table_size / NUM_SHARDS;
        
        // Initialize spatial buckets
        spatial_buckets = new SpatialBucket[SPATIAL_BUCKETS];
        u64 spb = wild_table_size / SPATIAL_BUCKETS;
        for (u32 i = 0; i < SPATIAL_BUCKETS; i++) {
            spatial_buckets[i].start_slot = i * spb;
            spatial_buckets[i].slot_count = spb;
            spatial_buckets[i].write_index.store(0);
            spatial_buckets[i].fill_count.store(0);
        }
        u64 remainder = wild_table_size - (spb * SPATIAL_BUCKETS);
        if (remainder > 0) {
            spatial_buckets[SPATIAL_BUCKETS - 1].slot_count += remainder;
        }
        
        initialized = true;
        printf("TameStore ALL-TAME: Ready! (2x TAMEs, 0 WILD storage, 16 bytes/entry)\n");
        return true;
    }
    
    void Destroy() {
        CloseDiscardedFile();
        for (int t = 0; t < 2; t++) {
            if (wild_table[t]) free(wild_table[t]);
            if (wild_shard_locks[t]) delete[] wild_shard_locks[t];
            wild_table[t] = nullptr;
            wild_shard_locks[t] = nullptr;
        }
        if (spatial_buckets) delete[] spatial_buckets;
        spatial_buckets = nullptr;
        initialized = false;
    }
    
    // ========================================================================
    // DISCARDED FILE MANAGEMENT
    // ========================================================================
    
    bool OpenDiscardedFile(const char* filename) {
        discarded_file_lock.lock();
        if (discarded_file) fclose(discarded_file);
        discarded_file = fopen(filename, "ab");
        discarded_file_lock.unlock();
        
        if (!discarded_file) {
            printf("ERROR: Cannot open discarded DPs file: %s\n", filename);
            return false;
        }
        printf("Saving discarded DPs to: %s\n", filename);
        return true;
    }
    
    void CloseDiscardedFile() {
        discarded_file_lock.lock();
        if (discarded_file) {
            fclose(discarded_file);
            discarded_file = nullptr;
            printf("Discarded DPs file closed. Total saved: %llu\n", 
                   (unsigned long long)discarded_saved.load());
        }
        discarded_file_lock.unlock();
    }
    
    u64 GetDiscardedSaved() { return discarded_saved.load(); }
    u64 GetDiscardedBytes() { return discarded_bytes.load(); }  // v44
    
    // v44: Flush savedps to disk (call periodically to prevent data loss)
    void FlushDiscardedFile() {
        discarded_file_lock.lock();
        if (discarded_file) {
            fflush(discarded_file);
        }
        discarded_file_lock.unlock();
    }
    
    // v44: Export ALL valid entries from both RAM tables to savedps file
    // Call on exit (Ctrl+C) to export table contents for offline cross-check
    // Returns number of entries exported
    u64 ExportRAMToSavedDPs() {
        if (!discarded_file || !initialized) return 0;
        
        printf("\n*** EXPORTING RAM TABLES TO SAVEDPS ***\n");
        
        u64 exported = 0;
        time_t start = time(NULL);
        
        for (int t = 0; t < 2; t++) {
            const char* tname = (t == 0) ? "W1" : "W2";
            u64 table_exported = 0;
            
            // v53: Skip non-allocated tables
            if (!wild_table[t]) {
                printf("  %s: not allocated (skipped)\n", tname);
                continue;
            }
            
            for (u64 i = 0; i < wild_table_size; i++) {
                if (!IsSlotEmpty(&wild_table[t][i])) {
                    // Write directly - no lock needed, GPU is stopped
                    fwrite(&wild_table[t][i], sizeof(WildEntryCompact), 1, discarded_file);
                    table_exported++;
                }
                
                // Progress every 100M entries
                if (i % 100000000 == 0 && i > 0) {
                    printf("  %s: exported %lluM / %lluM entries...\r", 
                           tname,
                           (unsigned long long)(table_exported / 1000000),
                           (unsigned long long)(wild_count[t].load() / 1000000));
                    fflush(stdout);
                }
            }
            
            exported += table_exported;
            printf("  %s: exported %llu entries                    \n", 
                   tname, (unsigned long long)table_exported);
        }
        
        fflush(discarded_file);
        
        u64 export_bytes = exported * sizeof(WildEntryCompact);
        discarded_saved.fetch_add(exported, std::memory_order_relaxed);
        discarded_bytes.fetch_add(export_bytes, std::memory_order_relaxed);
        
        time_t elapsed = time(NULL) - start;
        printf("*** RAM EXPORT COMPLETE: %llu entries (%.1f GB) in %llds ***\n\n",
               (unsigned long long)exported,
               (double)export_bytes / (1024.0*1024*1024),
               (long long)elapsed);
        
        return exported;
    }
    
    // ========================================================================
    // ADD TO TABLE - Store DP in specified table (0=WILD1, 1=WILD2)
    //
    // No collision check needed here! We're storing in own-type table,
    // so any X_sig match is just a duplicate (same type = useless).
    // ========================================================================
    
    bool AddToTable(int tidx, const u8* x_bytes, const u8* dist_bytes) {
        // v53: Protect against writes to non-allocated tables
        if (!wild_table[tidx]) return false;
        
        u64 idx = TableHash(x_bytes) % wild_table_size;
        int shard = GetShard(idx);
        wild_shard_locks[tidx][shard].lock();
        
        // When already rotating (and not frozen), skip probing - direct slot[0]
        if (circular_mode_active[tidx] && !no_rotation) goto do_overwrite;
        
        // v52: When frozen, DP was already checked against opposite table
        // Just discard it silently (table is static = no probe chain corruption)
        if (table_frozen[tidx]) {
            wild_shard_locks[tidx][shard].unlock();
            return false;
        }
        
        // Fill phase: Quadratic probing for empty slot or duplicate
        for (int probe = 0; probe < TAME_MAX_PROBE; probe++) {
            u64 slot = (idx + (u64)probe * (probe + 1) / 2) % wild_table_size;
            
            if (IsSlotEmpty(&wild_table[tidx][slot])) {
                // Empty slot - store here
                PackWild(&wild_table[tidx][slot], x_bytes, dist_bytes);
                wild_shard_locks[tidx][shard].unlock();
                wild_count[tidx].fetch_add(1, std::memory_order_relaxed);
                
                u32 bucket_id = GetSpatialBucketIdx(x_bytes);
                spatial_buckets[bucket_id].fill_count.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            
            // Check for duplicate X_sig in same-type table
            u8 tmp[24];
            if (CheckXSigMatch(&wild_table[tidx][slot], x_bytes, tmp)) {
                // Duplicate - skip
                wild_shard_locks[tidx][shard].unlock();
                duplicate_points.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
        }
        
        // ================================================================
        // TABLE FULL: Freeze (discard DP) or Rotate (overwrite slot[0])
        // ================================================================
        do_overwrite:
        
        // v52: FREEZE MODE - table stays static, no FP explosion
        if (no_rotation) {
            if (!table_frozen[tidx]) {
                table_frozen[tidx] = true;
                const char* tname = (tidx == 0) ? "TABLE0 (TAME/WILD1)" : "TABLE1 (WILD/WILD2)";
                printf("\n*** TABLE FROZEN: %s (%.1f%% full, %lluM entries) - new DPs checked then discarded ***\n",
                    tname, 100.0 * wild_count[tidx].load() / wild_table_size,
                    (unsigned long long)(wild_count[tidx].load() / 1000000));
            }
            wild_shard_locks[tidx][shard].unlock();
            // DP is still checked against OPPOSITE table before reaching here
            // It just won't be stored in own table (table is frozen)
            return false;
        }
        
        // ROTATION MODE (legacy): overwrite slot[0]
        if (!circular_mode_active[tidx]) {
            circular_mode_active[tidx] = true;
            const char* tname = (tidx == 0) ? "TABLE0 (TAME/WILD1)" : "TABLE1 (WILD/WILD2)";
            printf("\n*** ROTATION ACTIVATED: %s table ***\n", tname);
        }
        
        u64 target_slot = idx;
        
        // Save discarded DP before overwriting
        if (!IsSlotEmpty(&wild_table[tidx][target_slot]) && discarded_file) {
            SaveDiscardedDP(&wild_table[tidx][target_slot]);
        }
        
        // Overwrite slot[0] - always findable by opposite table's search
        PackWild(&wild_table[tidx][target_slot], x_bytes, dist_bytes);
        table_overwrites[tidx].fetch_add(1, std::memory_order_relaxed);
        
        u32 bucket_id = GetSpatialBucketIdx(x_bytes);
        spatial_buckets[bucket_id].write_index.fetch_add(1, std::memory_order_relaxed);
        
        wild_shard_locks[tidx][shard].unlock();
        return true;
    }
    
    // Legacy wrapper
    bool AddWild(const u8* x_bytes, const u8* dist_bytes) {
        u8 type = dist_bytes[22] & 0x03;
        if (type < 1 || type > 2) return false;
        return AddToTable(type - 1, x_bytes, dist_bytes);
    }
    
    // ========================================================================
    // CHECK WILD ONLY - v43: DUAL TABLE CROSS-CHECK
    //
    // WILD1 arrives -> search WILD2 table (idx=1) -> store in WILD1 table (idx=0)
    // WILD2 arrives -> search WILD1 table (idx=0) -> store in WILD2 table (idx=1)
    //
    // Every X_sig match in opposite table is GUARANTEED cross-type.
    // No type checking needed. Different distances => COLLISION.
    // ========================================================================
    
    int CheckWildOnly(const u8* x_bytes, const u8* wild_dist_bytes) {
        if (!initialized) return COLLISION_NONE;
        
        // Determine table indices from kangaroo type
        u8 type = wild_dist_bytes[22] & 0x03;
        if (type < 1 || type > 2) return COLLISION_NONE;
        
        int store_idx = type - 1;       // WILD1->0, WILD2->1
        int check_idx = 1 - store_idx;  // Opposite table
        
        u64 idx = TableHash(x_bytes) % wild_table_size;
        total_checks.fetch_add(1, std::memory_order_relaxed);
        
        // ================================================================
        // v49 FIX: ALWAYS use quadratic probing for reads!
        //
        // BUG FOUND: Rotation mode only checked slot[0], but entries
        // stored during fill phase at probe positions 1..255 were
        // INVISIBLE after rotation. This caused puzzle 80 to fail
        // (230 hash FPs, 0 real collisions) while puzzle 69 worked
        // (resolved before rotation activated at ~4% load).
        //
        // Fix: Read path always probes. Write path still uses direct
        // overwrite at slot[0] in rotation mode (that's fine).
        // Performance: ~4 extra reads per check at 93% load = negligible.
        // ================================================================
        int max_probe = circular_mode_active[check_idx] ? 16 : TAME_MAX_PROBE;
        
        for (int probe = 0; probe < max_probe; probe++) {
            u64 slot = (idx + (u64)probe * (probe + 1) / 2) % wild_table_size;
            
            int shard = GetShard(slot);
            wild_shard_locks[check_idx][shard].lock();
            WildEntryCompact entry_copy = wild_table[check_idx][slot];
            wild_shard_locks[check_idx][shard].unlock();
            
            if (IsSlotEmpty(&entry_copy)) break;
            
            u8 recovered_dist[24];
            if (CheckXSigMatch(&entry_copy, x_bytes, recovered_dist)) {
                // X_sig match in opposite table = guaranteed cross-type!
                // v56C FIX: Compare from byte 4 onward (bytes 0-3 are truncated/zeroed)
                // Old code: memcmp(recovered_dist, wild_dist_bytes, 22) always != 0
                //           because recovered[0..3]=0 vs real[0..3]!=0 → duplicate path was dead code
                if (memcmp(recovered_dist + 4, wild_dist_bytes + 4, 18) != 0) {
                    // Different distances => REAL COLLISION!
                    int result = ReportCollision(x_bytes, recovered_dist, wild_dist_bytes);
                    return result;
                }
                // Same distance (upper 144 bits match) = duplicate
                break;
            }
        }
        
        // No collision -> store in own table
        AddToTable(store_idx, x_bytes, wild_dist_bytes);
        
        if (has_collision.load(std::memory_order_acquire)) {
            return COLLISION_WILD_WILD;
        }
        return COLLISION_NONE;
    }
    
    // Tame-Wild mode: Store tames in table[0], wilds in table[1]
    bool AddTame(const u8* x, const u8* d) { 
        return AddToTable(0, x, d);  // table[0] = TAME storage
    }
    // ========================================================================
    // V48 HYBRID CHECK: Check wild against BOTH tables
    //   table[0] = W1 from TRAP → W-W collision (cross-type)
    //   table[1] = W2 from TRAP (+ pre-loaded) → W-W collision
    //   Then store new wild in appropriate table
    // v48 FIX: All reads under shard lock to prevent torn reads!
    // ========================================================================
    int CheckWild(const u8* x_bytes, const u8* wild_dist_bytes) {
        if (!initialized) return COLLISION_NONE;
        
        u64 idx = TableHash(x_bytes) % wild_table_size;
        total_checks.fetch_add(1, std::memory_order_relaxed);
        
        // v53 ALL-TAME: Only check table[0] (TAMEs), skip table[1]
        int max_tables = all_tame_mode ? 1 : 2;
        
        for (int t = 0; t < max_tables; t++) {
            int coll_type = (t == 0) ? COLLISION_TAME_WILD : COLLISION_WILD_WILD;
            
            if (t == 1 && wild_count[1].load(std::memory_order_relaxed) == 0) continue;
            
            int max_probe = circular_mode_active[t] ? 16 : TAME_MAX_PROBE;
            for (int probe = 0; probe < max_probe; probe++) {
                u64 slot = (idx + (u64)probe * (probe + 1) / 2) % wild_table_size;
                
                int shard = GetShard(slot);
                wild_shard_locks[t][shard].lock();
                WildEntryCompact entry_copy = wild_table[t][slot];
                wild_shard_locks[t][shard].unlock();
                
                if (IsSlotEmpty(&entry_copy)) break;
                
                u8 recovered_dist[24];
                if (CheckXSigMatch(&entry_copy, x_bytes, recovered_dist)) {
                    // v56C FIX: Compare from byte 4 (bytes 0-3 truncated/zeroed)
                    if (memcmp(recovered_dist + 4, wild_dist_bytes + 4, 18) != 0) {
                        return ReportCollisionTyped(x_bytes, recovered_dist, wild_dist_bytes, coll_type);
                    }
                    break;
                }
            }
        }
        
        // v53: In all_tame_mode, WILDs are NEVER stored — check-only against TAMEs
        if (!all_tame_mode) {
            AddToTable(1, x_bytes, wild_dist_bytes);
        }
        
        if (has_collision.load(std::memory_order_acquire)) {
            return all_tame_mode ? COLLISION_TAME_WILD : COLLISION_WILD_WILD;
        }
        return COLLISION_NONE;
    }
    
    bool HasCollision() { return has_collision.load(std::memory_order_acquire); }
    
    bool GetCollisionData(u8* out_dist1, u8* out_dist2, int* out_type = nullptr) {
        if (!has_collision.load(std::memory_order_acquire)) return false;
        collision_lock.lock();
        if (out_dist1) memcpy(out_dist1, collision_dist1, 24);
        if (out_dist2) memcpy(out_dist2, collision_dist2, 24);
        if (out_type) *out_type = collision_type;
        collision_lock.unlock();
        return true;
    }
    
    void ClearCollision() {
        collision_lock.lock();
        has_collision.store(false, std::memory_order_relaxed);
        collision_lock.unlock();
    }
    
    // ========================================================================
    // STATISTICS
    // ========================================================================
    
    u64 GetWildCount() { 
        return wild_count[0].load(std::memory_order_relaxed) + 
               wild_count[1].load(std::memory_order_relaxed); 
    }
    u64 GetWild1Count() { return wild_count[0].load(std::memory_order_relaxed); }
    u64 GetWild2Count() { return wild_count[1].load(std::memory_order_relaxed); }
    u64 GetWildTableSize() { return all_tame_mode ? wild_table_size : wild_table_size * 2; }
    u64 GetPerTableSize() { return wild_table_size; }
    u64 GetWildWildCollisions() { return wild_wild_collisions.load(); }
    u64 GetDuplicatePoints() { return duplicate_points.load(); }
    u64 GetTameWildCollisions() { return tame_wild_collisions.load(); }
    u64 GetTameCount() { return wild_count[0].load(std::memory_order_relaxed); }
    u64 GetPrimarySize() { return wild_table_size; }
    u64 GetOverflowSize() { return 0; }
    u64 GetOverflowCount() { return 0; }
    u64 GetSpatialOverwrites() { 
        return table_overwrites[0].load() + table_overwrites[1].load(); 
    }
    u64 GetSpatialRotations() { return spatial_rotations.load(); }
    
    void SetNoRotation(bool val) { no_rotation = val; }
    bool IsNoRotation() { return no_rotation; }
    bool IsTableFrozen(int t) { return table_frozen[t]; }
    bool IsAnyFrozen() { return table_frozen[0] || table_frozen[1]; }
    bool IsAllTameMode() { return all_tame_mode; }
    
    double GetWildLoadFactor() {
        u64 total = wild_count[0].load() + wild_count[1].load();
        u64 capacity = all_tame_mode ? wild_table_size : wild_table_size * 2;
        return (double)total / capacity * 100.0;
    }
    
    double GetFalsePositiveRate() {
        u64 checks = total_checks.load();
        return checks ? (double)false_positives.load() / checks * 100.0 : 0;
    }
    
    void PrintStats() {
        printf("\n");
        printf("=============================================================\n");
        printf("TameStore Stats%s:\n", all_tame_mode ? " (ALL-TAME mode)" : " (Dual Table + Disk)");
        printf("=============================================================\n");
        if (all_tame_mode) {
            printf("  TAMEs stored: %llu / %llu (%.2f%%)\n", 
                   (unsigned long long)wild_count[0].load(), 
                   (unsigned long long)wild_table_size, 
                   (double)wild_count[0].load() / wild_table_size * 100.0);
            printf("  WILDs: check-only (no storage)\n");
        } else {
            printf("  WILD1 stored: %llu / %llu (%.2f%%)\n", 
                   (unsigned long long)wild_count[0].load(), 
                   (unsigned long long)wild_table_size, 
                   (double)wild_count[0].load() / wild_table_size * 100.0);
            printf("  WILD2 stored: %llu / %llu (%.2f%%)\n", 
                   (unsigned long long)wild_count[1].load(), 
                   (unsigned long long)wild_table_size, 
                   (double)wild_count[1].load() / wild_table_size * 100.0);
            printf("  Total: %llu / %llu (%.2f%%)\n",
                   (unsigned long long)GetWildCount(),
                   (unsigned long long)(wild_table_size * 2),
                   GetWildLoadFactor());
        }
        
        bool any_rotating = circular_mode_active[0] || circular_mode_active[1];
        bool any_frozen = table_frozen[0] || table_frozen[1];
        if (any_frozen) {
            printf("  Tables: W1=%s, W2=%s (no rotation = no FP explosion)\n",
                   table_frozen[0] ? "FROZEN" : "filling",
                   table_frozen[1] ? "FROZEN" : "filling");
        } else if (any_rotating) {
            printf("  Rotation: W1=%s, W2=%s\n",
                   circular_mode_active[0] ? "ACTIVE" : "filling",
                   circular_mode_active[1] ? "ACTIVE" : "filling");
            printf("  Overwrites: W1=%llu, W2=%llu\n",
                   (unsigned long long)table_overwrites[0].load(),
                   (unsigned long long)table_overwrites[1].load());
            
            if (discarded_file) {
                printf("  Discarded DPs Saved: %llu\n", 
                       (unsigned long long)discarded_saved.load());
            }
        }
        
        printf("  Total checks: %llu\n", (unsigned long long)total_checks.load());
        printf("  Tame-Wild collisions: %llu\n", 
               (unsigned long long)tame_wild_collisions.load());
        printf("  Wild-Wild collisions: %llu\n", 
               (unsigned long long)wild_wild_collisions.load());
        printf("  Duplicates: %llu\n", 
               (unsigned long long)duplicate_points.load());
        
        // v44: Disk stats
        u64 dsaved = discarded_saved.load();
        u64 dbytes = discarded_bytes.load();
        if (dsaved > 0) {
            printf("  SavedDPs to disk: %llu (%.2f GB)\n",
                   (unsigned long long)dsaved,
                   (double)dbytes / (1024.0*1024*1024));
        }
        printf("=============================================================\n");
    }
    
    // Legacy interface
    u8* GetBloomPointer() { return nullptr; }
    size_t GetBloomSizeBytes() { return 0; }
    int GetAverageDistanceBits() { return 0; }
    int SampleTameDistances(u8* out, int max) { return 0; }

    // ========================================================================
    // V48 HYBRID: Load pre-computed WILDs into table[1]
    // ========================================================================
    
    // Load wilds from savedps file — NOT SUPPORTED
    // WildEntryCompact only stores 48-bit x_sig, but TableHash needs full 64-bit x.
    // Re-inserting entries would place them at wrong slots, making them unfindable.
    // Use -preloadckpt with a checkpoint file instead (direct slot copy, 100% correct).
    u64 LoadWildsFromDescartados(const char* filename) {
        printf("\n*** ERROR: Loading from savedps file not supported ***\n");
        printf("  WildEntryCompact stores only 48-bit X signature.\n");
        printf("  TableHash needs full 64-bit X — entries would go to wrong slots.\n");
        printf("  Use -preloadckpt with wild_checkpoint.dat instead.\n\n");
        return 0;
    }
    
    // Load wilds from checkpoint file (RCKDT43 format) - loads only table[1] data
    u64 LoadWildsFromCheckpoint(const char* filename) {
        if (!initialized || !wild_table[1]) return 0;
        
        FILE* fp = fopen(filename, "rb");
        if (!fp) {
            printf("WARNING: Cannot open checkpoint file: %s\n", filename);
            return 0;
        }
        
        CheckpointHeader header;
        if (fread(&header, sizeof(header), 1, fp) != 1) { fclose(fp); return false; }
        
        if (memcmp(header.magic, "RCKDT5C", 7) != 0) {
            if (memcmp(header.magic, "RCKDT5B", 7) == 0) {
                header.tables_present = 3;  // v56C always wrote both
            } else {
                printf("WARNING: Invalid checkpoint format (expected RCKDT5C or RCKDT5B)\n");
                fclose(fp);
                return 0;
            }
        }
        
        printf("\n=== HYBRID: Loading WILDs from checkpoint into table[1] ===\n");
        printf("  File: %s\n", filename);
        printf("  Checkpoint has W1: %lluM, W2: %lluM entries\n",
               (unsigned long long)(header.wild_count_w1 / 1000000),
               (unsigned long long)(header.wild_count_w2 / 1000000));
        
        if (header.wild_table_size != wild_table_size) {
            printf("ERROR: Table size mismatch! (file: %llu, current: %llu)\n",
                   (unsigned long long)header.wild_table_size,
                   (unsigned long long)wild_table_size);
            printf("  Use same -ramlimit as original run to match table sizes.\n");
            fclose(fp);
            return 0;
        }
        
        // v56C: Skip table[0] only if it was saved
        if (header.tables_present & 1) {
            fseek(fp, wild_table_size * sizeof(WildEntryCompact), SEEK_CUR);
        }
        
        // Read table[1] — only if it was saved
        if (!(header.tables_present & 2)) {
            printf("WARNING: Checkpoint has no table[1] data (ALL-TAME mode). Nothing to load.\n");
            fclose(fp);
            return 0;
        }
        
        // Read table[1] directly — entries stay in their original hash slots!
        size_t t_read = fread(wild_table[1], sizeof(WildEntryCompact), wild_table_size, fp);
        fclose(fp);
        
        if (t_read != wild_table_size) {
            printf("WARNING: Incomplete read (expected %llu entries)\n",
                   (unsigned long long)wild_table_size);
        }
        
        wild_count[1].store(header.wild_count_w2);
        circular_mode_active[1] = (header.circular_mode_w2 != 0);
        
        double load_pct = (double)header.wild_count_w2 / wild_table_size * 100.0;
        printf("  Direct copy: %lluM wilds loaded into table[1] (%.1f%% load)%s\n",
               (unsigned long long)(header.wild_count_w2 / 1000000), load_pct,
               circular_mode_active[1] ? " (ROTATING)" : "");
        printf("=== HYBRID: Table[1] pre-loaded! ===\n\n");
        
        return header.wild_count_w2;
    }

    // ========================================================================
    // CHECKPOINT SAVE/LOAD (format RCKDT43 - backwards compatible with v43/v44)
    // ========================================================================
    
#pragma pack(push, 1)
    struct CheckpointHeader {
        char magic[8];               // "RCKDT5C"
        u64 version;             
        u64 wild_table_size;         // Per-table size
        u64 wild_count_w1;
        u64 wild_count_w2;
        u64 overwrites_w1;
        u64 overwrites_w2;
        u8  circular_mode_w1;
        u8  circular_mode_w2;
        u8  tables_present;          // v56C: bitmask: bit0=table[0], bit1=table[1]
        u8  reserved[45];        
    };
#pragma pack(pop)
    
    bool SaveCheckpoint(const char* filename) {
        if (!initialized || !wild_table[0]) return false;
        
        printf("\n*** SAVING CHECKPOINT v56C: %s ***\n", filename);
        FILE* fp = fopen(filename, "wb");
        if (!fp) return false;
        
        CheckpointHeader header;
        memset(&header, 0, sizeof(header));
        memcpy(header.magic, "RCKDT5C", 8);
        header.version = 3;
        header.wild_table_size = wild_table_size;
        header.wild_count_w1 = wild_count[0].load();
        header.wild_count_w2 = wild_count[1].load();
        header.overwrites_w1 = table_overwrites[0].load();
        header.overwrites_w2 = table_overwrites[1].load();
        header.circular_mode_w1 = circular_mode_active[0] ? 1 : 0;
        header.circular_mode_w2 = circular_mode_active[1] ? 1 : 0;
        header.tables_present = (wild_table[0] ? 1 : 0) | (wild_table[1] ? 2 : 0);
        
        fwrite(&header, sizeof(header), 1, fp);
        
        // v56C: Only write tables that are allocated (skip NULL → saves ~117GB in ALL-TAME)
        for (int t = 0; t < 2; t++) {
            if (wild_table[t]) {
                fwrite(wild_table[t], sizeof(WildEntryCompact), wild_table_size, fp);
            }
            // NULL tables: simply not written (tables_present bitmask tells loader)
        }
        
        // Save spatial bucket states
        for (u32 i = 0; i < SPATIAL_BUCKETS; i++) {
            u64 wi = spatial_buckets[i].write_index.load();
            u64 fc = spatial_buckets[i].fill_count.load();
            fwrite(&wi, sizeof(u64), 1, fp);
            fwrite(&fc, sizeof(u64), 1, fp);
        }
        
        fclose(fp);
        
        // Show actual size saved
        size_t tables_written = (wild_table[0] ? 1 : 0) + (wild_table[1] ? 1 : 0);
        double gb_saved = (double)(tables_written * wild_table_size * sizeof(WildEntryCompact)) / (1024.0*1024*1024);
        printf("*** CHECKPOINT SAVED (W1/TAMEs: %lluM, W2: %lluM, tables: %zu, %.1f GB) ***\n",
               (unsigned long long)(header.wild_count_w1 / 1000000),
               (unsigned long long)(header.wild_count_w2 / 1000000),
               tables_written, gb_saved);
        return true;
    }
    
    bool LoadCheckpoint(const char* filename) {
        printf("\n*** LOADING CHECKPOINT: %s ***\n", filename);
        FILE* fp = fopen(filename, "rb");
        if (!fp) return false;
        
        CheckpointHeader header;
        if (fread(&header, sizeof(header), 1, fp) != 1) { fclose(fp); return false; }
        
        if (memcmp(header.magic, "RCKDT5C", 7) != 0) {
            // v56C: Backward compat with v56C checkpoints (same data, old header)
            if (memcmp(header.magic, "RCKDT5B", 7) == 0) {
                printf("  (Converting v56B checkpoint → v56C format)\n");
                header.tables_present = 3;  // v56C always wrote both tables
            } else {
                printf("ERROR: Invalid format (expected RCKDT5C or RCKDT5B)\n");
                fclose(fp);
                return false;
            }
        }
        
        if (header.wild_table_size != wild_table_size) {
            printf("ERROR: Table size mismatch (file: %llu, current: %llu)\n",
                   (unsigned long long)header.wild_table_size,
                   (unsigned long long)wild_table_size);
            fclose(fp);
            return false;
        }
        
        // v56C: Only read tables that were saved (tables_present bitmask)
        for (int t = 0; t < 2; t++) {
            bool in_file = (header.tables_present >> t) & 1;
            if (in_file && wild_table[t]) {
                // Table in file AND allocated → read
                fread(wild_table[t], sizeof(WildEntryCompact), wild_table_size, fp);
            } else if (in_file && !wild_table[t]) {
                // Table in file but NOT allocated → skip
                fseek(fp, wild_table_size * sizeof(WildEntryCompact), SEEK_CUR);
            }
            // If not in file: nothing to read, table stays zeroed
        }
        
        // Load spatial bucket states
        for (u32 i = 0; i < SPATIAL_BUCKETS; i++) {
            u64 wi, fc;
            fread(&wi, sizeof(u64), 1, fp);
            fread(&fc, sizeof(u64), 1, fp);
            spatial_buckets[i].write_index.store(wi);
            spatial_buckets[i].fill_count.store(fc);
        }
        
        fclose(fp);
        
        wild_count[0].store(header.wild_count_w1);
        wild_count[1].store(header.wild_count_w2);
        table_overwrites[0].store(header.overwrites_w1);
        table_overwrites[1].store(header.overwrites_w2);
        circular_mode_active[0] = (header.circular_mode_w1 != 0);
        circular_mode_active[1] = (header.circular_mode_w2 != 0);
        
        printf("*** CHECKPOINT LOADED (tables_present: %d) ***\n", header.tables_present);
        printf("    W1/TAMEs: %llu entries%s\n", (unsigned long long)header.wild_count_w1,
               circular_mode_active[0] ? " (ROTATING)" : "");
        if (header.tables_present & 2) {
            printf("    W2: %llu entries%s\n", (unsigned long long)header.wild_count_w2,
                   circular_mode_active[1] ? " (ROTATING)" : "");
        } else {
            printf("    W2: not saved (ALL-TAME mode)\n");
        }
        return true;
    }
};
