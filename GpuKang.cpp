// GpuKang.cpp
// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

#include "GpuKang.h"
#include "TameStore.h"

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
void CallGpuKernelGen(TKparams Kparams);
void CallGpuKernelABC(TKparams Kparams);
void AddPointsToList(u32* data, int cnt, u64 ops_cnt);
extern bool gGenMode; //tames generation mode
extern TameStore gTameStore;  // For resonant WILD generation

int RCGpuKang::CalcKangCnt()
{
	Kparams.BlockCnt = mpCnt;
	Kparams.BlockSize = IsOldGpu ? 512 : 256;
	
	// v45: GroupCnt = PNT_GROUP_CNT (compile-time constant)
	if (IsOldGpu) {
		Kparams.GroupCnt = 64;
	} else {
		Kparams.GroupCnt = V45_PNT_GROUP_CNT;
		if (GroupCnt > 0 && GroupCnt != V45_PNT_GROUP_CNT) {
			printf("Note: -groups %d ignored. Using compiled V45_PNT_GROUP_CNT=%d for max speed.\n", GroupCnt, V45_PNT_GROUP_CNT);
			printf("      To change, edit V45_PNT_GROUP_CNT in defs.h and recompile.\n");
		}
	}
	return Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
}

//executes in main thread
bool RCGpuKang::Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3)
{
	PntToSolve = _PntToSolve;
	Range = _Range;
	DP = _DP;
	EcJumps1 = _EcJumps1;
	EcJumps2 = _EcJumps2;
	EcJumps3 = _EcJumps3;
	StopFlag = false;
	Failed = false;
	WaveNumber = 0;  // Initialize wave counter for smart WILD generation
	UseResonantSpawn = true;  // v34: Enable resonant spawning by default
	RndPnts = nullptr;  // v38: Initialize to nullptr for ALL-WILD mode check
	u64 total_mem = 0;
	memset(dbg, 0, sizeof(dbg));
	memset(SpeedStats, 0, sizeof(SpeedStats));
	cur_stats_ind = 0;

	cudaError_t err;
	err = cudaSetDevice(CudaIndex);
	if (err != cudaSuccess)
		return false;

    Kparams.BlockCnt = mpCnt; // Original

	Kparams.BlockSize = IsOldGpu ? 512 : 256;

	// v45: GroupCnt = V45_PNT_GROUP_CNT for maximum loop unrolling
	Kparams.GroupCnt = IsOldGpu ? 64 : V45_PNT_GROUP_CNT;
	KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
	Kparams.KangCnt = KangCnt;
	Kparams.DP = DP;
	Kparams.KernelA_LDS_Size = 64 * JMP_CNT + 16 * Kparams.BlockSize;
	Kparams.KernelB_LDS_Size = 64 * JMP_CNT;
	Kparams.KernelC_LDS_Size = 96 * JMP_CNT;
	Kparams.IsGenMode = gGenMode;
	Kparams.AllWildsMode = false;  // Initialize to false, set true during HUNT
	
	
	

//allocate gpu mem
	u64 size;
	if (!IsOldGpu)
	{
		//L2	
		int L2size = Kparams.KangCnt * (3 * 32);
		total_mem += L2size;
		err = cudaMalloc((void**)&Kparams.L2, L2size);
		if (err != cudaSuccess)
		{
			printf("GPU %d, Allocate L2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
			return false;
		}
		size = L2size;
		if (size > persistingL2CacheMaxSize)
			size = persistingL2CacheMaxSize;
		err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); // set max allowed size for L2
		//persisting for L2
		cudaStreamAttrValue stream_attribute;                                                   
		stream_attribute.accessPolicyWindow.base_ptr = Kparams.L2;
		stream_attribute.accessPolicyWindow.num_bytes = size;										
		stream_attribute.accessPolicyWindow.hitRatio = 1.0;                                     
		stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;             
		stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  	
		err = cudaStreamSetAttribute(NULL, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
		if (err != cudaSuccess)
		{
			printf("GPU %d, cudaStreamSetAttribute failed: %s\n", CudaIndex, cudaGetErrorString(err));
			return false;
		}
	}
	size = MAX_DP_CNT * GPU_DP_SIZE + 16;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.DPs_out, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate GpuOut memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = KangCnt * 96;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.Kangs, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate pKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps1, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps2, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps3 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = 2 * (u64)KangCnt * STEP_CNT;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.JumpsList, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate JumpsList memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32)); //we store 16bytes of X
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.DPTable, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate DPTable memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = mpCnt * Kparams.BlockSize * sizeof(u64);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.L1S2, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate L1S2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * (2 * 32);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LastPnts, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * sizeof(u64);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LoopTable, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += 1024;
	err = cudaMalloc((void**)&Kparams.dbg_buf, 1024);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate dbg_buf memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = sizeof(u32) * KangCnt + 8;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LoopedKangs, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LoopedKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	DPs_out = (u32*)malloc(MAX_DP_CNT * GPU_DP_SIZE);

//jmp1
	u64* buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps1[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps1[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps1[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps1, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps1 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);
//jmp2
	buf = (u64*)malloc(JMP_CNT * 96);
	u64* jmp2_table = (u64*)malloc(JMP_CNT * 64);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps2[i].p.x.data, 32);
		memcpy(jmp2_table + i * 8, EcJumps2[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(jmp2_table + i * 8 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps2[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps2, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps2 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);

	err = cuSetGpuParams(Kparams, jmp2_table);
	if (err != cudaSuccess)
	{
		free(jmp2_table);
		printf("GPU %d, cuSetGpuParams failed: %s!\r\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(jmp2_table);
//jmp3
	buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps3[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps3[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps3[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps3, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps3 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);

	printf("GPU %d: allocated %llu MB, %d kangaroos. OldGpuMode: %s\r\n", CudaIndex, total_mem / (1024 * 1024), KangCnt, IsOldGpu ? "Yes" : "No");
	return true;
}

void RCGpuKang::Release()
{
	
	free(RndPnts);
	free(DPs_out);
	cudaFree(Kparams.LoopedKangs);
	cudaFree(Kparams.dbg_buf);
	cudaFree(Kparams.LoopTable);
	cudaFree(Kparams.LastPnts);
	cudaFree(Kparams.L1S2);
	cudaFree(Kparams.DPTable);
	cudaFree(Kparams.JumpsList);
	cudaFree(Kparams.Jumps3);
	cudaFree(Kparams.Jumps2);
	cudaFree(Kparams.Jumps1);
	cudaFree(Kparams.Kangs);
	cudaFree(Kparams.DPs_out);
	if (!IsOldGpu)
		cudaFree(Kparams.L2);
}

void RCGpuKang::Stop()
{
	StopFlag = true;
}

void RCGpuKang::GenerateRndDistances()
{
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		if (i < KangCnt / 3)
			d.RndBits(Range - 4); //TAME kangs
		else
		{
			d.RndBits(Range - 1);
			d.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		}
		memcpy(RndPnts[i].priv, d.data, 24);
	}
}

// Generate new random WILD positions for a new wave
// Uses fresh random positions each wave (proven to work!)
void RCGpuKang::GenerateSmartWildDistances()
{
	// During HUNT phase, ALL kangaroos are WILDs
	// Generate random distances in WILD range (Range-1 bits)
	// FIXED v37: Previously 1/3 were incorrectly using Range-4 (TAME range)
	
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		// ALL WILDs - use Range-1 bits for full coverage
		d.RndBits(Range - 1);
		d.data[0] &= 0xFFFFFFFFFFFFFFFE; // Must be even
		memcpy(RndPnts[i].priv, d.data, 24);
	}
}

// ============================================================================
// RESONANT WILD GENERATION - v35 Experimental
// Uses TAME distance statistics to spawn WILDs in "resonant" positions
// ============================================================================

// ============================================================================
// RESSONÂNCIA MULTI-BANDA V35 (CORRIGIDA & VACINADA)
// Estratégia Otimizada para Ranges Gigantes (ex: 135 bits)
// ============================================================================

void RCGpuKang::GenerateResonantWildDistances()
{
	u64 tame_count = gTameStore.GetTameCount();
	
	// 1. Safety check: if insufficient data, use default random.
	if (tame_count < 1000) {
		GenerateSmartWildDistances();
		return;
	}
	
	// 2. Coleta de Amostras Reais dos TAMEs
	const int NUM_SAMPLES = 256;
	u8 tame_samples[NUM_SAMPLES * 24];
	int num_samples = gTameStore.SampleTameDistances(tame_samples, NUM_SAMPLES);
	
	if (num_samples == 0) {
		GenerateSmartWildDistances();
		return;
	}

	// 3. Análise da Profundidade Média
	int avg_bits = gTameStore.GetAverageDistanceBits();
	
	// Se a média for inconsistente com o Range, ajusta ou reseta
	if (avg_bits < 10) avg_bits = 10;
	if (avg_bits > Range) avg_bits = Range - 1;

	printf("Multi-Band Resonance: Avg TAME depth ~2^%d bits. Deploying tactics with Anti-Clone Vaccine...\n", avg_bits);

	// 4. Geração dos Cangurus
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		
		// Escolhe uma amostra de TAME aleatória como base
		int sample_idx = rand() % num_samples;
		u8* ref_dist = tame_samples + sample_idx * 24;
		
		// Carrega a distância base (copia 24 bytes)
		memcpy(d.data, ref_dist, 24);
		
		// Limpeza de bits superiores (segurança para não exceder a memória da estrutura)
		int words = (Range + 63) / 64;
		for(int k=words; k<4; k++) d.data[k] = 0;

		// --- ESTRATÉGIA MULTI-BANDA (Tactical Deployment) ---
		// Divide os cangurus em grupos para cobrir diferentes áreas de busca
		
		int tactic = i % 10; // Divide em 10 fatias (0 a 9)
		
		if (tactic < 2) { 
			// TÁTICA 1: "Curto Alcance" (20% dos cangurus)
			// Procura chaves que estão "antes" da média dos TAMEs.
			// Divide a distância por 2, 4, 8 ou 16 (Shift Right)
			int shift = 1 + (rand() % 4);
			d.ShiftRight(shift);
		}
		else if (tactic < 8) {
			// TÁTICA 2: "Ressonância Padrão" (60% dos cangurus)
			// Procura na mesma região dos TAMEs, com variação (Jitter).
			
			EcInt jitter;
			// O Jitter deve ser proporcional ao tamanho do Range para ser útil
			// Para Range 135, um jitter de ~120 bits é bom.
			int jitter_bits = (avg_bits > 10) ? avg_bits - 4 : 5;
			jitter.RndBits(jitter_bits); 
			
			// Aplica o ruído (soma ou subtrai)
			if (rand() & 1) d.Add(jitter);
			else {
				if (jitter.IsLessThanU(d)) d.Sub(jitter);
				else d.Add(jitter); // Se não der pra subtrair, soma
			}
		}
		else {
			// TÁTICA 3: "Deep Dive" / Mergulho Profundo (20% dos cangurus)
			// Assume que a chave está MUITO além de onde os TAMEs pararam.
			// Multiplica a distância ou adiciona um offset gigante.
			
			// Tenta multiplicar a distância (Shift Left)
			int shift = 1 + (rand() % 3); // x2, x4, ou x8
			
			// Verifica se o shift não vai estourar o Range total
			EcInt limit_check = d;
			limit_check.ShiftLeft(shift);
			
			EcInt max_range_check;
			max_range_check.Set(1);
			max_range_check.ShiftLeft(Range - 1);
			
			if (limit_check.IsLessThanU(max_range_check)) {
				d.ShiftLeft(shift);
			} else {
				// Se shift for estourar, apenas adiciona um jitter grande
				EcInt deep_jitter;
				deep_jitter.RndBits(avg_bits);
				d.Add(deep_jitter);
			}
		}

		// --- VACINA ANTI-CLONE & FINALIZAÇÃO ---
		
		// 1. A VACINA: Força a inversão do bit 12.
		// This mathematically ensures Dist_Wild != original Dist_Tame.
		// Valor 4096 (2^12) é pequeno o suficiente para não estragar a ressonância,
		// mas grande o suficiente para ser um ponto distinto.
		d.data[0] ^= (1ULL << (DP + 4));

		// 2. Garante paridade (Obrigatório para o algoritmo Kangaroo: distância deve ser par)
		d.data[0] &= 0xFFFFFFFFFFFFFFFE;
		
		// 3. Bounds check (Wrap Around)
		// Se a distância gerada for maior que o Range do puzzle, fazemos o "módulo"
		// ou regeramos aleatoriamente para não desperdiçar o canguru.
		EcInt max_dist;
		max_dist.Set(1);
		max_dist.ShiftLeft(Range); 
		
		if (max_dist.IsLessThanU(d) || d.IsEqual(max_dist)) {
			// Fallback seguro: Gera aleatório puro dentro do range
			d.RndBits(Range - 1);
			d.data[0] &= 0xFFFFFFFFFFFFFFFE;
		}
		
		// Copia para o buffer que será enviado à GPU
		memcpy(RndPnts[i].priv, d.data, 24);
	}
}

bool RCGpuKang::Start()
{
	if (Failed)
		return false;

	cudaError_t err;
	err = cudaSetDevice(CudaIndex);
	if (err != cudaSuccess)
		return false;

	HalfRange.Set(1);
	HalfRange.ShiftLeft(Range - 1);
	PntHalfRange = ec.MultiplyG(HalfRange);
	NegPntHalfRange = PntHalfRange;
	NegPntHalfRange.y.NegModP();

	PntA = ec.AddPoints(PntToSolve, NegPntHalfRange);
	PntB = PntA;
	PntB.y.NegModP();

	RndPnts = (TPointPriv*)malloc(KangCnt * 96);
	GenerateRndDistances();
/* 
	//we can calc start points on CPU
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		memcpy(d.data, RndPnts[i].priv, 24);
		d.data[3] = 0;
		d.data[4] = 0;
		EcPoint p = ec.MultiplyG(d);
		memcpy(RndPnts[i].x, p.x.data, 32);
		memcpy(RndPnts[i].y, p.y.data, 32);
	}
	for (int i = KangCnt / 3; i < 2 * KangCnt / 3; i++)
	{
		EcPoint p;
		p.LoadFromBuffer64((u8*)RndPnts[i].x);
		p = ec.AddPoints(p, PntA);
		p.SaveToBuffer64((u8*)RndPnts[i].x);
	}
	for (int i = 2 * KangCnt / 3; i < KangCnt; i++)
	{
		EcPoint p;
		p.LoadFromBuffer64((u8*)RndPnts[i].x);
		p = ec.AddPoints(p, PntB);
		p.SaveToBuffer64((u8*)RndPnts[i].x);
	}
	//copy to gpu
	err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
/**/
	//but it's faster to calc then on GPU
	u8 buf_PntA[64], buf_PntB[64];
	PntA.SaveToBuffer64(buf_PntA);
	PntB.SaveToBuffer64(buf_PntB);
	for (int i = 0; i < KangCnt; i++)
	{
		if (i < KangCnt / 3)
			memset(RndPnts[i].x, 0, 64);
		else
			if (i < 2 * KangCnt / 3)
				memcpy(RndPnts[i].x, buf_PntA, 64);
			else
				memcpy(RndPnts[i].x, buf_PntB, 64);
	}
	//copy to gpu
	err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	CallGpuKernelGen(Kparams);

	err = cudaMemset(Kparams.L1S2, 0, mpCnt * Kparams.BlockSize * 8);
	if (err != cudaSuccess)
		return false;
	cudaMemset(Kparams.dbg_buf, 0, 1024);
	cudaMemset(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64));
	return true;
}

#ifdef DEBUG_MODE
int RCGpuKang::Dbg_CheckKangs()
{
	int kang_size = mpCnt * Kparams.BlockSize * Kparams.GroupCnt * 96;
	u64* kangs = (u64*)malloc(kang_size);
	cudaError_t err = cudaMemcpy(kangs, Kparams.Kangs, kang_size, cudaMemcpyDeviceToHost);
	int res = 0;
	for (int i = 0; i < KangCnt; i++)
	{
		EcPoint Pnt, p;
		Pnt.LoadFromBuffer64((u8*)&kangs[i * 12 + 0]);
		EcInt dist;
		dist.Set(0);
		memcpy(dist.data, &kangs[i * 12 + 8], 24);
		bool neg = false;
		if (dist.data[2] >> 63)
		{
			neg = true;
			memset(((u8*)dist.data) + 24, 0xFF, 16);
			dist.Neg();
		}
		p = ec.MultiplyG_Fast(dist);
		if (neg)
			p.y.NegModP();
		if (i < KangCnt / 3)
			p = p;
		else
			if (i < 2 * KangCnt / 3)
				p = ec.AddPoints(PntA, p);
			else
				p = ec.AddPoints(PntB, p);
		if (!p.IsEqual(Pnt))
			res++;
	}
	free(kangs);
	return res;
}
#endif

extern u32 gTotalErrors;

//executes in separate thread
void RCGpuKang::Execute()
{
	cudaSetDevice(CudaIndex);

	if (!Start())
	{
		gTotalErrors++;
		return;
	}
#ifdef DEBUG_MODE
	u64 iter = 1;
#endif
	cudaError_t err;	
	while (!StopFlag)
	{
		u64 t1 = GetTickCount64();
		cudaMemset(Kparams.DPs_out, 0, 4);
		cudaMemset(Kparams.DPTable, 0, KangCnt * sizeof(u32));
		cudaMemset(Kparams.LoopedKangs, 0, 8);
		CallGpuKernelABC(Kparams);
		int cnt;
		err = cudaMemcpy(&cnt, Kparams.DPs_out, 4, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			printf("GPU %d, CallGpuKernel failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
			gTotalErrors++;
			break;
		}
		
		if (cnt >= MAX_DP_CNT)
		{
			cnt = MAX_DP_CNT;
			printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", CudaIndex);
		}
		u64 pnt_cnt = (u64)KangCnt * STEP_CNT;

		if (cnt)
		{
			err = cudaMemcpy(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess)
			{
				gTotalErrors++;
				break;
			}
			AddPointsToList(DPs_out, cnt, (u64)KangCnt * STEP_CNT);
		}

		//dbg
		cudaMemcpy(dbg, Kparams.dbg_buf, 1024, cudaMemcpyDeviceToHost);

		u32 lcnt;
		cudaMemcpy(&lcnt, Kparams.LoopedKangs, 4, cudaMemcpyDeviceToHost);
		//printf("GPU %d, Looped: %d\r\n", CudaIndex, lcnt);

		u64 t2 = GetTickCount64();
		u64 tm = t2 - t1;
		if (!tm)
			tm = 1;
		int cur_speed = (int)(pnt_cnt / (tm * 1000));
		//printf("GPU %d kernel time %d ms, speed %d MH\r\n", CudaIndex, (int)tm, cur_speed);

		SpeedStats[cur_stats_ind] = cur_speed;
		cur_stats_ind = (cur_stats_ind + 1) % STATS_WND_SIZE;

#ifdef DEBUG_MODE
		if ((iter % 300) == 0)
		{
			int corr_cnt = Dbg_CheckKangs();
			if (corr_cnt)
			{
				printf("DBG: GPU %d, KANGS CORRUPTED: %d\r\n", CudaIndex, corr_cnt);
				gTotalErrors++;
			}
			else
				printf("DBG: GPU %d, ALL KANGS OK!\r\n", CudaIndex);
		}
		iter++;
#endif
	}

	Release();
}

int RCGpuKang::GetStatsSpeed()
{
	int res = SpeedStats[0];
	for (int i = 1; i < STATS_WND_SIZE; i++)
		res += SpeedStats[i];
	return res / STATS_WND_SIZE;
}

void RCGpuKang::SetGenMode(bool mode)
{
	Kparams.IsGenMode = mode;
}

bool RCGpuKang::ReinitForHunt()
{
	// Reinitialize kangaroos for HUNT phase with ALL WILDs optimization
	// 100% of kangaroos will be WILDs (50% more effective than 66%)
	
	cudaError_t err;
	err = cudaSetDevice(CudaIndex);
	if (err != cudaSuccess)
		return false;
	
	// v38: If called before Start() (ALL-WILD mode), initialize required variables
	if (RndPnts == nullptr) {
		// Allocate host memory for kangaroo data
		RndPnts = (TPointPriv*)malloc(KangCnt * 96);
		if (!RndPnts) {
			printf("GPU %d: Failed to allocate RndPnts in ReinitForHunt\n", CudaIndex);
			return false;
		}
		
		// Initialize PntA and PntB (WILD start points)
		HalfRange.Set(1);
		HalfRange.ShiftLeft(Range - 1);
		PntHalfRange = ec.MultiplyG(HalfRange);
		NegPntHalfRange = PntHalfRange;
		NegPntHalfRange.y.NegModP();
		
		PntA = ec.AddPoints(PntToSolve, NegPntHalfRange);  // PubKey - HalfRange*G
		PntB = PntA;
		PntB.y.NegModP();  // PubKey + HalfRange*G
	}
	
	// Set mode to HUNT with ALL WILDs
	Kparams.IsGenMode = false;
	Kparams.AllWildsMode = true;  // CRITICAL: Enable 100% WILDs mode!
	
	// Increment wave number for tracking
	WaveNumber++;
	
	// Generate distances for ALL kangaroos
	// v34: Use resonant spawning if enabled
	if (UseResonantSpawn) {
		GenerateResonantWildDistances();
	} else {
		GenerateSmartWildDistances();
	}
	
	// Prepare PntA/PntB offset for ALL kangaroos (not just 2/3)
	u8 buf_PntA[64], buf_PntB[64];
	PntA.SaveToBuffer64(buf_PntA);
	PntB.SaveToBuffer64(buf_PntB);
	
	// ALL kangaroos get PntA or PntB based on parity
	// v52 FIX: Must match GPU's BuildDP: kang_type = 1 + (kang_ind & 1)
	//   GPU: Even → WILD1 (type=1), Odd → WILD2 (type=2)
	//   CPU: Even → PntA (WILD1 start), Odd → PntB (WILD2 start)
	for (int i = 0; i < KangCnt; i++)
	{
		if (i & 1)
			memcpy(RndPnts[i].x, buf_PntB, 64);  // Odd indices: WILD2 (PntB)
		else
			memcpy(RndPnts[i].x, buf_PntA, 64);  // Even indices: WILD1 (PntA)
	}
	
	// Copy to GPU
	err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, ReinitForHunt cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	
	// Regenerate kangaroos - ALL will have PubKey offset now!
	CallGpuKernelGen(Kparams);
	
	// Reset DP table and loop detection
	cudaMemset(Kparams.DPTable, 0, KangCnt * sizeof(u32));
	cudaMemset(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64));
	cudaMemset(Kparams.LoopedKangs, 0, 8);
	
	return true;
}


