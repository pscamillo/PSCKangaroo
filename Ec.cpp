// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC
// Modified with Endomorphism support

#ifdef _MSC_VER
#include <intrin.h>
// MSVC doesn't support unsigned __int128. Emulate with struct + intrinsics.
// Note: uses unsigned __int64 directly since u64 typedef is not yet available.
struct uint128_t {
    unsigned __int64 lo, hi;
    uint128_t() : lo(0), hi(0) {}
    uint128_t(unsigned __int64 v) : lo(v), hi(0) {}
    static uint128_t mul(unsigned __int64 a, unsigned __int64 b) {
        uint128_t r;
        r.lo = _umul128(a, b, &r.hi);
        return r;
    }
    uint128_t operator+(unsigned __int64 v) const {
        uint128_t r;
        r.lo = lo + v;
        r.hi = hi + (r.lo < lo ? 1 : 0);
        return r;
    }
    uint128_t operator+(const uint128_t& o) const {
        uint128_t r;
        r.lo = lo + o.lo;
        r.hi = hi + o.hi + (r.lo < lo ? 1 : 0);
        return r;
    }
    uint128_t& operator+=(unsigned __int64 v) { *this = *this + v; return *this; }
    uint128_t& operator+=(const uint128_t& o) { *this = *this + o; return *this; }
    explicit operator unsigned __int64() const { return lo; }
    unsigned __int64 shr64() const { return hi; }
};
#define UINT128_MUL(a, b) uint128_t::mul((unsigned __int64)(a), (unsigned __int64)(b))
#define UINT128_SHR64(x) ((x).shr64())
#define UINT128_LO(x) ((unsigned __int64)(x))
#else
// GCC/Clang: native unsigned __int128
typedef unsigned __int128 uint128_t;
#define UINT128_MUL(a, b) ((uint128_t)(u64)(a) * (u64)(b))
#define UINT128_SHR64(x) ((u64)((x) >> 64))
#define UINT128_LO(x) ((u64)(x))
#endif

#include "defs.h"
#include "Ec.h"
#include <random>
#include "utils.h"

// https://en.bitcoin.it/wiki/Secp256k1
EcInt g_P; //FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
EcPoint g_G; //Generator point

// Curve order n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
EcInt g_N;

// Endomorphism constants (λ³ ≡ 1 mod n)
// LAMBDA  = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
// LAMBDA² = 0xAC9C52B33FA3CF1F5AD9E3FD77ED9BA4A880B9FC8EC739C2E0CFC810B51283CE
// (λ² + λ + 1 ≡ 0 mod n, so λ² = n - λ - 1)
EcInt g_Lambda;
EcInt g_Lambda2;

#define P_REV	0x00000001000003D1

#ifdef DEBUG_MODE
u8* GTable = NULL; //16x16-bit table
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool parse_u8(const char* s, u8* res)
{
	char cl = toupper(s[1]);
	char ch = toupper(s[0]);
	if (((cl < '0') || (cl > '9')) && ((cl < 'A') || (cl > 'F')))
		return false;
	if (((ch < '0') || (ch > '9')) && ((ch < 'A') || (ch > 'F')))
		return false;
	u8 l = ((cl >= '0') && (cl <= '9')) ? (cl - '0') : (cl - 'A' + 10);
	u8 h = ((ch >= '0') && (ch <= '9')) ? (ch - '0') : (ch - 'A' + 10);
	*res = l + (h << 4);
	return true;
}

bool EcPoint::IsEqual(EcPoint& pnt)
{
	return this->x.IsEqual(pnt.x) && this->y.IsEqual(pnt.y);
}

void EcPoint::LoadFromBuffer64(u8* buffer)
{
	memcpy(x.data, buffer, 32);
	x.data[4] = 0;
	memcpy(y.data, buffer + 32, 32);
	y.data[4] = 0;
}

void EcPoint::SaveToBuffer64(u8* buffer)
{
	memcpy(buffer, x.data, 32);
	memcpy(buffer + 32, y.data, 32);
}

bool EcPoint::SetHexStr(const char* str)
{
	EcPoint res;
	int len = (int)strlen(str);
	if (len < 66)
		return false;
	u8 type, b;
	if (!parse_u8(str, &type))
		return false;
	if ((type < 2) || (type > 4))
		return false;
	if (((type == 2) || (type == 3)) && (len != 66))
		return false;
	if ((type == 4) && (len != 130))
		return false;

	if (len == 66) //compressed
	{
		str += 2;
		for (int i = 0; i < 32; i++)
		{
			if (!parse_u8(str + 2 * i, &b))
				return false;
			((u8*)res.x.data)[31 - i] = b;
		}
		res.y = Ec::CalcY(res.x, type == 2);
		if (!Ec::IsValidPoint(res))
			return false;		
		*this = res;
		return true;
	}
	//uncompressed
	str += 2;
	for (int i = 0; i < 32; i++)
	{
		if (!parse_u8(str + 2 * i, &b))
			return false;
		((u8*)res.x.data)[31 - i] = b;

		if (!parse_u8(str + 2 * i + 64, &b))
			return false;
		((u8*)res.y.data)[31 - i] = b;
	}
	if (!Ec::IsValidPoint(res))
		return false;
	*this = res;
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// https://en.bitcoin.it/wiki/Secp256k1
void InitEc()
{
	g_P.SetHexStr("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"); //Fp
	g_G.x.SetHexStr("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"); //G.x
	g_G.y.SetHexStr("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"); //G.y
	
	// Initialize curve order n and endomorphism constants
	g_N.SetHexStr("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
	// v55 FIX: Lambda was wrong! Must be cube root of 1 mod n
	g_Lambda.SetHexStr("5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72");
	// Lambda² = n - Lambda - 1 (since λ² + λ + 1 ≡ 0 mod n)
	g_Lambda2.SetHexStr("AC9C52B33FA3CF1F5AD9E3FD77ED9BA4A880B9FC8EC739C2E0CFC810B51283CE");
	
#ifdef DEBUG_MODE
	GTable = (u8*)malloc(16 * 256 * 256 * 64);
	EcPoint pnt = g_G;
	for (int i = 0; i < 16; i++)
	{
		pnt.SaveToBuffer64(GTable + (i * 256 * 256) * 64);
		EcPoint tmp = pnt;
		pnt = Ec::DoublePoint(pnt);
		for (int j = 1; j < 256 * 256 - 1; j++)
		{
			pnt.SaveToBuffer64(GTable + (i * 256 * 256 + j) * 64);
			pnt = Ec::AddPoints(pnt, tmp);
		}
	}
#endif
};

void DeInitEc()
{
#ifdef DEBUG_MODE
	if (GTable)
		free(GTable);
#endif
}

// https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication#Point_addition
EcPoint Ec::AddPoints(EcPoint& pnt1, EcPoint& pnt2)
{
	EcPoint res;
	EcInt dx, dy, lambda, lambda2;

	dx = pnt2.x;
	dx.SubModP(pnt1.x);
	dx.InvModP();

	dy = pnt2.y;
	dy.SubModP(pnt1.y);

	lambda = dy;
	lambda.MulModP(dx);
	lambda2 = lambda;
	lambda2.MulModP(lambda);

	res.x = lambda2;
	res.x.SubModP(pnt1.x);
	res.x.SubModP(pnt2.x);

	res.y = pnt2.x;
	res.y.SubModP(res.x);
	res.y.MulModP(lambda);
	res.y.SubModP(pnt2.y);
	return res;
}

// https://en.wikipedia.org/wiki/Elliptic_curve_point_multiplication#Point_doubling
EcPoint Ec::DoublePoint(EcPoint& pnt)
{
	EcPoint res;
	EcInt t1, t2, lambda, lambda2;

	t1 = pnt.y;
	t1.AddModP(pnt.y);
	t1.InvModP();

	t2 = pnt.x;
	t2.MulModP(pnt.x);
	lambda = t2;
	lambda.AddModP(t2);
	lambda.AddModP(t2);
	lambda.MulModP(t1);
	lambda2 = lambda;
	lambda2.MulModP(lambda);

	res.x = lambda2;
	res.x.SubModP(pnt.x);
	res.x.SubModP(pnt.x);

	res.y = pnt.x;
	res.y.SubModP(res.x);
	res.y.MulModP(lambda);
	res.y.SubModP(pnt.y);
	return res;
}

//k up to 256 bits
EcPoint Ec::MultiplyG(EcInt& k)
{
	EcPoint res;
	EcPoint t = g_G;
	bool first = true;
	int n = 3;
	while ((n >= 0) && !k.data[n])
		n--;
	if (n < 0)
		return res; //error
	int index;                     
	_BitScanReverse64((DWORD*)&index, k.data[n]);
	for (int i = 0; i <= 64 * n + index; i++)
	{
		u8 v = (k.data[i / 64] >> (i % 64)) & 1;	
		if (v)
		{
			if (first)
			{
				first = false;
				res = t;
			}
			else
				res = Ec::AddPoints(res, t);
		}
		t = Ec::DoublePoint(t);
	}
	return res;
}

#ifdef DEBUG_MODE
//uses gTable (16x16-bit) to speedup calculation
EcPoint Ec::MultiplyG_Fast(EcInt& k)
{
	int i;
	u16 b;
	EcPoint pnt, res;
	for (i = 0; i < 16; i++)
	{
		b = k.GetU16(i);
		if (b)
			break;
	}
	if (i >= 16)
		return pnt;
	if (i < 16)
	{
		res.LoadFromBuffer64(GTable + (256 * 256 * i + (b - 1)) * 64);
		i++;
	}
	while (i < 16)
	{
		b = k.GetU16(i);
		if (b)
		{
			pnt.LoadFromBuffer64(GTable + (256 * 256 * i + (b - 1)) * 64);
			res = AddPoints(res, pnt);
		}
		i++;
	}
	return res;
}
#endif

EcInt Ec::CalcY(EcInt& x, bool is_even)
{
	EcInt res;
	EcInt tmp;
	tmp.Set(7);
	res = x;
	res.MulModP(x);
	res.MulModP(x);
	res.AddModP(tmp);
	res.SqrtModP();
	if ((res.data[0] & 1) == is_even)
		res.NegModP();
	return res;
}

bool Ec::IsValidPoint(EcPoint& pnt)
{
	EcInt x, y, seven;
	seven.Set(7);
	x = pnt.x;
	x.MulModP(pnt.x);
	x.MulModP(pnt.x);
	x.AddModP(seven);
	y = pnt.y;
	y.MulModP(pnt.y);
	return x.IsEqual(y);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Mul256_by_64(u64* input, u64 multiplier, u64* result)
{
	u64 h1, h2;
	result[0] = _umul128(input[0], multiplier, &h1);
	u8 carry = _addcarry_u64(0, _umul128(input[1], multiplier, &h2), h1, result + 1);
	carry = _addcarry_u64(carry, _umul128(input[2], multiplier, &h1), h2, result + 2);
	carry = _addcarry_u64(carry, _umul128(input[3], multiplier, &h2), h1, result + 3);
	_addcarry_u64(carry, 0, h2, result + 4);
}

void Mul320_by_64(u64* input, u64 multiplier, u64* result)
{
	u64 h1, h2;
	result[0] = _umul128(input[0], multiplier, &h1);
	u8 carry = _addcarry_u64(0, _umul128(input[1], multiplier, &h2), h1, result + 1);
	carry = _addcarry_u64(carry, _umul128(input[2], multiplier, &h1), h2, result + 2);
	carry = _addcarry_u64(carry, _umul128(input[3], multiplier, &h2), h1, result + 3);
	_addcarry_u64(carry, _umul128(input[4], multiplier, &h1), h2, result + 4);
}

void Add320_to_256(u64* in_out, u64* val)
{
	u8 c = _addcarry_u64(0, in_out[0], val[0], in_out);
	c = _addcarry_u64(c, in_out[1], val[1], in_out + 1);
	c = _addcarry_u64(c, in_out[2], val[2], in_out + 2);
	c = _addcarry_u64(c, in_out[3], val[3], in_out + 3);
	_addcarry_u64(c, 0, val[4], in_out + 4);
}

EcInt::EcInt()
{
	SetZero();
}

void EcInt::Assign(EcInt& val)
{
	memcpy(data, val.data, sizeof(data));
}

void EcInt::Set(u64 val)
{
	SetZero();
	data[0] = val;
}

void EcInt::SetZero()
{
	memset(data, 0, sizeof(data));
}

bool EcInt::SetHexStr(const char* str)
{
	SetZero();
	int len = (int)strlen(str);
	if (len > 64)
		return false;
	char s[64];
	memset(s, '0', 64);
	memcpy(s + 64 - len, str, len);
	for (int i = 0; i < 32; i++)
	{
		int n = 62 - 2 * i;
		u8 b;
		if (!parse_u8(s + n, &b))
			return false;
		((u8*)data)[i] = b;
	}
	return true;
}

void EcInt::GetHexStr(char* str)
{
	for (int i = 0; i < 32; i++)
		sprintf(str + 2 * i, "%02X", ((u8*)data)[31 - i]);
	str[64] = 0;
}

u16 EcInt::GetU16(int index)
{
	return (u16)(data[index / 4] >> (16 * (index % 4)));
}

//returns carry
bool EcInt::Add(EcInt& val)
{
	u8 c = _addcarry_u64(0, data[0], val.data[0], data + 0);
	c = _addcarry_u64(c, data[1], val.data[1], data + 1);
	c = _addcarry_u64(c, data[2], val.data[2], data + 2);
	c = _addcarry_u64(c, data[3], val.data[3], data + 3);
	return _addcarry_u64(c, data[4], val.data[4], data + 4) != 0;
}

//returns carry
bool EcInt::Sub(EcInt& val)
{
	u8 c = _subborrow_u64(0, data[0], val.data[0], data + 0);
	c = _subborrow_u64(c, data[1], val.data[1], data + 1);
	c = _subborrow_u64(c, data[2], val.data[2], data + 2);
	c = _subborrow_u64(c, data[3], val.data[3], data + 3);
	return _subborrow_u64(c, data[4], val.data[4], data + 4) != 0;
}

void EcInt::Neg()
{
	u8 c = _subborrow_u64(0, 0, data[0], data + 0);
	c = _subborrow_u64(c, 0, data[1], data + 1);
	c = _subborrow_u64(c, 0, data[2], data + 2);
	c = _subborrow_u64(c, 0, data[3], data + 3);
	_subborrow_u64(c, 0, data[4], data + 4);
}

void EcInt::Neg256()
{
	u8 c = _subborrow_u64(0, 0, data[0], data + 0);
	c = _subborrow_u64(c, 0, data[1], data + 1);
	c = _subborrow_u64(c, 0, data[2], data + 2);
	c = _subborrow_u64(c, 0, data[3], data + 3);
	data[4] = 0;
}

bool EcInt::IsLessThanU(EcInt& val)
{
	int i = 4;
	while (i >= 0)
	{
		if (data[i] != val.data[i])
			break;
		i--;
	}
	if (i < 0)
		return false;
	return data[i] < val.data[i];
}

bool EcInt::IsLessThanI(EcInt& val)
{
	if ((data[4] >> 63) && !(val.data[4] >> 63))
		return true;
	if (!(data[4] >> 63) && (val.data[4] >> 63))
		return false;

	int i = 4;
	while (i >= 0)
	{
		if (data[i] != val.data[i])
			break;
		i--;
	}
	if (i < 0)
		return false;
	return data[i] < val.data[i];
}

bool EcInt::IsEqual(EcInt& val)
{
	return memcmp(val.data, this->data, 40) == 0;
}

bool EcInt::IsZero()
{
	return ((data[0] == 0) && (data[1] == 0) && (data[2] == 0) && (data[3] == 0) && (data[4] == 0));
}

void EcInt::AddModP(EcInt& val)
{
	Add(val);
	if (!IsLessThanU(g_P)) 
		Sub(g_P);
}

void EcInt::SubModP(EcInt& val)
{
	if (Sub(val))
		Add(g_P);
}

//assume value < P
void EcInt::NegModP()
{
	Neg();
	Add(g_P);
}

void EcInt::ShiftRight(int nbits)
{
	int offset = nbits / 64;
	if (offset)
	{
		for (int i = 0; i < 5 - offset; i++)
			data[i] = data[i + offset];
		for (int i = 5 - offset; i < 5; i++)
			data[i] = 0;
		nbits -= 64 * offset;
	}
	data[0] = __shiftright128(data[0], data[1], nbits);
	data[1] = __shiftright128(data[1], data[2], nbits);
	data[2] = __shiftright128(data[2], data[3], nbits);
	data[3] = __shiftright128(data[3], data[4], nbits);
	data[4] = ((i64)data[4]) >> nbits;
}

void EcInt::ShiftLeft(int nbits)
{
	int offset = nbits / 64;
	if (offset)
	{
		for (int i = 4; i >= offset; i--)
			data[i] = data[i - offset];
		for (int i = offset - 1; i >= 0; i--)
			data[i] = 0;
		nbits -= 64 * offset;
	}
	data[4] = __shiftleft128(data[3], data[4], nbits);
	data[3] = __shiftleft128(data[2], data[3], nbits);
	data[2] = __shiftleft128(data[1], data[2], nbits);
	data[1] = __shiftleft128(data[0], data[1], nbits);
	data[0] = data[0] << nbits;
}

void EcInt::MulModP(EcInt& val)
{	
	u64 buff[8], tmp[5], h;
	//calc 512 bits
	Mul256_by_64(val.data, data[0], buff);
	Mul256_by_64(val.data, data[1], tmp);
	Add320_to_256(buff + 1, tmp);
	Mul256_by_64(val.data, data[2], tmp);
	Add320_to_256(buff + 2, tmp);
	Mul256_by_64(val.data, data[3], tmp);
	Add320_to_256(buff + 3, tmp);
	//fast mod P
	Mul256_by_64(buff + 4, P_REV, tmp);
	u8 c = _addcarry_u64(0, buff[0], tmp[0], buff);
	c = _addcarry_u64(c, buff[1], tmp[1], buff + 1);
	c = _addcarry_u64(c, buff[2], tmp[2], buff + 2);
	tmp[4] += _addcarry_u64(c, buff[3], tmp[3], buff + 3);
	c = _addcarry_u64(0, buff[0], _umul128(tmp[4], P_REV, &h), data);
	c = _addcarry_u64(c, buff[1], h, data + 1);
	c = _addcarry_u64(c, 0, buff[2], data + 2);
	data[4] = _addcarry_u64(c, buff[3], 0, data + 3);
	while (data[4])
		Sub(g_P);
}

void EcInt::Mul_u64(EcInt& val, u64 multiplier)
{
	Assign(val);
	Mul320_by_64(data, (u64)multiplier, data);
}

void EcInt::Mul_i64(EcInt& val, i64 multiplier)
{
	Assign(val);
	if (multiplier < 0)
	{
		Neg();
		multiplier = -multiplier;
	}
	Mul320_by_64(data, (u64)multiplier, data);
}

#define APPLY_DIV_SHIFT() kbnt -= index; val >>= index; matrix[0] <<= index; matrix[1] <<= index; 
	
// https://tches.iacr.org/index.php/TCHES/article/download/8298/7648/4494
//a bit tricky
void DIV_62(i64& kbnt, i64 modp, i64 val, i64* matrix)
{
	int index, cnt;
	_BitScanForward64((DWORD*)&index, val | 0x4000000000000000);
	APPLY_DIV_SHIFT();
	cnt = 62 - index;
	while (cnt > 0)
	{
		if (kbnt < 0)
		{
			kbnt = -kbnt;
			i64 tmp = -modp; modp = val; val = tmp;
			tmp = -matrix[0]; matrix[0] = matrix[2]; matrix[2] = tmp;
			tmp = -matrix[1]; matrix[1] = matrix[3]; matrix[3] = tmp;
		}
		int thr = cnt;
		if ((kbnt + 1) < cnt)
			thr = (int)(kbnt + 1);
		i64 mul = (-modp * val) & ((UINT64_MAX >> (64 - thr)) & 0x07);
		val += (modp * mul);
		matrix[2] += (matrix[0] * mul);
		matrix[3] += (matrix[1] * mul);
		_BitScanForward64((DWORD*)&index, val | (1ull << cnt));
		APPLY_DIV_SHIFT();
		cnt -= index;
	}
}

void EcInt::InvModP()
{
	i64 matrix[4];
	EcInt result, a, tmp, tmp2;
	EcInt modp, val;
	i64 kbnt = -1;
	matrix[1] = matrix[2] = 0;
	matrix[0] = matrix[3] = 1;	
	DIV_62(kbnt, g_P.data[0], data[0], matrix);
	modp.Mul_i64(g_P, matrix[0]);
	tmp.Mul_i64(*this, matrix[1]);
	modp.Add(tmp);
	modp.ShiftRight(62);
	val.Mul_i64(g_P, matrix[2]);
	tmp.Mul_i64(*this, matrix[3]);
	val.Add(tmp);
	val.ShiftRight(62);
	if (matrix[1] >= 0)
		result.Set(matrix[1]);
	else
	{
		result.Set(-matrix[1]);
		result.Neg();
	}
	if (matrix[3] >= 0)
		a.Set(matrix[3]);
	else
	{ 
		a.Set(-matrix[3]);
		a.Neg();
	}
	Mul320_by_64(g_P.data, (result.data[0] * 0xD838091DD2253531) & 0x3FFFFFFFFFFFFFFF, tmp.data);
	result.Add(tmp);
	result.ShiftRight(62);
	Mul320_by_64(g_P.data, (a.data[0] * 0xD838091DD2253531) & 0x3FFFFFFFFFFFFFFF, tmp.data);
	a.Add(tmp);
	a.ShiftRight(62);
	
	while (val.data[0] || val.data[1] || val.data[2] || val.data[3])
	{
		matrix[1] = matrix[2] = 0;
		matrix[0] = matrix[3] = 1;	
		DIV_62(kbnt, modp.data[0], val.data[0], matrix);
		tmp.Mul_i64(modp, matrix[0]);
		tmp2.Mul_i64(val, matrix[1]);
		tmp.Add(tmp2);
		tmp2.Mul_i64(val, matrix[3]);
		val.Mul_i64(modp, matrix[2]);
		val.Add(tmp2);
		val.ShiftRight(62);
		modp = tmp;
		modp.ShiftRight(62);
		tmp.Mul_i64(result, matrix[0]);
		tmp2.Mul_i64(a, matrix[1]);
		tmp.Add(tmp2);
		tmp2.Mul_i64(a, matrix[3]);
		a.Mul_i64(result, matrix[2]);
		a.Add(tmp2);
		Mul320_by_64(g_P.data, (a.data[0] * 0xD838091DD2253531) & 0x3FFFFFFFFFFFFFFF, tmp2.data);
		a.Add(tmp2);
		a.ShiftRight(62);	
		Mul320_by_64(g_P.data, (tmp.data[0] * 0xD838091DD2253531) & 0x3FFFFFFFFFFFFFFF, tmp2.data);
		result = tmp;
		result.Add(tmp2);
		result.ShiftRight(62);
	}
	Assign(result);
	if (modp.data[4] >> 63)
	{
		Neg();
		modp.Neg();	
	}

	if (modp.data[0] == 1) 
	{
		if (data[4] >> 63)
			Add(g_P);
		if (data[4] >> 63)
			Add(g_P);
		if (!IsLessThanU(g_P))
			Sub(g_P);
		if (!IsLessThanU(g_P))
			Sub(g_P);
	}
	else
		SetZero(); //error
}

// x = a^ { (p + 1) / 4 } mod p
void EcInt::SqrtModP()
{
	EcInt one, res;
	one.Set(1);
	EcInt exp = g_P;
	exp.Add(one);
	exp.ShiftRight(2);
	res.Set(1);
	EcInt cur = *this;
	while (!exp.IsZero())
	{
		if (exp.data[0] & 1)
			res.MulModP(cur);
		EcInt tmp = cur;
		tmp.MulModP(cur);
		cur = tmp;
		exp.ShiftRight(1);
	}
	*this = res;
}

std::mt19937_64 rng;
CriticalSection cs_rnd;

void SetRndSeed(u64 seed)
{
	rng.seed(seed);
}

void EcInt::RndBits(int nbits)
{
	SetZero();
	if (nbits > 256)
		nbits = 256;
	cs_rnd.Enter();
	for (int i = 0; i < (nbits + 63) / 64; i++)
		data[i] = rng();
	cs_rnd.Leave();
	data[nbits / 64] &= (1ull << (nbits % 64)) - 1;
}

//up to 256 bits only
void EcInt::RndMax(EcInt& max)
{
	SetZero();
	int n = 3;
	while ((n >= 0) && !max.data[n])
		n--;
	if (n < 0)
		return;
	u64 val = max.data[n];
	int k = 0;
	while ((val & 0x8000000000000000) == 0)
	{
		val <<= 1;
		k++;
	}
	int bits = 64 * n + (64 - k);
	RndBits(bits);
	while (!IsLessThanU(max)) // :)
		RndBits(bits);
}

// ============================================================================
// Helper: Reduce 512-bit value mod n using iterative 2^256 ≡ c (mod n) trick
// For secp256k1: c = 2^256 - n = 0x14551231950B75FC4402DA1732FC9BEBF (129 bits)
// 
// Algorithm: product = hi*2^256 + lo ≡ lo + hi*c (mod n)
// Since c is 129 bits and hi is 256 bits, hi*c is at most 385 bits.
// Repeat until result fits in 256 bits, then final subtract.
// Guaranteed O(1) iterations (typically 2-3 rounds + max 2 subtractions).
// ============================================================================
static void Reduce512ModN(u64 r[8], u64 out[4])
{
	// c = 2^256 mod n (the small constant)
	// c = 0x00000000_00000001_4551231950B75FC4_402DA1732FC9BEBF
	const u64 c0 = 0x402DA1732FC9BEBFull;
	const u64 c1 = 0x4551231950B75FC4ull;
	const u64 c2 = 0x0000000000000001ull;
	// c3 = 0
	
	// Working buffer: start with the full 512-bit product
	// We'll iteratively reduce: result = lo + hi * c
	u64 lo[4] = { r[0], r[1], r[2], r[3] };
	u64 hi[4] = { r[4], r[5], r[6], r[7] };
	
	// Iterate: while hi != 0, compute lo += hi * c, then split at 256 bits
	for (int round = 0; round < 4; round++)
	{
		if (!(hi[0] | hi[1] | hi[2] | hi[3])) break;
		
		// Compute hi * c (256-bit * 129-bit → up to 385-bit result)
		// c has 3 words: c0, c1, c2 (c2 = 1, so hi*c2 = hi)
		u64 prod[8] = {0};
		
		// Multiply hi[0..3] × c[0..2]
		for (int i = 0; i < 4; i++)
		{
			if (!hi[i]) continue;
			uint128_t carry;
			
			// × c0
			carry = UINT128_MUL(hi[i], c0) + prod[i];
			prod[i] = UINT128_LO(carry);
			
			// × c1
			carry = UINT128_MUL(hi[i], c1) + prod[i+1] + UINT128_SHR64(carry);
			prod[i+1] = UINT128_LO(carry);
			
			// × c2 (c2 = 1, so just add hi[i])
			carry = uint128_t(hi[i]) + prod[i+2] + UINT128_SHR64(carry);
			prod[i+2] = UINT128_LO(carry);
			
			// Propagate remaining carry
			u64 c = UINT128_SHR64(carry);
			for (int k = i+3; k < 8 && c; k++)
			{
				uint128_t s = uint128_t(prod[k]) + c;
				prod[k] = UINT128_LO(s);
				c = UINT128_SHR64(s);
			}
		}
		
		// Add lo to prod: result = lo + hi*c
		uint128_t sum = uint128_t(lo[0]) + prod[0];
		lo[0] = UINT128_LO(sum);
		sum = uint128_t(lo[1]) + prod[1] + UINT128_SHR64(sum);
		lo[1] = UINT128_LO(sum);
		sum = uint128_t(lo[2]) + prod[2] + UINT128_SHR64(sum);
		lo[2] = UINT128_LO(sum);
		sum = uint128_t(lo[3]) + prod[3] + UINT128_SHR64(sum);
		lo[3] = UINT128_LO(sum);
		
		// New hi = upper part of result
		u64 carry_out = UINT128_SHR64(sum);
		hi[0] = prod[4] + carry_out;
		carry_out = (hi[0] < prod[4]) ? 1 : 0;
		hi[1] = prod[5] + carry_out;
		carry_out = (hi[1] < prod[5]) ? 1 : 0;
		hi[2] = prod[6] + carry_out;
		carry_out = (hi[2] < prod[6]) ? 1 : 0;
		hi[3] = prod[7] + carry_out;
	}
	
	// Final: lo should now be close to n. Subtract n at most 3 times.
	for (int sub = 0; sub < 3; sub++)
	{
		// Check if lo >= n
		bool ge_n = false;
		if (lo[3] > g_N.data[3]) ge_n = true;
		else if (lo[3] == g_N.data[3]) {
			if (lo[2] > g_N.data[2]) ge_n = true;
			else if (lo[2] == g_N.data[2]) {
				if (lo[1] > g_N.data[1]) ge_n = true;
				else if (lo[1] == g_N.data[1]) {
					if (lo[0] >= g_N.data[0]) ge_n = true;
				}
			}
		}
		if (!ge_n) break;
		
		// lo -= n
		u64 borrow = 0;
		for (int i = 0; i < 4; i++)
		{
			u64 old = lo[i];
			u64 sub_val = g_N.data[i] + borrow;
			// Check for overflow of g_N.data[i] + borrow
			u64 new_borrow = (sub_val < g_N.data[i]) ? 1 : 0;  // overflow in addition
			lo[i] = old - sub_val;
			new_borrow |= (old < sub_val) ? 1 : 0;  // borrow in subtraction
			borrow = new_borrow;
		}
	}
	
	out[0] = lo[0]; out[1] = lo[1]; out[2] = lo[2]; out[3] = lo[3];
}

// Multiply by lambda mod n (for endomorphism)
// Uses schoolbook 256×256→512 multiplication + Reduce512ModN
void EcInt::MulLambdaModN()
{
	u64 r[8] = {0};
	
	for (int i = 0; i < 4; i++)
	{
		u64 carry = 0;
		for (int j = 0; j < 4; j++)
		{
			uint128_t prod = UINT128_MUL(data[i], g_Lambda.data[j]);
			prod += r[i+j];
			prod += carry;
			r[i+j] = UINT128_LO(prod);
			carry = UINT128_SHR64(prod);
		}
		r[i+4] = carry;
	}
	
	u64 out[4];
	Reduce512ModN(r, out);
	data[0] = out[0]; data[1] = out[1]; data[2] = out[2]; data[3] = out[3]; data[4] = 0;
}

// Multiply by lambda² mod n
void EcInt::MulLambda2ModN()
{
	u64 r[8] = {0};
	
	for (int i = 0; i < 4; i++)
	{
		u64 carry = 0;
		for (int j = 0; j < 4; j++)
		{
			uint128_t prod = UINT128_MUL(data[i], g_Lambda2.data[j]);
			prod += r[i+j];
			prod += carry;
			r[i+j] = UINT128_LO(prod);
			carry = UINT128_SHR64(prod);
		}
		r[i+4] = carry;
	}
	
	u64 out[4];
	Reduce512ModN(r, out);
	data[0] = out[0]; data[1] = out[1]; data[2] = out[2]; data[3] = out[3]; data[4] = 0;
}

