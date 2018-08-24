#ifndef _SECP256K1_CUH
#define _SECP256K1_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "ptx.cuh"
#include "bitreverse.h"


__device__ __forceinline__ void copyBigInt(const uint32_t *src, uint32_t *dest);

/**
 Prime modulus 2^256 - 2^32 - 977
 */
__constant__ uint32_t _P[8] = {
	0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
};

/**
 Base point X
 */
__constant__ uint32_t _GX[8] = {
	0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07, 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798
};


/**
 Base point Y
 */
__constant__ uint32_t _GY[8] = {
	0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8, 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8
};


/**
 * Group order
 */
__constant__ uint32_t _N[8] = {
	0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141
};

__constant__ uint32_t _BETA[8] = {
	0x7AE96A2B, 0x657C0710, 0x6E64479E, 0xAC3434E9, 0x9CF04975, 0x12F58995, 0xC1396C28, 0x719501EE
};


__constant__ uint32_t _LAMBDA[8] = {
	0x5363AD4C, 0xC05C30E0, 0xA5261C02, 0x8812645A, 0x122E22EA, 0x20816678, 0xDF02967C, 0x1B23BD72
};

__device__ int dnKEY_PER_THREAD;


__device__ __forceinline__ void copyBigInt(const uint32_t *src, uint32_t *dest)
{
	for(int i = 0; i < 8; i++) {
		dest[i] = src[i];
	}
}

__device__ bool equal(const uint32_t *a, const uint32_t *b)
{
	bool eq = true;

	for(int i = 0; i < 8; i++) {
		eq &= (a[i] == b[i]);
	}

	return eq;
}

/**
 * Reads an 8-word big integer from device memory
 */
__device__ void readInt(const uint32_t *ara, int idx, uint32_t x[8])
{
	int index = ((((blockDim.x * blockIdx.x + threadIdx.x)  * dnKEY_PER_THREAD) + idx) * 8);

	for (int i = 0; i < 8; i++) {
		x[i] = ara[index];
		index++;
	}
}

/**
 * Writes an 8-word big integer to device memory
 */
__device__ void writeInt(uint32_t *ara, int idx, const uint32_t x[8])
{
	int index = ((((blockDim.x * blockIdx.x + threadIdx.x) * dnKEY_PER_THREAD) + idx) * 8);

	for (int i = 0; i < 8; i++) {
		ara[index] = x[i];
		index++; 
	}
}

__device__ void writePoint(uint32_t *arx, uint32_t *ary, int idx, const uint32_t x[8], const uint32_t y[8])
{
	int index = ((((blockDim.x * blockIdx.x + threadIdx.x) * dnKEY_PER_THREAD) + idx) * 8);

	for (int i = 0; i < 8; i++) {
		arx[index] = x[i];
		ary[index] = y[i];
		index++;
	}
}

/**
 * Subtraction mod p
 */
__device__ void subModP(const uint32_t a[8], const uint32_t b[8], uint32_t c[8])
{
	sub_cc(c[7], a[7], b[7]);
	subc_cc(c[6], a[6], b[6]);
	subc_cc(c[5], a[5], b[5]);
	subc_cc(c[4], a[4], b[4]);
	subc_cc(c[3], a[3], b[3]);
	subc_cc(c[2], a[2], b[2]);
	subc_cc(c[1], a[1], b[1]);
	subc_cc(c[0], a[0], b[0]);

	uint32_t borrow = 0;
	subc(borrow, 0, 0);

	if (borrow) {
		add_cc(c[7], c[7], _P[7]);
		addc_cc(c[6], c[6], _P[6]);
		addc_cc(c[5], c[5], _P[5]);
		addc_cc(c[4], c[4], _P[4]);
		addc_cc(c[3], c[3], _P[3]);
		addc_cc(c[2], c[2], _P[2]);
		addc_cc(c[1], c[1], _P[1]);
		addc(c[0], c[0], _P[0]);
	}
}

__device__ uint32_t add(const uint32_t a[8], const uint32_t b[8], uint32_t c[8])
{
	add_cc(c[7], a[7], b[7]);
	addc_cc(c[6], a[6], b[6]);
	addc_cc(c[5], a[5], b[5]);
	addc_cc(c[4], a[4], b[4]);
	addc_cc(c[3], a[3], b[3]);
	addc_cc(c[2], a[2], b[2]);
	addc_cc(c[1], a[1], b[1]);
	addc_cc(c[0], a[0], b[0]);

	uint32_t carry = 0;
	addc(carry, 0, 0);

	return carry;
}

__device__ uint32_t sub(const uint32_t a[8], const uint32_t b[8], uint32_t c[8])
{
	sub_cc(c[7], a[7], b[7]);
	subc_cc(c[6], a[6], b[6]);
	subc_cc(c[5], a[5], b[5]);
	subc_cc(c[4], a[4], b[4]);
	subc_cc(c[3], a[3], b[3]);
	subc_cc(c[2], a[2], b[2]);
	subc_cc(c[1], a[1], b[1]);
	subc_cc(c[0], a[0], b[0]);

	uint32_t borrow = 0;
	subc(borrow, 0, 0);

	return (borrow & 0x01);
}

/**
   Subtract using two's compliment
 */
__device__ uint32_t sub2c(const uint32_t a[8], const uint32_t b[8], uint32_t c[8])
{
	add_cc(c[7], a[7], ~b[7]);
	addc_cc(c[6], a[6], ~b[6]);
	addc_cc(c[5], a[5], ~b[5]);
	addc_cc(c[4], a[4], ~b[4]);
	addc_cc(c[3], a[3], ~b[3]);
	addc_cc(c[2], a[2], ~b[2]);
	addc_cc(c[1], a[1], ~b[1]);
	addc_cc(c[0], a[0], ~b[0]);

	uint32_t carry = 0;
	addc(carry, 0, 0);

	add_cc(c[7], c[7], 1);
	addc_cc(c[6], c[6], 0);
	addc_cc(c[5], c[5], 0);
	addc_cc(c[4], c[4], 0);
	addc_cc(c[3], c[3], 0);
	addc_cc(c[2], c[2], 0);
	addc_cc(c[1], c[1], 0);
	addc_cc(c[0], c[0], 0);

	addc(carry, carry, 0);

	return carry;
}

__device__ void divBy2(uint32_t x[8])
{
	x[7] = (x[7] >> 1) | (x[6] << 31);
	x[6] = (x[6] >> 1) | (x[5] << 31);
	x[5] = (x[5] >> 1) | (x[4] << 31);
	x[4] = (x[4] >> 1) | (x[3] << 31);
	x[3] = (x[3] >> 1) | (x[2] << 31);
	x[2] = (x[2] >> 1) | (x[1] << 31);
	x[1] = (x[1] >> 1) | (x[0] << 31);
	x[0] = (x[0] >> 1);
}

__device__ void addModP(const uint32_t a[8], const uint32_t b[8], uint32_t c[8])
{
	add_cc(c[7], a[7], b[7]);
	addc_cc(c[6], a[6], b[6]);
	addc_cc(c[5], a[5], b[5]);
	addc_cc(c[4], a[4], b[4]);
	addc_cc(c[3], a[3], b[3]);
	addc_cc(c[2], a[2], b[2]);
	addc_cc(c[1], a[1], b[1]);
	addc_cc(c[0], a[0], b[0]);

	uint32_t carry = 0;
	addc(carry, 0, 0);

	bool gt = false;
	for(int i = 0; i < 8; i++) {
		if(c[i] > _P[i]) {
			gt = true;
			break;
		} else if(c[i] < _P[i]) {
			break;
		}
	}

	if(carry || gt) {
		sub_cc(c[7], c[7], _P[7]);
		subc_cc(c[6], c[6], _P[6]);
		subc_cc(c[5], c[5], _P[5]);
		subc_cc(c[4], c[4], _P[4]);
		subc_cc(c[3], c[3], _P[3]);
		subc_cc(c[2], c[2], _P[2]);
		subc_cc(c[1], c[1], _P[1]);
		subc(c[0], c[0], _P[0]);
	}
}



/**
 * Multiplication mod P
*/
__device__ void mulModP(const uint32_t a[8], const uint32_t b[8], uint32_t c[8])
{
	uint32_t high[8];

	uint32_t t = a[7];

	// a[7] * b (low)
	for (int i = 7; i >= 0; i--) {
		c[i] = t * b[i];
		high[i] = 0;
	}

	// a[7] * b (high)
	mad_hi_cc(c[6], t, b[7], c[6]);
	madc_hi_cc(c[5], t, b[6], c[5]);
	madc_hi_cc(c[4], t, b[5], c[4]);
	madc_hi_cc(c[3], t, b[4], c[3]);
	madc_hi_cc(c[2], t, b[3], c[2]);
	madc_hi_cc(c[1], t, b[2], c[1]);
	madc_hi_cc(c[0], t, b[1], c[0]);
	madc_hi(high[7], t, b[0], high[7]);

	// a[6] * b (low)
	t = a[6];
	mad_lo_cc(c[6], t, b[7], c[6]);
	madc_lo_cc(c[5], t, b[6], c[5]);
	madc_lo_cc(c[4], t, b[5], c[4]);
	madc_lo_cc(c[3], t, b[4], c[3]);
	madc_lo_cc(c[2], t, b[3], c[2]);
	madc_lo_cc(c[1], t, b[2], c[1]);
	madc_lo_cc(c[0], t, b[1], c[0]);
	madc_lo_cc(high[7], t, b[0], high[7]);
	addc(high[6], high[6], 0);

	// a[6] * b (high)
	mad_hi_cc(c[5], t, b[7], c[5]);
	madc_hi_cc(c[4], t, b[6], c[4]);
	madc_hi_cc(c[3], t, b[5], c[3]);
	madc_hi_cc(c[2], t, b[4], c[2]);
	madc_hi_cc(c[1], t, b[3], c[1]);
	madc_hi_cc(c[0], t, b[2], c[0]);
	madc_hi_cc(high[7], t, b[1], high[7]);
	madc_hi(high[6], t, b[0], high[6]);

	// a[5] * b (low)
	t = a[5];
	mad_lo_cc(c[5], t, b[7], c[5]);
	madc_lo_cc(c[4], t, b[6], c[4]);
	madc_lo_cc(c[3], t, b[5], c[3]);
	madc_lo_cc(c[2], t, b[4], c[2]);
	madc_lo_cc(c[1], t, b[3], c[1]);
	madc_lo_cc(c[0], t, b[2], c[0]);
	madc_lo_cc(high[7], t, b[1], high[7]);
	madc_lo_cc(high[6], t, b[0], high[6]);
	addc(high[5], high[5], 0);

	// a[5] * b (high)
	mad_hi_cc(c[4], t, b[7], c[4]);
	madc_hi_cc(c[3], t, b[6], c[3]);
	madc_hi_cc(c[2], t, b[5], c[2]);
	madc_hi_cc(c[1], t, b[4], c[1]);
	madc_hi_cc(c[0], t, b[3], c[0]);
	madc_hi_cc(high[7], t, b[2], high[7]);
	madc_hi_cc(high[6], t, b[1], high[6]);
	madc_hi(high[5], t, b[0], high[5]);

	// a[4] * b (low)
	t = a[4];
	mad_lo_cc(c[4], t, b[7], c[4]);
	madc_lo_cc(c[3], t, b[6], c[3]);
	madc_lo_cc(c[2], t, b[5], c[2]);
	madc_lo_cc(c[1], t, b[4], c[1]);
	madc_lo_cc(c[0], t, b[3], c[0]);
	madc_lo_cc(high[7], t, b[2], high[7]);
	madc_lo_cc(high[6], t, b[1], high[6]);
	madc_lo_cc(high[5], t, b[0], high[5]);
	addc(high[4], high[4], 0);

	// a[4] * b (high)
	mad_hi_cc(c[3], t, b[7], c[3]);
	madc_hi_cc(c[2], t, b[6], c[2]);
	madc_hi_cc(c[1], t, b[5], c[1]);
	madc_hi_cc(c[0], t, b[4], c[0]);
	madc_hi_cc(high[7], t, b[3], high[7]);
	madc_hi_cc(high[6], t, b[2], high[6]);
	madc_hi_cc(high[5], t, b[1], high[5]);
	madc_hi(high[4], t, b[0], high[4]);

	// a[3] * b (low)
	t = a[3];
	mad_lo_cc(c[3], t, b[7], c[3]);
	madc_lo_cc(c[2], t, b[6], c[2]);
	madc_lo_cc(c[1], t, b[5], c[1]);
	madc_lo_cc(c[0], t, b[4], c[0]);
	madc_lo_cc(high[7], t, b[3], high[7]);
	madc_lo_cc(high[6], t, b[2], high[6]);
	madc_lo_cc(high[5], t, b[1], high[5]);
	madc_lo_cc(high[4], t, b[0], high[4]);
	addc(high[3], high[3], 0);

	// a[3] * b (high)
	mad_hi_cc(c[2], t, b[7], c[2]);
	madc_hi_cc(c[1], t, b[6], c[1]);
	madc_hi_cc(c[0], t, b[5], c[0]);
	madc_hi_cc(high[7], t, b[4], high[7]);
	madc_hi_cc(high[6], t, b[3], high[6]);
	madc_hi_cc(high[5], t, b[2], high[5]);
	madc_hi_cc(high[4], t, b[1], high[4]);
	madc_hi(high[3], t, b[0], high[3]);

	// a[2] * b (low)
	t = a[2];
	mad_lo_cc(c[2], t, b[7], c[2]);
	madc_lo_cc(c[1], t, b[6], c[1]);
	madc_lo_cc(c[0], t, b[5], c[0]);
	madc_lo_cc(high[7], t, b[4], high[7]);
	madc_lo_cc(high[6], t, b[3], high[6]);
	madc_lo_cc(high[5], t, b[2], high[5]);
	madc_lo_cc(high[4], t, b[1], high[4]);
	madc_lo_cc(high[3], t, b[0], high[3]);
	addc(high[2], high[2], 0);

	// a[2] * b (high)
	mad_hi_cc(c[1], t, b[7], c[1]);
	madc_hi_cc(c[0], t, b[6], c[0]);
	madc_hi_cc(high[7], t, b[5], high[7]);
	madc_hi_cc(high[6], t, b[4], high[6]);
	madc_hi_cc(high[5], t, b[3], high[5]);
	madc_hi_cc(high[4], t, b[2], high[4]);
	madc_hi_cc(high[3], t, b[1], high[3]);
	madc_hi(high[2], t, b[0], high[2]);

	// a[1] * b (low)
	t = a[1];
	mad_lo_cc(c[1], t, b[7], c[1]);
	madc_lo_cc(c[0], t, b[6], c[0]);
	madc_lo_cc(high[7], t, b[5], high[7]);
	madc_lo_cc(high[6], t, b[4], high[6]);
	madc_lo_cc(high[5], t, b[3], high[5]);
	madc_lo_cc(high[4], t, b[2], high[4]);
	madc_lo_cc(high[3], t, b[1], high[3]);
	madc_lo_cc(high[2], t, b[0], high[2]);
	addc(high[1], high[1], 0);

	// a[1] * b (high)
	mad_hi_cc(c[0], t, b[7], c[0]);
	madc_hi_cc(high[7], t, b[6], high[7]);
	madc_hi_cc(high[6], t, b[5], high[6]);
	madc_hi_cc(high[5], t, b[4], high[5]);
	madc_hi_cc(high[4], t, b[3], high[4]);
	madc_hi_cc(high[3], t, b[2], high[3]);
	madc_hi_cc(high[2], t, b[1], high[2]);
	madc_hi(high[1], t, b[0], high[1]);

	// a[0] * b (low)
	t = a[0];
	mad_lo_cc(c[0], t, b[7], c[0]);
	madc_lo_cc(high[7], t, b[6], high[7]);
	madc_lo_cc(high[6], t, b[5], high[6]);
	madc_lo_cc(high[5], t, b[4], high[5]);
	madc_lo_cc(high[4], t, b[3], high[4]);
	madc_lo_cc(high[3], t, b[2], high[3]);
	madc_lo_cc(high[2], t, b[1], high[2]);
	madc_lo_cc(high[1], t, b[0], high[1]);
	addc(high[0], high[0], 0);

	// a[0] * b (high)
	mad_hi_cc(high[7], t, b[7], high[7]);
	madc_hi_cc(high[6], t, b[6], high[6]);
	madc_hi_cc(high[5], t, b[5], high[5]);
	madc_hi_cc(high[4], t, b[4], high[4]);
	madc_hi_cc(high[3], t, b[3], high[3]);
	madc_hi_cc(high[2], t, b[2], high[2]);
	madc_hi_cc(high[1], t, b[1], high[1]);
	madc_hi(high[0], t, b[0], high[0]);



	// At this point we have 16 32-bit words representing a 512-bit value
	// high[0 ... 7] and c[0 ... 7]
	const uint32_t s = 977;
	
	// Store high[6] and high[7] since they will be overwritten
	uint32_t high7 = high[7];
	uint32_t high6 = high[6];


	// Take high 256 bits, multiply by 2^32, add to low 256 bits
	// That is, take high[0 ... 7], shift it left 1 word and add it to c[0 ... 7]
	add_cc(c[6], high[7], c[6]);
	addc_cc(c[5], high[6], c[5]);
	addc_cc(c[4], high[5], c[4]);
	addc_cc(c[3], high[4], c[3]);
	addc_cc(c[2], high[3], c[2]);
	addc_cc(c[1], high[2], c[1]);
	addc_cc(c[0], high[1], c[0]);
	addc_cc(high[7], high[0], 0);
	addc(high[6], 0, 0);


	// Take high 256 bits, multiply by 977, add to low 256 bits
	// That is, take high[0 ... 5], high6, high7, multiply by 977 and add to c[0 ... 7]
	mad_lo_cc(c[7], high7, s, c[7]);
	madc_lo_cc(c[6], high6, s, c[6]);
	madc_lo_cc(c[5], high[5], s, c[5]);
	madc_lo_cc(c[4], high[4], s, c[4]);
	madc_lo_cc(c[3], high[3], s, c[3]);
	madc_lo_cc(c[2], high[2], s, c[2]);
	madc_lo_cc(c[1], high[1], s, c[1]);
	madc_lo_cc(c[0], high[0], s, c[0]);
	addc_cc(high[7], high[7], 0);
	addc(high[6], high[6], 0);


	mad_hi_cc(c[6], high7, s, c[6]);
	madc_hi_cc(c[5], high6, s, c[5]);
	madc_hi_cc(c[4], high[5], s, c[4]);
	madc_hi_cc(c[3], high[4], s, c[3]);
	madc_hi_cc(c[2], high[3], s, c[2]);
	madc_hi_cc(c[1], high[2], s, c[1]);
	madc_hi_cc(c[0], high[1], s, c[0]);
	madc_hi_cc(high[7], high[0], s, high[7]);
	addc(high[6], high[6], 0);


	// Repeat the same steps, but this time we only need to handle high[6] and high[7]
	high7 = high[7];
	high6 = high[6];

	// Take the high 64 bits, multiply by 2^32 and add to the low 256 bits
	add_cc(c[6], high[7], c[6]);
	addc_cc(c[5], high[6], c[5]);
	addc_cc(c[4], c[4], 0);
	addc_cc(c[3], c[3], 0);
	addc_cc(c[2], c[2], 0);
	addc_cc(c[1], c[1], 0);
	addc_cc(c[0], c[0], 0);
	addc(high[7], 0, 0);


	// Take the high 64 bits, multiply by 977 and add to the low 256 bits
	mad_lo_cc(c[7], high7, s, c[7]);
	madc_lo_cc(c[6], high6, s, c[6]);
	addc_cc(c[5], c[5], 0);
	addc_cc(c[4], c[4], 0);
	addc_cc(c[3], c[3], 0);
	addc_cc(c[2], c[2], 0);
	addc_cc(c[1], c[1], 0);
	addc_cc(c[0], c[0], 0);
	addc(high[7], high[7], 0);

	mad_hi_cc(c[6], high7, s, c[6]);
	madc_hi_cc(c[5], high6, s, c[5]);
	addc_cc(c[4], c[4], 0);
	addc_cc(c[3], c[3], 0);
	addc_cc(c[2], c[2], 0);
	addc_cc(c[1], c[1], 0);
	addc_cc(c[0], c[0], 0);
	addc(high[7], high[7], 0);


	bool overflow = high[7] != 0;

	uint32_t borrow = sub(c, _P, c);

	if(overflow) {
		if(!borrow) {
			sub(c, _P, c);
		}
	} else {
		if(borrow) {
			add(c, _P, c);
		}
	}
}

/**
 * Square mod P
 * b = a * a
 */
__device__ void squareModP(const uint32_t a[8], uint32_t b[8])
{
	mulModP(a, a, b);
}

/**
 * Square mod P
 * x = x * x
 */
__device__ void squareModP(uint32_t x[8])
{
	uint32_t tmp[8];
	squareModP(x, tmp);
	copyBigInt(tmp, x);
}

/**
 * Multiply mod P
 * c = a * c
 */
__device__ void mulModP(const uint32_t a[8], uint32_t c[8])
{
	uint32_t tmp[8];
	mulModP(a, c, tmp);
	copyBigInt(tmp, c);
}

/**
 * Multiplicative inverse mod P using Fermat's method of x^(p-2) mod p
 */
__device__ void invModP(uint32_t value[8])
{
	uint32_t x[8];

	copyBigInt(value, x);

	uint32_t y[8] = { 0, 0, 0, 0, 0, 0, 0, 1 };

	// 0xd - 1101
	mulModP(x, y);
	squareModP(x);
	//mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);


	// 0x2 - 0010
	//mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	//mulModP(x, y);
	squareModP(x);
	//mulModP(x, y);
	squareModP(x);

	// 0xc = 0x1100
	//mulModP(x, y);
	squareModP(x);
	//mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);

	// 0xfffff
	for(int i = 0; i < 20; i++) {
		mulModP(x, y);
		squareModP(x);
	}

	// 0xe - 1110
	//mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);
	mulModP(x, y);
	squareModP(x);

	// 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffff
	for(int i = 0; i < 219; i++) {
		mulModP(x, y);
		squareModP(x);
	}
	mulModP(x, y);

	copyBigInt(y, value);
}

__device__ void invModP(const uint32_t *value, uint32_t *inverse)
{
	copyBigInt(value, inverse);

	invModP(inverse);
}

__device__ void negModP(const uint32_t *value, uint32_t *negative)
{
	sub_cc(negative[0], _P[0], value[0]);
	subc_cc(negative[1], _P[1], value[1]);
	subc_cc(negative[2], _P[2], value[2]);
	subc_cc(negative[3], _P[3], value[3]);
	subc_cc(negative[4], _P[4], value[4]);
	subc_cc(negative[5], _P[5], value[5]);
	subc_cc(negative[6], _P[6], value[6]);
	subc(negative[7], _P[7], value[7]);
}


__device__ __forceinline__ void beginBatchAdd(const uint32_t *px, uint32_t *xPtr, uint32_t *chain, int i, uint32_t inverse[8])
{
	uint32_t x[8];
	readInt(xPtr, i, x);

	// x = Gx - x
	subModP(px, x, x);

	// Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
	// c[2] = diff2 * diff1 * diff0, etc
	mulModP(x, inverse);

	//memcpy(chain + (i * 8 * sizeof(uint32_t)), inverse, 8 * sizeof(uint32_t));

	int index = (i * 8);
	for (int j = 0; j < 8; j++) {
		chain[index] = inverse[j];
		index++;
	}
	//writeInt(chain, i, inverse);
}

__device__ __forceinline__ void beginBatchAddWithDouble(const uint32_t *px, const uint32_t *py, uint32_t *xPtr, uint32_t *chain, int i, uint32_t inverse[8])
{
	uint32_t x[8];
	readInt(xPtr, i, x);

	if(equal(px, x)) {
		addModP(py, py, x);
	} else {
		// x = Gx - x
		subModP(px, x, x);
	}

	// Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
	// c[2] = diff2 * diff1 * diff0, etc
	mulModP(x, inverse);

	writeInt(chain, i, inverse);
}


__device__ void completeBatchAddWithDouble(const uint32_t *px, const uint32_t *py, uint32_t *xPtr, uint32_t *yPtr, int i, uint32_t *chain, uint32_t *inverse, uint32_t newX[8], uint32_t newY[8])
{
	uint32_t s[8];
	uint32_t x[8];
	uint32_t y[8];

	readInt(xPtr, i, x);
	readInt(yPtr, i, y);

	if(i >= 1) {
		uint32_t c[8];

		readInt(chain, i - 1, c);

		mulModP(inverse, c, s);

		uint32_t diff[8];
		if(equal(px, x)) {
			addModP(py, py, diff);
		} else {
			subModP(px, x, diff);
		}

		mulModP(diff, inverse);
	} else {
		copyBigInt(inverse, s);
	}


	if(equal(px, x)) {
		// currently s = 1 / 2y

		uint32_t x2[8];
		uint32_t tx2[8];

		// 3x^2
		mulModP(x, x, x2);
		addModP(x2, x2, tx2);
		addModP(x2, tx2, tx2);


		// s = 3x^2 * 1/2y
		mulModP(tx2, s);

		// s^2
		uint32_t s2[8];
		mulModP(s, s, s2);

		// Rx = s^2 - 2px
		subModP(s2, x, newX);
		subModP(newX, x, newX);

		// Ry = s(px - rx) - py
		uint32_t k[8];
		subModP(px, newX, k);
		mulModP(s, k, newY);
		subModP(newY, py, newY);

	} else {

		uint32_t rise[8];
		subModP(py, y, rise);

		mulModP(rise, s);

		// Rx = s^2 - Gx - Qx
		uint32_t s2[8];
		mulModP(s, s, s2);

		subModP(s2, px, newX);
		subModP(newX, x, newX);

		// Ry = s(px - rx) - py
		uint32_t k[8];
		subModP(px, newX, k);
		mulModP(s, k, newY);
		subModP(newY, py, newY);
	}
}


__device__ void completeBatchAdd(const uint32_t *px, const uint32_t *py, uint32_t *xPtr, uint32_t *yPtr, int i, uint32_t *chain, uint32_t *inverse, uint32_t newX[8], uint32_t newY[8])
{
	uint32_t s[8];
	uint32_t x[8];

	readInt(xPtr, i, x);

	if(i >= 1) {
		uint32_t c[8];

		//readInt(chain, i - 1, c);
		int index = ((i - 1) * 8);
		for (int j = 0; j < 8; j++) {
			c[j] = chain[index];
			index++;
		}

		//memcpy(c, chain + ((i-1) * 8 * sizeof(uint32_t)), 8 * sizeof(uint32_t));

		mulModP(inverse, c, s);

		uint32_t diff[8];
		subModP(px, x, diff);
		mulModP(diff, inverse);
	} else {
		copyBigInt(inverse, s);
	}

	uint32_t y[8];
	readInt(yPtr, i, y);

	uint32_t rise[8];
	subModP(py, y, rise);

	mulModP(rise, s);

	// Rx = s^2 - Gx - Qx
	uint32_t s2[8];
	mulModP(s, s, s2);
	subModP(s2, px, newX);
	subModP(newX, x, newX);

	// Ry = s(px - rx) - py
	uint32_t k[8];
	subModP(px, newX, k);
	mulModP(s, k, newY);
	subModP(newY, py, newY);
}


__device__ __forceinline__ void doBatchInverse(uint32_t inverse[8])
{
	invModP(inverse);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void dec_fe_mul_inner(const uint32_t *a, const uint32_t *b, uint32_t *r) {
	uint64_t c = (uint64_t)a[0] * b[0];
	uint32_t t0 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[0] * b[1] +
		(uint64_t)a[1] * b[0];
	uint32_t t1 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[0] * b[2] +
		(uint64_t)a[1] * b[1] +
		(uint64_t)a[2] * b[0];
	uint32_t t2 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[0] * b[3] +
		(uint64_t)a[1] * b[2] +
		(uint64_t)a[2] * b[1] +
		(uint64_t)a[3] * b[0];
	uint32_t t3 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[0] * b[4] +
		(uint64_t)a[1] * b[3] +
		(uint64_t)a[2] * b[2] +
		(uint64_t)a[3] * b[1] +
		(uint64_t)a[4] * b[0];
	uint32_t t4 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[0] * b[5] +
		(uint64_t)a[1] * b[4] +
		(uint64_t)a[2] * b[3] +
		(uint64_t)a[3] * b[2] +
		(uint64_t)a[4] * b[1] +
		(uint64_t)a[5] * b[0];
	uint32_t t5 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[0] * b[6] +
		(uint64_t)a[1] * b[5] +
		(uint64_t)a[2] * b[4] +
		(uint64_t)a[3] * b[3] +
		(uint64_t)a[4] * b[2] +
		(uint64_t)a[5] * b[1] +
		(uint64_t)a[6] * b[0];
	uint32_t t6 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[0] * b[7] +
		(uint64_t)a[1] * b[6] +
		(uint64_t)a[2] * b[5] +
		(uint64_t)a[3] * b[4] +
		(uint64_t)a[4] * b[3] +
		(uint64_t)a[5] * b[2] +
		(uint64_t)a[6] * b[1] +
		(uint64_t)a[7] * b[0];
	uint32_t t7 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[0] * b[8] +
		(uint64_t)a[1] * b[7] +
		(uint64_t)a[2] * b[6] +
		(uint64_t)a[3] * b[5] +
		(uint64_t)a[4] * b[4] +
		(uint64_t)a[5] * b[3] +
		(uint64_t)a[6] * b[2] +
		(uint64_t)a[7] * b[1] +
		(uint64_t)a[8] * b[0];
	uint32_t t8 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[0] * b[9] +
		(uint64_t)a[1] * b[8] +
		(uint64_t)a[2] * b[7] +
		(uint64_t)a[3] * b[6] +
		(uint64_t)a[4] * b[5] +
		(uint64_t)a[5] * b[4] +
		(uint64_t)a[6] * b[3] +
		(uint64_t)a[7] * b[2] +
		(uint64_t)a[8] * b[1] +
		(uint64_t)a[9] * b[0];
	uint32_t t9 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[1] * b[9] +
		(uint64_t)a[2] * b[8] +
		(uint64_t)a[3] * b[7] +
		(uint64_t)a[4] * b[6] +
		(uint64_t)a[5] * b[5] +
		(uint64_t)a[6] * b[4] +
		(uint64_t)a[7] * b[3] +
		(uint64_t)a[8] * b[2] +
		(uint64_t)a[9] * b[1];
	uint32_t t10 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[2] * b[9] +
		(uint64_t)a[3] * b[8] +
		(uint64_t)a[4] * b[7] +
		(uint64_t)a[5] * b[6] +
		(uint64_t)a[6] * b[5] +
		(uint64_t)a[7] * b[4] +
		(uint64_t)a[8] * b[3] +
		(uint64_t)a[9] * b[2];
	uint32_t t11 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[3] * b[9] +
		(uint64_t)a[4] * b[8] +
		(uint64_t)a[5] * b[7] +
		(uint64_t)a[6] * b[6] +
		(uint64_t)a[7] * b[5] +
		(uint64_t)a[8] * b[4] +
		(uint64_t)a[9] * b[3];
	uint32_t t12 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[4] * b[9] +
		(uint64_t)a[5] * b[8] +
		(uint64_t)a[6] * b[7] +
		(uint64_t)a[7] * b[6] +
		(uint64_t)a[8] * b[5] +
		(uint64_t)a[9] * b[4];
	uint32_t t13 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[5] * b[9] +
		(uint64_t)a[6] * b[8] +
		(uint64_t)a[7] * b[7] +
		(uint64_t)a[8] * b[6] +
		(uint64_t)a[9] * b[5];
	uint32_t t14 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[6] * b[9] +
		(uint64_t)a[7] * b[8] +
		(uint64_t)a[8] * b[7] +
		(uint64_t)a[9] * b[6];
	uint32_t t15 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[7] * b[9] +
		(uint64_t)a[8] * b[8] +
		(uint64_t)a[9] * b[7];
	uint32_t t16 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[8] * b[9] +
		(uint64_t)a[9] * b[8];
	uint32_t t17 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[9] * b[9];
	uint32_t t18 = c & 0x3FFFFFFUL; c = c >> 26;
	uint32_t t19 = c;

	c = t0 + (uint64_t)t10 * 0x3D10UL;
	t0 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t1 + (uint64_t)t10 * 0x400UL + (uint64_t)t11 * 0x3D10UL;
	t1 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t2 + (uint64_t)t11 * 0x400UL + (uint64_t)t12 * 0x3D10UL;
	t2 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t3 + (uint64_t)t12 * 0x400UL + (uint64_t)t13 * 0x3D10UL;
	r[3] = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t4 + (uint64_t)t13 * 0x400UL + (uint64_t)t14 * 0x3D10UL;
	r[4] = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t5 + (uint64_t)t14 * 0x400UL + (uint64_t)t15 * 0x3D10UL;
	r[5] = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t6 + (uint64_t)t15 * 0x400UL + (uint64_t)t16 * 0x3D10UL;
	r[6] = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t7 + (uint64_t)t16 * 0x400UL + (uint64_t)t17 * 0x3D10UL;
	r[7] = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t8 + (uint64_t)t17 * 0x400UL + (uint64_t)t18 * 0x3D10UL;
	r[8] = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t9 + (uint64_t)t18 * 0x400UL + (uint64_t)t19 * 0x1000003D10UL;
	r[9] = c & 0x03FFFFFUL; c = c >> 22;
	uint64_t d = t0 + c * 0x3D1UL;
	r[0] = d & 0x3FFFFFFUL; d = d >> 26;
	d = d + t1 + c * 0x40;
	r[1] = d & 0x3FFFFFFUL; d = d >> 26;
	r[2] = t2 + d;
}

__device__ void dec_fe_mul(ec_fe_t *r, const ec_fe_t *a, const ec_fe_t *b) {
	dec_fe_mul_inner(a->n, b->n, r->n);
}

__device__ void dec_fe_sqr_inner(const uint32_t *a, uint32_t *r) {
	uint64_t c = (uint64_t)a[0] * a[0];
	uint32_t t0 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[0] * 2) * a[1];
	uint32_t t1 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[0] * 2) * a[2] +
		(uint64_t)a[1] * a[1];
	uint32_t t2 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[0] * 2) * a[3] +
		(uint64_t)(a[1] * 2) * a[2];
	uint32_t t3 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[0] * 2) * a[4] +
		(uint64_t)(a[1] * 2) * a[3] +
		(uint64_t)a[2] * a[2];
	uint32_t t4 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[0] * 2) * a[5] +
		(uint64_t)(a[1] * 2) * a[4] +
		(uint64_t)(a[2] * 2) * a[3];
	uint32_t t5 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[0] * 2) * a[6] +
		(uint64_t)(a[1] * 2) * a[5] +
		(uint64_t)(a[2] * 2) * a[4] +
		(uint64_t)a[3] * a[3];
	uint32_t t6 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[0] * 2) * a[7] +
		(uint64_t)(a[1] * 2) * a[6] +
		(uint64_t)(a[2] * 2) * a[5] +
		(uint64_t)(a[3] * 2) * a[4];
	uint32_t t7 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[0] * 2) * a[8] +
		(uint64_t)(a[1] * 2) * a[7] +
		(uint64_t)(a[2] * 2) * a[6] +
		(uint64_t)(a[3] * 2) * a[5] +
		(uint64_t)a[4] * a[4];
	uint32_t t8 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[0] * 2) * a[9] +
		(uint64_t)(a[1] * 2) * a[8] +
		(uint64_t)(a[2] * 2) * a[7] +
		(uint64_t)(a[3] * 2) * a[6] +
		(uint64_t)(a[4] * 2) * a[5];
	uint32_t t9 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[1] * 2) * a[9] +
		(uint64_t)(a[2] * 2) * a[8] +
		(uint64_t)(a[3] * 2) * a[7] +
		(uint64_t)(a[4] * 2) * a[6] +
		(uint64_t)a[5] * a[5];
	uint32_t t10 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[2] * 2) * a[9] +
		(uint64_t)(a[3] * 2) * a[8] +
		(uint64_t)(a[4] * 2) * a[7] +
		(uint64_t)(a[5] * 2) * a[6];
	uint32_t t11 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[3] * 2) * a[9] +
		(uint64_t)(a[4] * 2) * a[8] +
		(uint64_t)(a[5] * 2) * a[7] +
		(uint64_t)a[6] * a[6];
	uint32_t t12 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[4] * 2) * a[9] +
		(uint64_t)(a[5] * 2) * a[8] +
		(uint64_t)(a[6] * 2) * a[7];
	uint32_t t13 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[5] * 2) * a[9] +
		(uint64_t)(a[6] * 2) * a[8] +
		(uint64_t)a[7] * a[7];
	uint32_t t14 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[6] * 2) * a[9] +
		(uint64_t)(a[7] * 2) * a[8];
	uint32_t t15 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[7] * 2) * a[9] +
		(uint64_t)a[8] * a[8];
	uint32_t t16 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)(a[8] * 2) * a[9];
	uint32_t t17 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + (uint64_t)a[9] * a[9];
	uint32_t t18 = c & 0x3FFFFFFUL; c = c >> 26;
	uint32_t t19 = c;

	c = t0 + (uint64_t)t10 * 0x3D10UL;
	t0 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t1 + (uint64_t)t10 * 0x400UL + (uint64_t)t11 * 0x3D10UL;
	t1 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t2 + (uint64_t)t11 * 0x400UL + (uint64_t)t12 * 0x3D10UL;
	t2 = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t3 + (uint64_t)t12 * 0x400UL + (uint64_t)t13 * 0x3D10UL;
	r[3] = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t4 + (uint64_t)t13 * 0x400UL + (uint64_t)t14 * 0x3D10UL;
	r[4] = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t5 + (uint64_t)t14 * 0x400UL + (uint64_t)t15 * 0x3D10UL;
	r[5] = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t6 + (uint64_t)t15 * 0x400UL + (uint64_t)t16 * 0x3D10UL;
	r[6] = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t7 + (uint64_t)t16 * 0x400UL + (uint64_t)t17 * 0x3D10UL;
	r[7] = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t8 + (uint64_t)t17 * 0x400UL + (uint64_t)t18 * 0x3D10UL;
	r[8] = c & 0x3FFFFFFUL; c = c >> 26;
	c = c + t9 + (uint64_t)t18 * 0x400UL + (uint64_t)t19 * 0x1000003D10UL;
	r[9] = c & 0x03FFFFFUL; c = c >> 22;
	uint64_t d = t0 + c * 0x3D1UL;
	r[0] = d & 0x3FFFFFFUL; d = d >> 26;
	d = d + t1 + c * 0x40;
	r[1] = d & 0x3FFFFFFUL; d = d >> 26;
	r[2] = t2 + d;
}

__device__ void dec_fe_sqr(ec_fe_t *r, const ec_fe_t *a) {
	dec_fe_sqr_inner(a->n, r->n);
}

__device__ void dec_fe_inv(ec_fe_t *r, const ec_fe_t *a) {

	ec_fe_t x2;
	dec_fe_sqr(&x2, a);
	dec_fe_mul(&x2, &x2, a);

	ec_fe_t x3;
	dec_fe_sqr(&x3, &x2);
	dec_fe_mul(&x3, &x3, a);

	ec_fe_t x6 = x3;
	for (int j = 0; j<3; j++) dec_fe_sqr(&x6, &x6);
	dec_fe_mul(&x6, &x6, &x3);

	ec_fe_t x9 = x6;
	for (int j = 0; j<3; j++) dec_fe_sqr(&x9, &x9);
	dec_fe_mul(&x9, &x9, &x3);

	ec_fe_t x11 = x9;
	for (int j = 0; j<2; j++) dec_fe_sqr(&x11, &x11);
	dec_fe_mul(&x11, &x11, &x2);

	ec_fe_t x22 = x11;
	for (int j = 0; j<11; j++) dec_fe_sqr(&x22, &x22);
	dec_fe_mul(&x22, &x22, &x11);

	ec_fe_t x44 = x22;
	for (int j = 0; j<22; j++) dec_fe_sqr(&x44, &x44);
	dec_fe_mul(&x44, &x44, &x22);

	ec_fe_t x88 = x44;
	for (int j = 0; j<44; j++) dec_fe_sqr(&x88, &x88);
	dec_fe_mul(&x88, &x88, &x44);

	ec_fe_t x176 = x88;
	for (int j = 0; j<88; j++) dec_fe_sqr(&x176, &x176);
	dec_fe_mul(&x176, &x176, &x88);

	ec_fe_t x220 = x176;
	for (int j = 0; j<44; j++) dec_fe_sqr(&x220, &x220);
	dec_fe_mul(&x220, &x220, &x44);

	ec_fe_t x223 = x220;
	for (int j = 0; j<3; j++) dec_fe_sqr(&x223, &x223);
	dec_fe_mul(&x223, &x223, &x3);

	// The final result is then assembled using a sliding window over the blocks.

	ec_fe_t t1 = x223;
	for (int j = 0; j<23; j++) dec_fe_sqr(&t1, &t1);
	dec_fe_mul(&t1, &t1, &x22);
	for (int j = 0; j<5; j++) dec_fe_sqr(&t1, &t1);
	dec_fe_mul(&t1, &t1, a);
	for (int j = 0; j<3; j++) dec_fe_sqr(&t1, &t1);
	dec_fe_mul(&t1, &t1, &x2);
	for (int j = 0; j<2; j++) dec_fe_sqr(&t1, &t1);
	dec_fe_mul(r, &t1, a);
}

__device__ void dec_fe_set_int(ec_fe_t *r, int a) {
	r->n[0] = a;
	r->n[1] = r->n[2] = r->n[3] = r->n[4] = r->n[5] = r->n[6] = r->n[7] = r->n[8] = r->n[9] = 0;
}

__device__ void dec_ge_set_gej(ec_ge_t *r, ec_gej_t *a) {
	ec_fe_t z2, z3;
	r->infinity = a->infinity;
	dec_fe_inv(&a->z, &a->z);
	dec_fe_sqr(&z2, &a->z);
	dec_fe_mul(&z3, &a->z, &z2);
	dec_fe_mul(&a->x, &a->x, &z2);
	dec_fe_mul(&a->y, &a->y, &z3);
	dec_fe_set_int(&a->z, 1);
	r->x = a->x;
	r->y = a->y;
}

__device__ int dec_fe_is_odd(ec_fe_t *r) {
	return r->n[0] & 1;
}

__device__ void dec_fe_normalize_var(ec_fe_t *r) {
	uint32_t t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4],
		t5 = r->n[5], t6 = r->n[6], t7 = r->n[7], t8 = r->n[8], t9 = r->n[9];

	/* Reduce t9 at the start so there will be at most a single carry from the first pass */
	uint32_t m;
	uint32_t x = t9 >> 22; t9 &= 0x03FFFFFUL;

	/* The first pass ensures the magnitude is 1, ... */
	t0 += x * 0x3D1UL; t1 += (x << 6);
	t1 += (t0 >> 26); t0 &= 0x3FFFFFFUL;
	t2 += (t1 >> 26); t1 &= 0x3FFFFFFUL;
	t3 += (t2 >> 26); t2 &= 0x3FFFFFFUL; m = t2;
	t4 += (t3 >> 26); t3 &= 0x3FFFFFFUL; m &= t3;
	t5 += (t4 >> 26); t4 &= 0x3FFFFFFUL; m &= t4;
	t6 += (t5 >> 26); t5 &= 0x3FFFFFFUL; m &= t5;
	t7 += (t6 >> 26); t6 &= 0x3FFFFFFUL; m &= t6;
	t8 += (t7 >> 26); t7 &= 0x3FFFFFFUL; m &= t7;
	t9 += (t8 >> 26); t8 &= 0x3FFFFFFUL; m &= t8;

	/* At most a single final reduction is needed; check if the value is >= the field characteristic */
	x = (t9 >> 22) | ((t9 == 0x03FFFFFUL) & (m == 0x3FFFFFFUL)
		& ((t1 + 0x40UL + ((t0 + 0x3D1UL) >> 26)) > 0x3FFFFFFUL));

	if (x) {
		t0 += 0x3D1UL; t1 += (x << 6);
		t1 += (t0 >> 26); t0 &= 0x3FFFFFFUL;
		t2 += (t1 >> 26); t1 &= 0x3FFFFFFUL;
		t3 += (t2 >> 26); t2 &= 0x3FFFFFFUL;
		t4 += (t3 >> 26); t3 &= 0x3FFFFFFUL;
		t5 += (t4 >> 26); t4 &= 0x3FFFFFFUL;
		t6 += (t5 >> 26); t5 &= 0x3FFFFFFUL;
		t7 += (t6 >> 26); t6 &= 0x3FFFFFFUL;
		t8 += (t7 >> 26); t7 &= 0x3FFFFFFUL;
		t9 += (t8 >> 26); t8 &= 0x3FFFFFFUL;

		/* Mask off the possible multiple of 2^256 from the final reduction */
		t9 &= 0x03FFFFFUL;
	}

	r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
	r->n[5] = t5; r->n[6] = t6; r->n[7] = t7; r->n[8] = t8; r->n[9] = t9;
}

__device__ void dec_fe_get_b32(unsigned char *r, const ec_fe_t *a) {

	for (int i = 0; i<32; i++) {
		int c = 0;
		for (int j = 0; j<4; j++) {
			int limb = (8 * i + 2 * j) / 26;
			int shift = (8 * i + 2 * j) % 26;
			c |= ((a->n[limb] >> shift) & 0x3) << (2 * j);
		}

		// endianize r[31 - i] = c;
		int z = (i % 4) - 3;
		int s = (i / 4) * 4;
		r[(31 - s) + z] = c;
	}

}

__device__ void dec_gej_set_ge(ec_gej_t *r, const ec_ge_t *a) {
	r->infinity = a->infinity;
	r->x = a->x;
	r->y = a->y;
	dec_fe_set_int(&r->z, 1);
}

__device__ void dec_fe_normalize_weak(ec_fe_t *r) {
	uint32_t t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4],
		t5 = r->n[5], t6 = r->n[6], t7 = r->n[7], t8 = r->n[8], t9 = r->n[9];


	uint32_t x = t9 >> 22; t9 &= 0x03FFFFFUL;

	t0 += x * 0x3D1UL; t1 += (x << 6);

	t1 += (t0 >> 26); t0 &= 0x3FFFFFFUL;
	t2 += (t1 >> 26); t1 &= 0x3FFFFFFUL;
	t3 += (t2 >> 26); t2 &= 0x3FFFFFFUL;
	t4 += (t3 >> 26); t3 &= 0x3FFFFFFUL;
	t5 += (t4 >> 26); t4 &= 0x3FFFFFFUL;
	t6 += (t5 >> 26); t5 &= 0x3FFFFFFUL;
	t7 += (t6 >> 26); t6 &= 0x3FFFFFFUL;
	t8 += (t7 >> 26); t7 &= 0x3FFFFFFUL;
	t9 += (t8 >> 26); t8 &= 0x3FFFFFFUL;

	r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
	r->n[5] = t5; r->n[6] = t6; r->n[7] = t7; r->n[8] = t8; r->n[9] = t9;
}

__device__ void dec_fe_negate(ec_fe_t *r, const ec_fe_t *a, int m) {

	r->n[0] = 0x3FFFC2FUL * 2 * (m + 1) - a->n[0];

	//for (int j = 1; j<10; j++) r->n[j] = 0x3FFFFBFUL * 2 * (m + 1) - a->n[j];

	r->n[1] = 0x3FFFFBFUL * 2 * (m + 1) - a->n[1];
	r->n[2] = 0x3FFFFFFUL * 2 * (m + 1) - a->n[2];
	r->n[3] = 0x3FFFFFFUL * 2 * (m + 1) - a->n[3];
	r->n[4] = 0x3FFFFFFUL * 2 * (m + 1) - a->n[4];
	r->n[5] = 0x3FFFFFFUL * 2 * (m + 1) - a->n[5];
	r->n[6] = 0x3FFFFFFUL * 2 * (m + 1) - a->n[6];
	r->n[7] = 0x3FFFFFFUL * 2 * (m + 1) - a->n[7];
	r->n[8] = 0x3FFFFFFUL * 2 * (m + 1) - a->n[8];
	r->n[9] = 0x03FFFFFUL * 2 * (m + 1) - a->n[9];

}

__device__ void dec_fe_mul_int(ec_fe_t *r, int a)
{
	for (int j = 0; j<10; j++) r->n[j] *= a;
}

__device__ void dec_fe_add(ec_fe_t *r, const ec_fe_t *a)
{
	for (int j = 0; j < 10; j++) r->n[j] += a->n[j];
}

__device__ int dec_fe_normalizes_to_zero_var(ec_fe_t *r) {
	uint32_t t0, t1, t2, t3, t4, t5, t6, t7, t8, t9;
	uint32_t z0, z1;
	uint32_t x;

	t0 = r->n[0];
	t9 = r->n[9];

	/* Reduce t9 at the start so there will be at most a single carry from the first pass */
	x = t9 >> 22;

	/* The first pass ensures the magnitude is 1, ... */
	t0 += x * 0x3D1UL;

	/* z0 tracks a possible raw value of 0, z1 tracks a possible raw value of P */
	z0 = t0 & 0x3FFFFFFUL;
	z1 = z0 ^ 0x3D0UL;

	/* Fast return path should catch the majority of cases */
	if ((z0 != 0UL) & (z1 != 0x3FFFFFFUL)) {
		return 0;
	}

	t1 = r->n[1];
	t2 = r->n[2];
	t3 = r->n[3];
	t4 = r->n[4];
	t5 = r->n[5];
	t6 = r->n[6];
	t7 = r->n[7];
	t8 = r->n[8];

	t9 &= 0x03FFFFFUL;
	t1 += (x << 6);

	t1 += (t0 >> 26); t0 = z0;
	t2 += (t1 >> 26); t1 &= 0x3FFFFFFUL; z0 |= t1; z1 &= t1 ^ 0x40UL;
	t3 += (t2 >> 26); t2 &= 0x3FFFFFFUL; z0 |= t2; z1 &= t2;
	t4 += (t3 >> 26); t3 &= 0x3FFFFFFUL; z0 |= t3; z1 &= t3;
	t5 += (t4 >> 26); t4 &= 0x3FFFFFFUL; z0 |= t4; z1 &= t4;
	t6 += (t5 >> 26); t5 &= 0x3FFFFFFUL; z0 |= t5; z1 &= t5;
	t7 += (t6 >> 26); t6 &= 0x3FFFFFFUL; z0 |= t6; z1 &= t6;
	t8 += (t7 >> 26); t7 &= 0x3FFFFFFUL; z0 |= t7; z1 &= t7;
	t9 += (t8 >> 26); t8 &= 0x3FFFFFFUL; z0 |= t8; z1 &= t8;
	z0 |= t9; z1 &= t9 ^ 0x3C00000UL;

	return (z0 == 0) | (z1 == 0x3FFFFFFUL);
}

__device__ void dec_gej_double_var(ec_gej_t *r, const ec_gej_t *a, ec_fe_t *rzr) {
	/* Operations: 3 mul, 4 sqr, 0 normalize, 12 mul_int/add/negate */
	ec_fe_t t1, t2, t3, t4;
	/** For ec, 2Q is infinity if and only if Q is infinity. This is because if 2Q = infinity,
	*  Q must equal -Q, or that Q.y == -(Q.y), or Q.y is 0. For a point on y^2 = x^3 + 7 to have
	*  y=0, x^3 must be -7 mod p. However, -7 has no cube root mod p.
	*/
	r->infinity = a->infinity;
	if (r->infinity) {
		if (rzr) {
			dec_fe_set_int(rzr, 1);
		}
		return;
	}

	if (rzr) {
		*rzr = a->y;
		dec_fe_normalize_weak(rzr);
		dec_fe_mul_int(rzr, 2);
	}

	dec_fe_mul(&r->z, &a->z, &a->y);
	dec_fe_mul_int(&r->z, 2);       /* Z' = 2*Y*Z (2) */
	dec_fe_sqr(&t1, &a->x);
	dec_fe_mul_int(&t1, 3);         /* T1 = 3*X^2 (3) */
	dec_fe_sqr(&t2, &t1);           /* T2 = 9*X^4 (1) */
	dec_fe_sqr(&t3, &a->y);
	dec_fe_mul_int(&t3, 2);         /* T3 = 2*Y^2 (2) */
	dec_fe_sqr(&t4, &t3);
	dec_fe_mul_int(&t4, 2);         /* T4 = 8*Y^4 (2) */
	dec_fe_mul(&t3, &t3, &a->x);    /* T3 = 2*X*Y^2 (1) */
	r->x = t3;
	dec_fe_mul_int(&r->x, 4);       /* X' = 8*X*Y^2 (4) */
	dec_fe_negate(&r->x, &r->x, 4); /* X' = -8*X*Y^2 (5) */
	dec_fe_add(&r->x, &t2);         /* X' = 9*X^4 - 8*X*Y^2 (6) */
	dec_fe_negate(&t2, &t2, 1);     /* T2 = -9*X^4 (2) */
	dec_fe_mul_int(&t3, 6);         /* T3 = 12*X*Y^2 (6) */
	dec_fe_add(&t3, &t2);           /* T3 = 12*X*Y^2 - 9*X^4 (8) */
	dec_fe_mul(&r->y, &t1, &t3);    /* Y' = 36*X^3*Y^2 - 27*X^6 (1) */
	dec_fe_negate(&t2, &t4, 2);     /* T2 = -8*Y^4 (3) */
	dec_fe_add(&r->y, &t2);         /* Y' = 36*X^3*Y^2 - 27*X^6 - 8*Y^4 (4) */
}

__device__ void dec_gej_add_ge_var(ec_gej_t *r, const ec_gej_t *a, const ec_ge_t *b, ec_fe_t *rzr) {
	/* 8 mul, 3 sqr, 4 normalize, 12 mul_int/add/negate */
	ec_fe_t z12, u1, u2, s1, s2, h, i, i2, h2, h3, t;

	if (a->infinity) {
		dec_gej_set_ge(r, b);
		return;
	}

	if (b->infinity) {
		if (rzr) {
			dec_fe_set_int(rzr, 1);
		}
		*r = *a;
		return;
	}
	r->infinity = 0;

	dec_fe_sqr(&z12, &a->z);

	u1 = a->x; dec_fe_normalize_weak(&u1);
	dec_fe_mul(&u2, &b->x, &z12);
	s1 = a->y; dec_fe_normalize_weak(&s1);
	dec_fe_mul(&s2, &b->y, &z12);
	dec_fe_mul(&s2, &s2, &a->z);
	dec_fe_negate(&h, &u1, 1);
	dec_fe_add(&h, &u2);
	dec_fe_negate(&i, &s1, 1);
	dec_fe_add(&i, &s2);

	if (dec_fe_normalizes_to_zero_var(&h)) {
		if (dec_fe_normalizes_to_zero_var(&i)) {
			dec_gej_double_var(r, a, rzr);
		}
		else {
			if (rzr) {
				dec_fe_set_int(rzr, 0);
			}
			r->infinity = 1;
		}
		return;
	}
	dec_fe_sqr(&i2, &i);
	dec_fe_sqr(&h2, &h);
	dec_fe_mul(&h3, &h, &h2);
	if (rzr) {
		*rzr = h;
	}
	dec_fe_mul(&r->z, &a->z, &h);
	dec_fe_mul(&t, &u1, &h2);
	r->x = t;
	dec_fe_mul_int(&r->x, 2);
	dec_fe_add(&r->x, &h3);
	dec_fe_negate(&r->x, &r->x, 3);
	dec_fe_add(&r->x, &i2);
	dec_fe_negate(&r->y, &r->x, 5);
	dec_fe_add(&r->y, &t);
	dec_fe_mul(&r->y, &r->y, &i);
	dec_fe_mul(&h3, &h3, &s1);
	dec_fe_negate(&h3, &h3, 1);
	dec_fe_add(&r->y, &h3);
}

#endif