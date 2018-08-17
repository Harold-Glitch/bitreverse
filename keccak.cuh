#pragma once
#ifndef _KECCAK_CUH
#define _KECCAK_CUH

#include <stdint.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ptx.cuh"

typedef union {
	uint8_t b[200];
	uint64_t q[25];
	uint32_t d[50];
} d_ethhash;

__device__ const uint64_t d_keccakf_rndc[24] = {
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a,	0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008a,	0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089,	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	0x000000000000800a, 0x800000008000000a, 0x8000000080008081,	0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

__device__ const uint64_t d_keccakf_rotc[24] = {
	1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

__device__ const int d_keccakf_piln[24] = {
	10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

__device__ uint64_t d_rotl64(uint64_t var, uint64_t hops)
{
	return (var << hops) | (var >> (64 - hops));
}

__device__ void k_hash_eth(uint8_t *pubKeyX, uint8_t *pubKeyY, uint64_t *pout)
{
	d_ethhash e;

	memset(&e, 0, sizeof(d_ethhash));

	memcpy(e.b, pubKeyX, 32);
	memcpy(e.b + 32, pubKeyY, 32);

	uint64_t * st = e.q;
	e.d[16] ^= 0x01;
	e.d[33] ^= 0x80000000;

	// variables
	int i, j, r;
	uint64_t t, bc[5];

	// actual iteration
	for (r = 0; r < 24; r++) {
		// Theta - unrolled
		bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
		bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
		bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
		bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
		bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

		for (i = 0; i < 5; i++) {
			t = bc[(i + 4) % 5] ^ d_rotl64(bc[(i + 1) % 5], (uint64_t)1);

			st[i] ^= t;
			st[i + 5] ^= t;
			st[i + 10] ^= t;
			st[i + 15] ^= t;
			st[i + 20] ^= t;
		}

		// Rho Pi
		t = st[1];
		for (i = 0; i < 24; i++) {
			j = d_keccakf_piln[i];
			bc[0] = st[j];

			st[j] = d_rotl64(t, d_keccakf_rotc[i]);
			t = bc[0];
		}

		//  Chi
		for (j = 0; j < 25; j += 5) {
			bc[0] = st[j + 0];
			bc[1] = st[j + 1];
			bc[2] = st[j + 2];
			bc[3] = st[j + 3];
			bc[4] = st[j + 4];

			st[j + 0] ^= (~bc[1]) & bc[2];
			st[j + 1] ^= (~bc[2]) & bc[3];
			st[j + 2] ^= (~bc[3]) & bc[4];
			st[j + 3] ^= (~bc[4]) & bc[0];
			st[j + 4] ^= (~bc[0]) & bc[1];
		}

		//  Iota
		st[0] ^= d_keccakf_rndc[r];
	}

	memcpy(pout, e.b + 12, 20);
}

#endif
