#include <string>
#include <cstring>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
namespace fs = boost::filesystem;

#ifndef _CRYPTO_UTIL_H
#define _CRYPTO_UTIL_H

#define _CRT_SECURE_NO_WARNINGS

template<class = void>
void
cpuid(
	std::uint32_t id,
	std::uint32_t& eax,
	std::uint32_t& ebx,
	std::uint32_t& ecx,
	std::uint32_t& edx)
{
#ifdef _WIN32
	int regs[4];
	__cpuid(regs, id);
	eax = regs[0];
	ebx = regs[1];
	ecx = regs[2];
	edx = regs[3];
#else
	__get_cpuid(id, &eax, &ebx, &ecx, &edx);
#endif
}


struct entropy_t { uint32_t process_id; 
	uint64_t thread_id; 
	uint64_t time_since_epoch; 
	cudaDeviceProp prop; 
	uint32_t watt;
	uint32_t temperature;
	uint32_t fan;
	uint32_t cpuid[8]; 
	double kps; 
	uint32_t performance; 
	uint64_t counter;  
	uint8_t rand[32];
	uint8_t state[32]; 
	uint8_t sha[32];
};

void seedkey(entropy_t &entropy);
void privkey(uint32_t header, entropy_t &entropy, uint8_t *h_raw);
std::string sha512(const void* dat, size_t len);

std::string peek_id(fs::path filepath);

typedef union {
	uint8_t b[200];
	uint64_t q[25];
	uint32_t d[50];
} ethhash;

void sha3_keccakf(ethhash * const pHash);

class SHA512 {

protected:
	typedef unsigned char uint8;
	typedef unsigned long uint32;
	typedef unsigned long long uint64;
	const static uint64 sha512_k[];
	static const uint32_t SHA384_512_BLOCK_SIZE = (1024 / 8);

public:
	void init();
	void update(const unsigned char *message, uint32_t len);
	void final(unsigned char *digest);
	static const uint32_t DIGEST_SIZE = (512 / 8);

protected:
	void transform(const unsigned char *message, uint32_t block_nb);
	uint32_t m_tot_len;
	uint32_t m_len;
	unsigned char m_block[2 * SHA384_512_BLOCK_SIZE];
	uint64 m_h[8];
};

#define SHA2_SHFR(x, n)    (x >> n)
#define SHA2_ROTR(x, n)   ((x >> n) | (x << ((sizeof(x) << 3) - n)))
#define SHA2_ROTL(x, n)   ((x << n) | (x >> ((sizeof(x) << 3) - n)))
#define SHA2_CH(x, y, z)  ((x & y) ^ (~x & z))
#define SHA2_MAJ(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define SHA512_F1(x) (SHA2_ROTR(x, 28) ^ SHA2_ROTR(x, 34) ^ SHA2_ROTR(x, 39))
#define SHA512_F2(x) (SHA2_ROTR(x, 14) ^ SHA2_ROTR(x, 18) ^ SHA2_ROTR(x, 41))
#define SHA512_F3(x) (SHA2_ROTR(x,  1) ^ SHA2_ROTR(x,  8) ^ SHA2_SHFR(x,  7))
#define SHA512_F4(x) (SHA2_ROTR(x, 19) ^ SHA2_ROTR(x, 61) ^ SHA2_SHFR(x,  6))
#define SHA2_UNPACK32(x, str)                 \
{                                             \
    *((str) + 3) = (uint8) ((x)      );       \
    *((str) + 2) = (uint8) ((x) >>  8);       \
    *((str) + 1) = (uint8) ((x) >> 16);       \
    *((str) + 0) = (uint8) ((x) >> 24);       \
}
#define SHA2_UNPACK64(x, str)                 \
{                                             \
    *((str) + 7) = (uint8) ((x)      );       \
    *((str) + 6) = (uint8) ((x) >>  8);       \
    *((str) + 5) = (uint8) ((x) >> 16);       \
    *((str) + 4) = (uint8) ((x) >> 24);       \
    *((str) + 3) = (uint8) ((x) >> 32);       \
    *((str) + 2) = (uint8) ((x) >> 40);       \
    *((str) + 1) = (uint8) ((x) >> 48);       \
    *((str) + 0) = (uint8) ((x) >> 56);       \
}
#define SHA2_PACK64(str, x)                   \
{                                             \
    *(x) =   ((uint64) *((str) + 7)      )    \
           | ((uint64) *((str) + 6) <<  8)    \
           | ((uint64) *((str) + 5) << 16)    \
           | ((uint64) *((str) + 4) << 24)    \
           | ((uint64) *((str) + 3) << 32)    \
           | ((uint64) *((str) + 2) << 40)    \
           | ((uint64) *((str) + 1) << 48)    \
           | ((uint64) *((str) + 0) << 56);   \
}

#endif
