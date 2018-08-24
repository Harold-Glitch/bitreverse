#pragma once
#include <stdint.h>
#ifndef _WIN32
#include <cpuid.h>
#endif

#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif


#define APP_NAME "bitreverse"

#define ALT_DATA_FILE "box.alt"
#define ETH_DATA_FILE "box.eth"
#define SEC_DATA_FILE "box.sec"

#define MAX_DEVICES 64
#define THREADS_PER_BLOCK 256

#define KEY_LENGTH_BITS 256
#define KEY_LENGTH_BYTE (KEY_LENGTH_BITS / 8)
#define SECONDS_ADJUST_SLEEP 3


#define FILL_FACTOR_HASHTABLE 8

#ifndef EC_TAG_PUBKEY_EVEN
#define EC_TAG_PUBKEY_EVEN 0x02
#endif

#ifndef EC_TAG_PUBKEY_ODD
#define EC_TAG_PUBKEY_ODD 0x03
#endif


//#define KEY_PER_THREAD 32

uint32_t load(std::string fname, uint8_t **ppaddr, uint8_t **pphash);
std::string formatThousands(uint64_t x, uint64_t div);
uint32_t performance();

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}

	return;
}

/*
* 32-bit integer manipulation macros (little endian)
*/
#ifndef GET_UINT32_LE
#define GET_UINT32_LE(n,b,i)                            \
{                                                       \
    (n) = ( (uint32_t) (b)[(i)    ]       )             \
        | ( (uint32_t) (b)[(i) + 1] <<  8 )             \
        | ( (uint32_t) (b)[(i) + 2] << 16 )             \
        | ( (uint32_t) (b)[(i) + 3] << 24 );            \
}
#endif

#ifndef PUT_UINT32_LE
#define PUT_UINT32_LE(n,b,i)                                    \
{                                                               \
    (b)[(i)    ] = (uint8_t) ( ( (n)       ) & 0xFF );    \
    (b)[(i) + 1] = (uint8_t) ( ( (n) >>  8 ) & 0xFF );    \
    (b)[(i) + 2] = (uint8_t) ( ( (n) >> 16 ) & 0xFF );    \
    (b)[(i) + 3] = (uint8_t) ( ( (n) >> 24 ) & 0xFF );    \
}
#endif


typedef struct {
	uint32_t n[10];
} ec_fe_t;

/** A group element of the ec curve, in affine coordinates. */
typedef struct {
	ec_fe_t x;
	ec_fe_t y;
	int infinity; // whether this represents the point at infinity
} ec_ge_t;

typedef struct {
	ec_fe_t x; // actual X: x/z^2
	ec_fe_t y; // actual Y: y/z^3
	ec_fe_t z;
	int infinity; // whether this represents the point at infinity
} ec_gej_t;

typedef struct {
	int	device;
	uint32_t header;
	uint64_t block_size;
	uint32_t key_per_thread;
} arguments_t;

typedef struct {
	uint32_t fan_speed;
	uint32_t temperature;
	uint32_t watt;
} monitor_t;

typedef struct {
	uint64_t processed;
	double speed;
	bool ready;
	uint32_t fan_speed;
	uint32_t temperature;
	uint32_t watt;
} dx_device_t;

typedef struct {
	dx_device_t device[MAX_DEVICES];
	uint32_t count;
	uint64_t processed;
	double speed;
	bool ready;
} dx_process_t;



#define FROMHEX_MAXLEN 256

inline void fromhex(const char *str, uint8_t *buf, size_t len)
{
	for (size_t i = 0; i < len; i++) {
		uint8_t c = 0;
		if (str[i * 2] >= '0' && str[i * 2] <= '9') c += (str[i * 2] - '0') << 4;
		if ((str[i * 2] & ~0x20) >= 'A' && (str[i * 2] & ~0x20) <= 'F') c += (10 + (str[i * 2] & ~0x20) - 'A') << 4;
		if (str[i * 2 + 1] >= '0' && str[i * 2 + 1] <= '9') c += (str[i * 2 + 1] - '0');
		if ((str[i * 2 + 1] & ~0x20) >= 'A' && (str[i * 2 + 1] & ~0x20) <= 'F') c += (10 + (str[i * 2 + 1] & ~0x20) - 'A');
		buf[i] = c;
	}
}

#undef max
#undef min


