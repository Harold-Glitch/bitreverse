//
// BITREVERSE - CUDA, MultiGPU, Bitcoin Altcoins and Ethereum address collision finder
// Harold-Glitch 2017-2018 - harold.glitch@gmail.com
//
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <list>
#include <string>
#include <fstream>
#include <chrono>
#include <random>
#include <ctime>
#include <thread>
#include <mutex>
#include <iostream>
#include <algorithm>
#include <iomanip>

#include <signal.h>

#ifdef _WIN32
#include <process.h>
#endif

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

namespace fs = boost::filesystem;

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <device_functions.h>
#include <device_launch_parameters.h>

#include "ptx.cuh"
#include "secp256k1.cuh"

#include "secp256k1.h"

#include "bitreverse.h"
#include "web.h"
#include "rng.h"
#include "sha256.cuh"
#include "keccak.cuh"
#include "ripemd160.cuh"

#include "INIReader.h"

std::vector<ec_ge_t> _ecVector;
size_t _sec_filesize;

auto _now = std::chrono::high_resolution_clock::now();

// needed to know which block header to mine 
std::string _url("http://bitreverse.io/");

std::mutex _mutex;
std::string _id;

dx_process_t dx_process = { 0 };

std::vector<std::list<monitor_t>> _gpu_monitor;
fs::path _res_data_path;

uint8_t *_addr_btc = nullptr;
uint8_t *_hash_addr_btc = nullptr;
uint32_t _addr_btc_count = 0;
uint32_t _hash_btc_count = 0;

uint8_t *_addr_eth = nullptr;
uint8_t *_hash_addr_eth = nullptr;
uint32_t _addr_eth_count = 0;
uint32_t _hash_eth_count = 0;

std::string get_version(std::string id)
{
	std::string url(_url);
	url.append("full.php?id=");
	url.append(id);

	http::Request request(url);
	http::Response response = request.send("GET");

	std::string str(response.body.begin(), response.body.end());

	return str;
}

uint32_t which_block(std::string id)
{
	std::string url(_url);
	url.append("which_block.php?id=");
	url.append(id);

	http::Request request(url);
	http::Response response = request.send("GET");

	std::string str(response.body.begin(), response.body.end());

	uint32_t val = 0;
	string::size_type loc = str.find("\"block\":\"", 0);
	if (loc != string::npos) {
		std::string hex = str.substr(loc + 9, 8);
		fromhex(hex.c_str(), (unsigned char *)&val, sizeof(val));
	}

	return val;
}

// ts_: Thread Safe functions
void ts_entropy(entropy_t *entropy)
{
	std::lock_guard<std::mutex> lock(_mutex);
	entropy->watt += dx_process.device[0].watt;
	entropy->fan += dx_process.device[0].fan_speed;
	entropy->temperature += dx_process.device[0].temperature;
}

void ts_increment(int device, uint64_t processed, double elapsed)
{
	std::lock_guard<std::mutex> lock(_mutex);

	if (dx_process.device[device].ready == false)
		dx_process.device[device].ready = true;

	dx_process.device[device].processed += processed;
	dx_process.device[device].speed = processed / elapsed;

	dx_process.processed += processed;

	dx_process.ready = true;
	dx_process.speed = 0;
	for (int n = 0; n < dx_process.count; n++) {
		dx_process.speed += dx_process.device[n].speed;

		if (dx_process.device[n].ready == false)
			dx_process.ready = false;
	}
}

vector<string> ts_read(std::string file)
{
	ifstream stream(file);
	string line;
	vector<string> lines;
	while (getline(stream, line)) {
		lines.push_back(line);
	}
	stream.close();
	return lines;
}

void ts_append(std::string file, std::string line)
{
	std::lock_guard<std::mutex> lock(_mutex);

	std::time_t t = std::time(0);   // get time now
	std::tm* now = std::localtime(&t);
	ostringstream os;

	FILE *fp = fopen(file.c_str(), "a+");
	if (fp != NULL) {
		fprintf(fp, "\"%d-%02d-%02d\",%s\n", (now->tm_year + 1900), (now->tm_mon + 1), now->tm_mday, line.c_str());
		fclose(fp);
	}
}


// ws_: WebService REST request/response
void wsDevice(WPP::Request* req, WPP::Response* res)
{
	nvmlReturn_t result;
	nvmlDevice_t device;
	unsigned int fan_speed;
	unsigned int temperature;
	unsigned int watt;
	int nDevices;

	cudaGetDeviceCount(&nDevices);

	if (_gpu_monitor.size() == 0) {
		for (int i = 0; i < nDevices; i++) {
			_gpu_monitor.push_back(list<monitor_t>());
		}
	}

	result = nvmlInit();
	if (NVML_SUCCESS == result) {

		for (int i = 0; i < nDevices; i++) {

			result = nvmlDeviceGetHandleByIndex(i, &device);
			nvmlDeviceGetFanSpeed(device, &fan_speed);
			nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
			nvmlDeviceGetPowerUsage(device, &watt);

			monitor_t monitor;
			monitor.fan_speed = fan_speed;
			monitor.temperature = temperature;
			monitor.watt = watt;

			_gpu_monitor[i].push_back(monitor);

			dx_process.device[i].fan_speed = fan_speed;
			dx_process.device[i].temperature = temperature;
			dx_process.device[i].watt = watt;
		}

		nvmlShutdown();
	}

	for (int i = 0; i < nDevices; i++)
		if (_gpu_monitor[i].size() > 16)
			_gpu_monitor[i].pop_front();

	res->body << "{\"data\": { \"series\": [";
	char sz[4049];
	for (int i = 0; i < nDevices; i++) {

		if (i > 0) res->body << ",";

		res->body << "{ \"name\": \"" << i << "\", \"data\": [";
		int v = 0;
		for (monitor_t val : _gpu_monitor[i]) {
			if (v > 0) res->body << ",";
			sprintf(sz, "%d", val.temperature);
			res->body << sz;
			v++;
		}
		res->body << "]}";
	}
	res->body << "]}}";

}

void wsUser(WPP::Request* req, WPP::Response* res)
{
	std::string address;
	fs::path res_path(_res_data_path);
	res_path /= "address.txt";

	std::map<std::string, std::string>::iterator iter;
	bool arg = false;
	for (iter = req->query.begin(); iter != req->query.end(); ++iter)
		if (iter->first.compare("address") == 0) {
			arg = true;
			address = string(iter->second);

			FILE *fp = fopen(res_path.string().c_str(), "w");
			if (fp != NULL) {
				fprintf(fp, "%s\n", address.c_str());
				fclose(fp);
			}
		}

	if (boost::filesystem::exists(res_path)) {

		ifstream stream(res_path.string());
		string line;
		while (getline(stream, line)) {
			address = line;
		}
		stream.close();
	}

	res->body << "{ \"id\": \"" << _id << "\", \"address\": \"" << address << "\"}";
}

void wsFound(WPP::Request* req, WPP::Response* res)
{
	fs::path res_path(_res_data_path);
	res_path /= "found.txt";
	char sz[4049];

	vector<string> vs = ts_read(res_path.string());

	res->body << "{ \"key\": [";

	for (int n = 0; n < vs.size(); n++) {
		char c = ',';
		if (n == vs.size() - 1) c = ' ';
		sprintf(sz, "[%s]%c ", vs[n].c_str(), c);

		res->body << sz;
	}

	res->body << "]}";
}

void wsVersion(WPP::Request* req, WPP::Response* res)
{
	res->body << get_version(_id);
}

void wsStatus(WPP::Request* req, WPP::Response* res)
{
	std::lock_guard<std::mutex> lock(_mutex);

	int nDevices;
	char sz[4049];

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - _now;

	typedef std::chrono::duration<long double> MySecondTick;
	MySecondTick up_sec(elapsed);
	typedef std::chrono::duration<double, std::ratio<60>> MyMinuteTick;
	MyMinuteTick up_min(elapsed);
	typedef std::chrono::duration<double, std::ratio<60 * 60>> MyHourTick;
	MyHourTick up_hour(elapsed);

	uint32_t uup_sec = ((uint32_t)up_sec.count()) % 60;
	uint32_t uup_min = ((uint32_t)up_min.count()) % 60;
	uint32_t uup_hour = ((uint32_t)up_hour.count());

#ifdef _WIN32
	TCHAR  infoBuf[256];
	DWORD  bufCharCount = 256;
	GetComputerName(infoBuf, &bufCharCount);
	res->body << "{ \"host\": " << "\"" << infoBuf << "\"" << ", ";
#else
	char hostname[HOST_NAME_MAX];
	gethostname(hostname, HOST_NAME_MAX);
	res->body << "{ \"host\": " << "\"" << hostname << "\"" << ", ";
#endif

	res->body << "\"up\": " << "\"" << std::fixed << std::setfill('0') << uup_hour << ":" << std::fixed << std::setw(2) << std::setfill('0')
		<< uup_min << ":" << std::fixed << std::setw(2) << std::setfill('0') << uup_sec << "\"" << ", ";
	res->body << "\"bitcoin\": " << "\"" << formatThousands(_addr_btc_count, 1) << "\"" << ", ";
	res->body << "\"ethereum\": " << "\"" << formatThousands(_addr_eth_count, 1) << "\"" << ", ";
	res->body << "\"processed\": " << "\"" << formatThousands(dx_process.processed, 1) << "\"" << ", ";
	res->body << "\"speed\": " << "\"" << formatThousands(dx_process.speed, 1) << "\"" << ", ";
	res->body << "\"device\": [ ";

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		char c = ',';
		if (i == nDevices - 1) c = ' ';
		sprintf(sz, "[\"%d\", \"%s\", \"%d.%d\", \"%.2f\", \"%d\", \"%.0f\"]%c", i, prop.name, prop.major, prop.minor, (float)dx_process.device[i].watt / 1000, dx_process.device[i].fan_speed, dx_process.device[i].speed, c);
		res->body << sz;
	}

	res->body << "]}";
}

void wsHome(WPP::Request* req, WPP::Response* res) {

	res->body << "HELLO" << "\n";
}

void wsRestart(WPP::Request* req, WPP::Response* res) {
	res->body << "</shutdown>";
}

struct ws_thread
{
	int port = 0;

	ws_thread(int port) : port(port) { }

	void run()
	{		
		printf("$SERVER-I-REST, Server started on port %d\n", port);

		try {
			WPP::Server server;

			server.get("/restart", &wsRestart);
			server.get("/found", &wsFound);
			server.get("/device", &wsDevice);
			server.get("/status", &wsStatus);
			server.get("/user", &wsUser);
			server.get("/version", &wsVersion);
			server.get("/", &wsHome);

			server.start(port);
		}
		catch (WPP::Exception e) {			
			std::cerr << "$SERVER-E-REST " << e.what() << std::endl;
			exit(0);
		}
	}
};

__device__ const uint8_t *d_fromhex(const char *str)
{
	uint8_t buf[FROMHEX_MAXLEN];
	size_t len = 32;
	//if (len > FROMHEX_MAXLEN) len = FROMHEX_MAXLEN;
	for (size_t i = 0; i < len; i++) {
		uint8_t c = 0;
		if (str[i * 2] >= '0' && str[i * 2] <= '9') c += (str[i * 2] - '0') << 4;
		if ((str[i * 2] & ~0x20) >= 'A' && (str[i * 2] & ~0x20) <= 'F') c += (10 + (str[i * 2] & ~0x20) - 'A') << 4;
		if (str[i * 2 + 1] >= '0' && str[i * 2 + 1] <= '9') c += (str[i * 2 + 1] - '0');
		if ((str[i * 2 + 1] & ~0x20) >= 'A' && (str[i * 2 + 1] & ~0x20) <= 'F') c += (10 + (str[i * 2 + 1] & ~0x20) - 'A');
		buf[i] = c;
	}
	return buf;
}

inline int pid()
{
#ifdef _WIN32
	return _getpid();
#else
	return getpid();
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

__constant__ uint32_t _INC_X[8];
__constant__ uint32_t _INC_Y[8];

__device__ int dnAddressFound;
__device__ int dnIteration;
__device__ int dnIterationFound;
__device__ int dnAddressType;

__global__ void k_generate(ec_ge_t *dec, uint8_t *ppriKey, uint8_t *ppubKeyX, uint8_t *ppubKeyY)
{
	uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

	if (dnAddressFound != -1)
		return;

	uint32_t precomputedOffset[16] = { 0, 65536, 131072, 196608, 262144, 327680, 393216, 458752, 524288, 589824, 655360, 720896, 786432, 851968, 917504, 983040 };

	uint8_t u8[32];
	ec_ge_t p;
	ec_gej_t pj;
	pj.infinity = 1;

	uint8_t *priKey = ppriKey + (id * 32);
	uint8_t *pubKeyX = ppubKeyX + (id * 32);
	uint8_t *pubKeyY = ppubKeyY + (id * 32); 
	
	for (int n = 0; n < 32; n++)
		u8[n] = priKey[n];

	for (int n = 0; n < 16; n++)
		dec_gej_add_ge_var(&pj, &pj, &dec[precomputedOffset[n] + (u8[30 - (n * 2)] << 8) + u8[31 - (n * 2)]], NULL);

	dec_ge_set_gej(&p, &pj);
	
	dec_fe_normalize_var(&p.x);
	dec_fe_normalize_var(&p.y);

	dec_fe_get_b32(pubKeyX, &p.x);
	dec_fe_get_b32(pubKeyY, &p.y);
}

__global__ void k_next(int keyperthread, uint32_t *px, uint32_t *py)
{
	if (dnAddressFound != -1)
		return;

	uint32_t inverse[8] = { 0,0,0,0,0,0,0,1 };
	uint32_t x[8];
	uint32_t y[8];
	
	uint32_t chain[8 * 128];

	for (int i = 0; i < keyperthread; i++) {

		readInt(px, i, x);
		readInt(py, i, y);

		beginBatchAdd(_INC_X, px, chain, i, inverse);
	}

	doBatchInverse(inverse);

	for (int i = keyperthread - 1; i >= 0; i--) {

		completeBatchAdd(_INC_X, _INC_Y, px, py, i, chain, inverse, x, y);
		writeInt(px, i, x);	
		writeInt(py, i, y);
	}
}


// Kernel code
__global__ void k_compare(uint64_t* pAddress, uint8_t* pExist, int nAddressCount, uint64_t* pHash64, int type)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (dnAddressFound != -1)
		return;

	uint32_t *p32 = (uint32_t *)pHash64;
	p32 += id * 6;

	uint32_t nStopCount = (nAddressCount * FILL_FACTOR_HASHTABLE);

	uint32_t base = 0;
	if (pExist[(p32[0] % nStopCount)] != 0) base += nStopCount; else return;
	if (pExist[base + (p32[1] % nStopCount)] != 0) base += nStopCount; else return;
	if (pExist[base + (p32[2] % nStopCount)] != 0) base += nStopCount; else return;
	if (pExist[base + (p32[3] % nStopCount)] != 0) base += nStopCount; else return;
	if (pExist[base + (p32[4] % nStopCount)] == 0) return;

	uint64_t hash = pHash64[(id * 3)];
	int64_t compare;

	for (int high = nAddressCount - 1, low = 0, i = high >> 1; high >= low; i = low + ((high - low) >> 1)) {

		compare = hash - pAddress[(i * 3)];

		if (compare == 0) {
			compare = pHash64[(id * 3) + 1] - pAddress[(i * 3) + 1];

			if (compare == 0) {
				compare = (pHash64[(id * 3) + 2] & 0x00000000ffffffff) - (pAddress[(i * 3) + 2] & 0x00000000ffffffff);

				if (compare == 0) {
					//printf(" [%llx %llx] ", pAddress[(i * 3)-3], pHash64[(id * 3)-3]);
					atomicMax(&dnIterationFound, dnIteration);
					atomicMax(&dnAddressFound, id); high = -1;
					atomicMax(&dnAddressType, type); 
				}
			}
		}

		low = (compare > 0) ? (i + 1) : low;
		high = (compare < 0) ? (i - 1) : high;
	}
}


__global__ void k_digest(uint8_t *ppubKeyX, uint8_t *ppubKeyY, uint64_t *pdigest_u, uint64_t *pdigest_c, uint64_t *pdigest_h)
{
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
	
	uint32_t x[8];
	uint32_t y[8];

	uint32_t hash_u[8];
	uint32_t hash_c[8];

	// Uncompressed	-> pdigest_u
	memcpy(x, ppubKeyX + (id * 32), 32);
	memcpy(y, ppubKeyY + (id * 32), 32);
	
	sha256PublicKey(x, y, hash_u);

	for (int j = 0; j < 8; j++)
		hash_u[j] = endian(hash_u[j]);

	ripemd160(hash_u, (uint32_t *)(&pdigest_u[(id * 3)]));
	
	// Compressed -> pdigest_c
	sha256PublicKeyCompressed(x, y, hash_c);

	for (int j = 0; j < 8; j++)
		hash_c[j] = endian(hash_c[j]);

	ripemd160(hash_c, (uint32_t *)(&pdigest_c[(id * 3)]));
	
	// Ethereum	-> pdigest_h
	uint32_t *pubKeyX = (uint32_t *)ppubKeyX + (id * 8);
	uint32_t *pubKeyY = (uint32_t *)ppubKeyY + (id * 8);

	for (int i = 0; i < 8; i++) {
		x[i] = endian(pubKeyX[i]);
		y[i] = endian(pubKeyY[i]);
	}

	k_hash_eth((uint8_t *)x, (uint8_t *)y, &pdigest_h[(id * 3)]);
}

cudaError_t setIncrementorPoint(const secp256k1::uint256 &x, const secp256k1::uint256 &y)
{
	uint32_t xWords[8];
	uint32_t yWords[8];

	x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
	y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

	cudaError_t err = cudaMemcpyToSymbol(_INC_X, xWords, sizeof(uint32_t) * 8);
	if (err) {
		return err;
	}

	return cudaMemcpyToSymbol(_INC_Y, yWords, sizeof(uint32_t) * 8);
}


void cracker(arguments_t ta)
{	
	CudaSafeCall(cudaSetDevice(ta.device));
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	uint32_t BLOCK_SIZE = (ta.block_size * ta.block_size);
	entropy_t entropy = { 0 };

	// Add CUDA device properties to entropy
	cudaGetDeviceProperties(&entropy.prop, ta.device);	

	std::string device_name = std::string(entropy.prop.name);	
	printf("$BITREVERSE-I-DEVINIT, initializing device %d - %s\n", ta.device, device_name.c_str());

	uint64_t *d_address_btc = 0;
	uint64_t *d_address_eth = 0;

	uint8_t *d_hash_address_btc = 0;
	uint8_t *d_hash_address_eth = 0;

	uint64_t *d_digest_u = 0;
	uint64_t *d_digest_c = 0;
	uint64_t *d_digest_h = 0;

	uint8_t *d_publicX = 0;
	uint8_t *d_publicY = 0;

	uint8_t *d_private = 0;

	uint32_t cpu_performance = performance();

	ec_ge_t *ecPrecomputed;

	char sz[128];	
	char key[128];	
	
	uint8_t *h_private;
	cudaMallocHost((void**)&h_private, BLOCK_SIZE * 32);

	uint8_t *h_raw;
	cudaMallocHost((void**)&h_raw, sizeof(uint8_t) * 32);
	
	CudaSafeCall(cudaMalloc((void**)&d_private, BLOCK_SIZE * 32 ));
	CudaSafeCall(cudaMalloc((void**)&d_publicX, BLOCK_SIZE * 32 ));
	CudaSafeCall(cudaMalloc((void**)&d_publicY, BLOCK_SIZE * 32 ));

	CudaSafeCall(cudaMalloc((void**)&d_digest_u, BLOCK_SIZE * (sizeof(uint64_t) * 3)));
	CudaSafeCall(cudaMalloc((void**)&d_digest_c, BLOCK_SIZE * (sizeof(uint64_t) * 3)));
	CudaSafeCall(cudaMalloc((void**)&d_digest_h, BLOCK_SIZE * (sizeof(uint64_t) * 3)));

	_mutex.lock();

	CudaSafeCall(cudaMalloc((void**)&ecPrecomputed, _sec_filesize));
	CudaSafeCall(cudaMemcpy(ecPrecomputed, _ecVector.data(), _sec_filesize, cudaMemcpyHostToDevice));

	uint32_t addr_btc_count = _addr_btc_count;
	uint32_t hash_btc_count = _hash_btc_count;
	uint32_t addr_eth_count = _addr_eth_count;
	uint32_t hash_eth_count = _hash_eth_count;

	CudaSafeCall(cudaMalloc((void**)&d_address_btc, sizeof(uint64_t) * 3 * addr_btc_count));
	CudaSafeCall(cudaMemcpy(d_address_btc, _addr_btc, sizeof(uint64_t) * 3 * addr_btc_count, cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMalloc((void**)&d_hash_address_btc, sizeof(uint8_t) * 5 * hash_btc_count));
	CudaSafeCall(cudaMemcpy(d_hash_address_btc, _hash_addr_btc, sizeof(uint8_t) * 5 * hash_btc_count, cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMalloc((void**)&d_address_eth, sizeof(uint64_t) * 3 * addr_eth_count));
	CudaSafeCall(cudaMemcpy(d_address_eth, _addr_eth, sizeof(uint64_t) * 3 * addr_eth_count, cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMalloc((void**)&d_hash_address_eth, sizeof(uint8_t) * 5 * hash_eth_count));
	CudaSafeCall(cudaMemcpy(d_hash_address_eth, _hash_addr_eth, sizeof(uint8_t) * 5 * hash_eth_count, cudaMemcpyHostToDevice));	

	_mutex.unlock();

	int keyperthread = ta.key_per_thread;
	dim3 griddim, griddim2, blockdim;
	blockdim = dim3(THREADS_PER_BLOCK, 1, 1);
	griddim = dim3(BLOCK_SIZE / blockdim.x, 1, 1);
	griddim2 = dim3(BLOCK_SIZE / keyperthread / blockdim.x, 1, 1);

	float sleeptime = 0;
	double kps = 0;
	char ckps[1024];
	
	// Add CPU benchmark to entropy
	entropy.performance = performance();
	entropy.process_id = (uint32_t)pid();

	int inc_loop = 256;
	// Set the incrementor
	uint64_t block_offset = (uint64_t)1;
	uint64_t offset = 1;

	secp256k1::ecpoint g = secp256k1::G();
	secp256k1::ecpoint p = secp256k1::multiplyPoint(secp256k1::uint256((unsigned long long)block_offset), g);
	setIncrementorPoint(p.x, p.y);

	uint64_t processed = 0;

	CudaSafeCall(cudaMemcpyToSymbol((const void *)&dnKEY_PER_THREAD, &keyperthread, sizeof(int), 0, cudaMemcpyHostToDevice));

	while (true) {

		// Add key per second to entropy 
		entropy.kps = kps;
		ts_entropy(&entropy);
		seedkey(entropy);
		
		printf("$BITREVERSE-I-SEEDING, Seeding new batch for device %d - kt:%d bs:%d\n", ta.device, keyperthread, BLOCK_SIZE);

		for (int n = 0; n < BLOCK_SIZE; n++) {

			// Markers created inside the address database for testing purpose 
			// "2B9C8EE6D177F7DE6E147AB1BE07C9B23C494322E7F1B941B00EE0C580C477A9" // BTC uncompressed
			// "95CA96E176DB1188EB85C271FAA2186023F6A898322D27BAB38418F8F3F4CC17" // BTC compressed
			// "D57A16054DDEF68836EEDA78AADC76E9B1D5124EAA052E6B4FBACA95C7217D7E" // Dash
			// "28B3C1358122FC30D5A6100FF0455B3E0DB419855D0DFAE28CFBF1F798C7CEFA" // LTC
			// "208065A247EDBE5DF4D86FBDC0171303F23A76961BE9F6013850DD2BDC759BBB" // ETH

			//if(n == 10000) 			
			//fromhex("2B9C8EE6D177F7DE6E147AB1BE07C9B23C494322E7F1B941B00EE0C580C477A9", h_raw, 32); 
			//else
			privkey(ta.header, entropy, h_raw);
			
			secp256k1::uint256 k(h_raw, secp256k1::uint256::BigEndian);
			k.toByteArray(h_private + (n * KEY_LENGTH_BYTE));
		}
		
		CudaSafeCall(cudaMemcpy(d_private, h_private, BLOCK_SIZE * 32, cudaMemcpyHostToDevice));

		uint64_t remaining = 0x4000000000;
		uint32_t count = BLOCK_SIZE;
		uint32_t chunks = 512;		

		int nAddressFound = -1;
		int nAddressType = -1;
		int nIteration = 0;
		int nIterationFound = -1;
		CudaSafeCall(cudaMemcpyToSymbol((const void *)&dnAddressFound, &nAddressFound, sizeof(int), 0, cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpyToSymbol((const void *)&dnAddressType, &nAddressType, sizeof(int), 0, cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpyToSymbol((const void *)&dnIterationFound, &nIterationFound, sizeof(int), 0, cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpyToSymbol((const void *)&dnIteration, &nIteration, sizeof(int), 0, cudaMemcpyHostToDevice));

		// generate
		for (int chunk = 0 ; chunk < chunks ; chunk++) {

			auto start = std::chrono::high_resolution_clock::now();

			int nAddressFound = -1;
			CudaSafeCall(cudaMemcpyToSymbol((const void *)&dnAddressFound, &nAddressFound, sizeof(int), 0, cudaMemcpyHostToDevice));

			for (int n = 0; n < inc_loop ; n++) {

				if (chunk == 0 && n == 0) {
					k_generate << < griddim, blockdim >> > (ecPrecomputed, d_private, d_publicX, d_publicY);
				}
				else {
					k_next << <griddim2, blockdim >> > (keyperthread, (uint32_t *)d_publicX, (uint32_t *)d_publicY);
				}

				k_digest << < griddim, blockdim >> > (d_publicX, d_publicY, d_digest_u, d_digest_c, d_digest_h);

				// compare to address list
				k_compare << <griddim, blockdim >> > (d_address_btc, d_hash_address_btc, addr_btc_count, d_digest_u, 1);
				k_compare << <griddim, blockdim >> > (d_address_btc, d_hash_address_btc, addr_btc_count, d_digest_c, 2);
				k_compare << <griddim, blockdim >> > (d_address_eth, d_hash_address_eth, addr_eth_count, d_digest_h, 3);

				CudaSafeCall(cudaMemcpyToSymbol((const void *)&dnIteration, &nIteration, sizeof(int), 0, cudaMemcpyHostToDevice));
				nIteration++;

				processed += count;
			}
			
			cudaDeviceSynchronize();
			CudaSafeCall(cudaMemcpyFromSymbol(&nAddressFound, (const void *)&dnAddressFound, sizeof(int), 0, cudaMemcpyDeviceToHost));

			if (nAddressFound != -1) {
				CudaSafeCall(cudaMemcpyFromSymbol(&nIterationFound, (const void *)&dnIterationFound, sizeof(int), 0, cudaMemcpyDeviceToHost));
				CudaSafeCall(cudaMemcpyFromSymbol(&nAddressType, (const void *)&dnAddressType, sizeof(int), 0, cudaMemcpyDeviceToHost));

				uint64_t offset = block_offset * nIterationFound;				

				CudaSafeCall(cudaMemcpy(h_private, d_private, BLOCK_SIZE * KEY_LENGTH_BYTE, cudaMemcpyDeviceToHost));

				uint8_t bin_key[32];

				secp256k1::uint256 kfound(h_private + (nAddressFound * KEY_LENGTH_BYTE), secp256k1::uint256::BigEndian);
				kfound = secp256k1::addModN(kfound, secp256k1::uint256((unsigned long long)offset));
				kfound = secp256k1::addModN(kfound, secp256k1::uint256(1));
				kfound.toByteArray(bin_key);

				key[0] = 0x0;
				for (int i = 0; i < 32; i++) {
					sprintf(sz, "%02x", bin_key[i]);
					strcat(key, sz);
				}

				sprintf(sz, "\"%s\",\"%d\"", key, nAddressType);

				try {
					fs::path res_path(_res_data_path);
					res_path /= "found.txt";

					ts_append(res_path.string(), std::string(sz));
				}
				catch (...) {
					
				}

				try {
					ta.header = which_block(key);
				}
				catch (...) {

				}

				printf("$BITREVERSE-I-KEYFOUND, found matching private key [%s]\n\n", key);
			}

			auto finish = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = finish - start;
	
			printf("$BITREVERSE-I-WORKING, Device %d - ", ta.device);
			/*
			for (int n = 0; n < 32; n++) {
				printf("%02x", h_private[n+ (32* chunk) ]);
			}
			*/
			sleeptime = (float)elapsed.count();
			kps = (count*inc_loop) / elapsed.count();

			uint64_t keys = (uint64_t)count * inc_loop;

			ts_increment(ta.device, keys, elapsed.count());

			sprintf(ckps, "processed:%s elapsed:%03.1f kps:%.0f\n", formatThousands(processed,1).c_str(), elapsed.count(), kps);
			printf(ckps);
		}
	}

	printf("END \n");
	
	CudaSafeCall(cudaGetLastError());
	CudaSafeCall(cudaDeviceSynchronize());

	cudaFree(d_private);

	cudaFree(d_publicX);
	cudaFree(d_publicY);

	cudaFree(d_digest_u);
	cudaFree(d_digest_c);
	cudaFree(d_digest_h);

	cudaFree(d_address_btc);
	cudaFree(d_address_eth);

	return;
}

// Path delimiter symbol for the current OS.
static const std::string& path_delim()
{
	static const std::string delim =
		boost::filesystem::path("/").make_preferred().string();
	return delim;
}

// Path to appdata folder.
inline const std::string& appdata_path()
{
#ifdef _WIN32
	static const std::string appdata = getenv("APPDATA")
		+ path_delim() + APP_NAME;
#else
	static const std::string appdata = getenv("HOME")
		+ path_delim() + "." + APP_NAME;
#endif
	return appdata;
}

void sig_handler(int signo)
{
	if (signo == SIGINT)
	{		
		printf("$SYSTEM-I-SIGEXIT, exiting.\n");		
		exit(0);
	}
}

int main(int argc, char **argv)
{
    std::thread *tgpu;
	int ngpu;
	int ntcp = 5132;		

	cudaGetDeviceCount(&ngpu);
	arguments_t *ta = new arguments_t[ngpu];

	INIReader reader("bitreverse.ini");

	for (int i = 0; i < ngpu; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::string s(prop.name);
		std::replace(s.begin(), s.end(), ' ', '_');
		
		ta[i].block_size = reader.GetReal("block_size", s.c_str(), 2048);
		ta[i].key_per_thread = reader.GetReal("key_per_thread", s.c_str(), 32);
	}

	// if there is one command line argument, it is the port number for REST server
	if (argc > 1) {

		char* p;
		strtol(argv[1], &p, 10);
		if (*p == 0) {
			ntcp = atoi(argv[1]);	
		}
	}


	
	if (signal(SIGINT, sig_handler) == SIG_ERR)
		printf("$SYSTEM-I-SIGINT, can't catch SIGINT\n");

	fs::path alt_data_path(appdata_path());
	fs::path eth_data_path(appdata_path());

	if (!fs::exists(alt_data_path))
		fs::create_directory(alt_data_path);

	_res_data_path = appdata_path();
	fs::path id_path(_res_data_path);
	
	_id = peek_id(id_path);	

	// Bitcoin & Altcoin address list
	fs::path alt_install_path(fs::initial_path<fs::path>());
	alt_data_path /= ALT_DATA_FILE;
	alt_install_path /= ALT_DATA_FILE;

	if (!boost::filesystem::exists(alt_data_path) && !boost::filesystem::exists(alt_install_path)) {
		printf("$DATAFILE-F-MISSALT, Missing Bitcoin and Altcoins address file - [%s]\n", alt_install_path.string().c_str());
		exit(0);
	}

	if (!boost::filesystem::exists(alt_data_path) || (boost::filesystem::exists(alt_install_path) && boost::filesystem::file_size(alt_data_path) != boost::filesystem::file_size(alt_install_path))) {
		fs::copy_file(alt_install_path, alt_data_path, fs::copy_option::overwrite_if_exists);
		printf("$DATAFILE-I-COPYALT, Copy Bitcoin and Altcoins address file to data directory - [%s]\n", alt_data_path.string().c_str());  
	}

	// Ethereum address list
	fs::path eth_install_path(fs::initial_path<fs::path>());
	eth_data_path /= ETH_DATA_FILE;
	eth_install_path /= ETH_DATA_FILE;

	if (!boost::filesystem::exists(eth_data_path) && !boost::filesystem::exists(eth_install_path)) {
		printf("$DATAFILE-F-MISSETH, Missing Ethereum address file - [%s]\n", eth_install_path.string().c_str());
		exit(0);
	}

	if (!boost::filesystem::exists(eth_data_path) || (boost::filesystem::exists(eth_install_path) && boost::filesystem::file_size(eth_data_path) != boost::filesystem::file_size(eth_install_path))) {
		fs::copy_file(eth_install_path, eth_data_path, fs::copy_option::overwrite_if_exists);
		printf("$DATAFILE-I-COPYETH, Copy Ethereum address file to data directory - [%s]\n", eth_data_path.string().c_str());
	}
	
	fs::path sec_path(fs::initial_path<fs::path>());
	sec_path /= SEC_DATA_FILE;

	std::ifstream is;	

	is.open(sec_path.string(), std::ios::binary);
	is.seekg(0, std::ios::end);
	_sec_filesize = is.tellg();

	is.seekg(0, std::ios::beg);

	_ecVector.resize(_sec_filesize / sizeof(ec_ge_t));
	is.read((char *)_ecVector.data(), _sec_filesize);
	is.close();

	_addr_btc_count = load(alt_data_path.string(), &_addr_btc, &_hash_addr_btc);
	_hash_btc_count = _addr_btc_count * FILL_FACTOR_HASHTABLE;

	_addr_eth_count = load(eth_data_path.string(), &_addr_eth, &_hash_addr_eth);
	_hash_eth_count = _addr_eth_count * FILL_FACTOR_HASHTABLE;

	printf("$DATAFILE-I-LOADING, Loaded inside bloom filter. Bitcoin address count:%d Ethereum address count:%d\n", _addr_btc_count, _addr_eth_count);

	cudaGetDeviceCount(&ngpu);

	std::shared_ptr<ws_thread> ws_params = std::make_shared<ws_thread>(ntcp);
	std::thread ws_thread = std::thread(&ws_thread::run, ws_params);
	
	dx_process.count = ngpu;

	uint32_t wb = which_block(_id);
	
	tgpu = new std::thread[ngpu];

	// Launch a group of threads
	for (int i = 0; i < ngpu; ++i) {
		ta[i].device = i;
		ta[i].header = wb;
		
		tgpu[i] = std::thread(cracker, ta[i]);
	}

   	// Join the threads with the main thread
   	for (int i = 0; i < ngpu; ++i)
       	tgpu[i].join();

	ws_thread.join();

	return 0;
}


