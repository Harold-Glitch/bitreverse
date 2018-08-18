#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "program.fatbin.c"
extern void __device_stub__Z10k_generateP7ec_ge_tPhS1_S1_(ec_ge_t *, uint8_t *, uint8_t *, uint8_t *);
extern void __device_stub__Z6k_nextPjS_(uint32_t *, uint32_t *);
extern void __device_stub__Z9k_comparePyPhiS_i(uint64_t *, uint8_t *, int, uint64_t *, int);
extern void __device_stub__Z8k_digestPhS_PyS0_S0_(uint8_t *, uint8_t *, uint64_t *, uint64_t *, uint64_t *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void);
#pragma section(".CRT$XCU",read)
__declspec(allocate(".CRT$XCU"))static void (*__dummy_static_init__sti____cudaRegisterAll[])(void) = {__sti____cudaRegisterAll};
void __device_stub__Z10k_generateP7ec_ge_tPhS1_S1_(
ec_ge_t *__par0, 
uint8_t *__par1, 
uint8_t *__par2, 
uint8_t *__par3)
{
__cudaLaunchPrologue(4);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaSetupArgSimple(__par3, 24Ui64);
__cudaLaunch(((char *)((void ( *)(ec_ge_t *, uint8_t *, uint8_t *, uint8_t *))k_generate)));
}
void k_generate( ec_ge_t *__cuda_0,uint8_t *__cuda_1,uint8_t *__cuda_2,uint8_t *__cuda_3)
{__device_stub__Z10k_generateP7ec_ge_tPhS1_S1_( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
}
#line 1 "x64/Release/program.compute_61.cudafe1.stub.c"
void __device_stub__Z6k_nextPjS_(
uint32_t *__par0, 
uint32_t *__par1)
{
__cudaLaunchPrologue(2);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaLaunch(((char *)((void ( *)(uint32_t *, uint32_t *))k_next)));
}
void k_next( uint32_t *__cuda_0,uint32_t *__cuda_1)
{__device_stub__Z6k_nextPjS_( __cuda_0,__cuda_1);
}
#line 1 "x64/Release/program.compute_61.cudafe1.stub.c"
void __device_stub__Z9k_comparePyPhiS_i(
uint64_t *__par0, 
uint8_t *__par1, 
int __par2, 
uint64_t *__par3, 
int __par4)
{
__cudaLaunchPrologue(5);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaSetupArgSimple(__par3, 24Ui64);
__cudaSetupArgSimple(__par4, 32Ui64);
__cudaLaunch(((char *)((void ( *)(uint64_t *, uint8_t *, int, uint64_t *, int))k_compare)));
}
void k_compare( uint64_t *__cuda_0,uint8_t *__cuda_1,int __cuda_2,uint64_t *__cuda_3,int __cuda_4)
{__device_stub__Z9k_comparePyPhiS_i( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
}
#line 1 "x64/Release/program.compute_61.cudafe1.stub.c"
void __device_stub__Z8k_digestPhS_PyS0_S0_(
uint8_t *__par0, 
uint8_t *__par1, 
uint64_t *__par2, 
uint64_t *__par3, 
uint64_t *__par4)
{
__cudaLaunchPrologue(5);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 16Ui64);
__cudaSetupArgSimple(__par3, 24Ui64);
__cudaSetupArgSimple(__par4, 32Ui64);
__cudaLaunch(((char *)((void ( *)(uint8_t *, uint8_t *, uint64_t *, uint64_t *, uint64_t *))k_digest)));
}
void k_digest( uint8_t *__cuda_0,uint8_t *__cuda_1,uint64_t *__cuda_2,uint64_t *__cuda_3,uint64_t *__cuda_4)
{__device_stub__Z8k_digestPhS_PyS0_S0_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
}
#line 1 "x64/Release/program.compute_61.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback(
void **__T2606)
{
__nv_dummy_param_ref(__T2606);
__nv_save_fatbinhandle_for_managed_rt(__T2606);
__cudaRegisterEntry(__T2606, ((void ( *)(uint8_t *, uint8_t *, uint64_t *, uint64_t *, uint64_t *))k_digest), _Z8k_digestPhS_PyS0_S0_, (-1));
__cudaRegisterEntry(__T2606, ((void ( *)(uint64_t *, uint8_t *, int, uint64_t *, int))k_compare), _Z9k_comparePyPhiS_i, (-1));
__cudaRegisterEntry(__T2606, ((void ( *)(uint32_t *, uint32_t *))k_next), _Z6k_nextPjS_, (-1));
__cudaRegisterEntry(__T2606, ((void ( *)(ec_ge_t *, uint8_t *, uint8_t *, uint8_t *))k_generate), _Z10k_generateP7ec_ge_tPhS1_S1_, (-1));
__cudaRegisterVariable(__T2606, __shadow_var(_lock,::_lock), 0, 4Ui64, 0, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_P,::_P), 0, 32Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_GX,::_GX), 0, 32Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_GY,::_GY), 0, 32Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_N,::_N), 0, 32Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_BETA,::_BETA), 0, 32Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_LAMBDA,::_LAMBDA), 0, 32Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_K,::_K), 0, 256Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_IV,::_IV), 0, 32Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(d_keccakf_rndc,::d_keccakf_rndc), 0, 192Ui64, 0, 0);
__cudaRegisterVariable(__T2606, __shadow_var(d_keccakf_rotc,::d_keccakf_rotc), 0, 192Ui64, 0, 0);
__cudaRegisterVariable(__T2606, __shadow_var(d_keccakf_piln,::d_keccakf_piln), 0, 96Ui64, 0, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_RIPEMD160_IV,::_RIPEMD160_IV), 0, 20Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_K0,::_K0), 0, 4Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_K1,::_K1), 0, 4Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_K2,::_K2), 0, 4Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_K3,::_K3), 0, 4Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_K4,::_K4), 0, 4Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_K5,::_K5), 0, 4Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_K6,::_K6), 0, 4Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_K7,::_K7), 0, 4Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_INC_X,::_INC_X), 0, 32Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(_INC_Y,::_INC_Y), 0, 32Ui64, 1, 0);
__cudaRegisterVariable(__T2606, __shadow_var(dnAddressFound,::dnAddressFound), 0, 4Ui64, 0, 0);
__cudaRegisterVariable(__T2606, __shadow_var(dnIteration,::dnIteration), 0, 4Ui64, 0, 0);
__cudaRegisterVariable(__T2606, __shadow_var(dnIterationFound,::dnIterationFound), 0, 4Ui64, 0, 0);
__cudaRegisterVariable(__T2606, __shadow_var(dnAddressType,::dnAddressType), 0, 4Ui64, 0, 0);
}
static void __sti____cudaRegisterAll(void)
{
__cudaRegisterBinary(__nv_cudaEntityRegisterCallback);
}
