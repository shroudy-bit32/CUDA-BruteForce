/*
////////////////////////SOME INFO'S ABOUT GPU ARCHITECTURE/////////////////////////////
//			MEMORY TYPES:
//		Register ->	Fastest memory units	(There is 65536 Register unit in RTX4080 )
//		Shared Memory -> Second fastest memory unit (Sometimes its related to process u will make)
//		Constant Memory
//      L1 Cache
//		L2 Cache
//		Local memory -> Very slow
//
//-----------------------RTX 4080 MOBILE (Reminding for my device)-----------------------------------------------
//		65536 register at shader memory
//		7424 CUDA thread
//		64 KB Cconstant Memory
////////////////////////////////////////////////////////////////////////////////////////
*/






#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <nvml.h>
#include <thread>

//#include <openssl/sha.h>		 we do all things ourselves so we dont need this 😎

using namespace std;


//will be added defines or just make a config.h

__device__ __constant__ uint8_t	TARGET_HASH[32]{
	0x5e, 0x88, 0x41, 0x79, 0x4b, 0xe6, 0x0e, 0x26,
	0x8d, 0x70, 0x9d, 0x3d, 0x6f, 0xa7, 0x4f, 0x3f,
	0xa1, 0x9f, 0xa1, 0x5a, 0xc7, 0xff, 0xa7, 0x9f,
	0x7a, 0x79, 0x4e, 0x3d, 0x30, 0x49, 0x44, 0x2c		// Example hash ("password")
};


int calc_opt_mem(int a);
void device_properties();

__global__ void kernel(char* D_charset, int lengthSet, int numberOfdigit, char* c, int chunksize, uint64_t offset);

__device__ void generatePassword(char* D_charset, int lengthSet, int numberOfdigit, char* c, int chunksize, uint64_t offset, uint64_t globalThreadID);

__device__ void sha256(char* c, int numberOfdigit, uint64_t globalThreadID);
__device__ uint32_t Q0(uint32_t x);
__device__ uint32_t Q1(uint32_t x);
__device__ uint32_t E0(uint32_t x);
__device__ uint32_t E1(uint32_t x);
__device__ uint32_t CH(uint32_t x, uint32_t y, uint32_t z);
__device__ uint32_t MAJ(uint32_t x, uint32_t y, uint32_t z);
__device__ uint32_t ROTR(uint32_t x, uint8_t n);
__device__ uint32_t SHR(uint32_t x, uint8_t n);

__device__ void control(uint8_t hash, uint64_t globalThreadID);


int main()
{

	const string charset = "abcdefghijklmnopqrstuvwxyz";

	int lengthSet = charset.length();
	//uint64_t totalComb = lengthSet;				//thats bc the first random password will be 1 digit thats why at first lengthSet equal to totalcomb


	uint8_t flags = 0;

	char* D_charset;
	cudaMalloc(&D_charset, lengthSet * sizeof(char));
	cudaMemcpy(D_charset, charset.c_str(), lengthSet * sizeof(char), cudaMemcpyHostToDevice);


	//Device processes
	int numberOfdigit = 1;
	int chunksize = 100000000; //if numberofdigit is so big there could be security problems (so big totalcomb) so we need to divide it with the chunk size hundred million
	//Fun fact:  we dont use totalcomb anymore bc its fills memory so much and just chunk solution is better
	uint64_t offset = 0;	   //if totalcomb so big we need to optimize GPU threads so we are using offset to know which combination gpu stayed at then starting from there to try combinations

	int threadPerBlock = 256;
	uint64_t blockPerGrid;

	char* c;

	size_t size = chunksize * sizeof(char);

	cudaMalloc(&c, size);




	return 0;
}

void device_properties()			//we nearly get info enough to make HWmonitor
{
	cudaDeviceProp properties;

	cudaGetDeviceProperties(&properties, 0);


	size_t totalVram, freeVram;

	cudaMemGetInfo(&freeVram, &totalVram);

	string arch;

	if (properties.major == 9) { arch = "Ada Lovelace"; }
	if (properties.major == 8) { arch = "Ampere"; }
	if (properties.major == 7 && properties.minor == 0) { arch = "Volta"; }
	else if (properties.major == 7 && properties.minor == 5) { arch = "Turing"; }
	if (properties.major == 6) { arch = "Pascal"; }
	if (properties.major == 5) { arch = "Maxwell"; }
	if (properties.major == 3) { arch = "Kepler"; }
	if (properties.major == 2) { arch = "Fermi"; }
	if (properties.major == 1) { arch = "Tesla"; }



	nvmlReturn_t result;
	nvmlDevice_t device;
	unsigned int clockSpeed;
	unsigned int SMspeed;
	unsigned int boostClock;
	unsigned int power;
	unsigned int defaultLimit;
	nvmlEnableState_t mode;
	nvmlDevicePerfModes_t perfModes;
	nvmlUtilization_t utilization;
	size_t l2CacheKB;

	result = nvmlInit();
	if (NVML_SUCCESS != result) { printf("NVML Init failed:  %s\n", nvmlErrorString(result)); return; }

	result = nvmlDeviceGetHandleByIndex(0, &device);
	if (NVML_SUCCESS != result) { printf("Handling failed:  %s", nvmlErrorString(result)); nvmlShutdown(); return; }


	result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clockSpeed);
	if (NVML_SUCCESS != result) { printf("Failed to get clock speed:  %s", nvmlErrorString(result)); }


	result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &SMspeed);
	if (NVML_SUCCESS != result) { printf("Failed to get Shader memory speed:  %s", nvmlErrorString(result)); }


	result = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_GRAPHICS, &boostClock);
	if (NVML_SUCCESS != result) { printf("Failed to get boost clock info:  %s", nvmlErrorString(result)); }


	result = nvmlDeviceGetPowerUsage(device, &power);
	if (NVML_SUCCESS != result) { printf("Failed to get power usage: %s", nvmlErrorString(result)); }


	result = nvmlDeviceGetPowerManagementDefaultLimit(device, &defaultLimit);
	if (NVML_SUCCESS != result) { printf("Failed to get default power limit:  %s", nvmlErrorString(result)); }

	unsigned int max_power_limit;
	result = nvmlDeviceGetEnforcedPowerLimit(device, &max_power_limit);
	if (NVML_SUCCESS != result) { printf("Failed to get enforced power limit:  %s", nvmlErrorString(result)); }


	result = nvmlDeviceGetPowerManagementMode(device, &mode);
	if (NVML_SUCCESS != result) { printf("Failed to get power management mode:  %s", nvmlErrorString(result)); }


	result = nvmlDeviceGetPerformanceModes(device, &perfModes);
	if (NVML_SUCCESS != result) { printf("Failed to get peformance mode:  %s", nvmlErrorString(result)); }


	result = nvmlDeviceGetUtilizationRates(device, &utilization);
	if (NVML_SUCCESS != result) { printf("Failed to get utilization rate:  %s", nvmlErrorString(result)); }


	l2CacheKB = properties.l2CacheSize / 1024;

	nvmlShutdown();

}



int calc_opt_mem(int a)
{
	return 4 * a / 5;
}




__device__ __constant__ uint32_t K[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
	0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
	0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
	0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
	0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
	0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};


__global__ void kernel(char* D_charset, int lengthSet, int numberOfdigit, char* c, int chunksize, uint64_t offset)
{
	uint64_t globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;

	//generate random passwords
	generatePassword(D_charset, lengthSet, numberOfdigit, c, chunksize, offset, globalThreadID);

	//hash all generated passwords
	sha256(c, numberOfdigit, globalThreadID);

}


__device__ void generatePassword(char* D_charset, int lengthSet, int numberOfdigit, char* c, int chunksize, uint64_t offset, uint64_t globalThreadID)
{
	uint64_t globalCombID = offset + globalThreadID;


	for (int a = numberOfdigit - 1; a >= 0; a--)
	{
		c[globalThreadID * numberOfdigit + a] = D_charset[globalCombID % lengthSet];
		globalCombID /= lengthSet;
	}

}



__device__ void sha256(char* c, int numberOfdigit, uint64_t globalThreadID)
{
	uint8_t padded[64];							//every password combination is separated to thread's local/ registry memory
	uint64_t bitLength = numberOfdigit * 8;
	uint8_t hash[32];
	uint32_t w[64];


	//--------------------------------------------------------SHA 256 PADDİNG-------------------------------------------------------)

	for (int a = 0; a < numberOfdigit; a++)
	{
		padded[a] = c[globalThreadID * numberOfdigit + a];
	}


	//-----------------------------------------------------------------------------------
	padded[numberOfdigit] = 0x80;				//appending '1' bit to right of the password
	//-----------------------------------------------------------------------------------

	for (int a = numberOfdigit + 1; a < 56; a++)
	{
		padded[a] = 0x00;						//appending 0 till it reaches 448 unit bit
	}

	for (int a = 0; a < 8; a++)
	{
		padded[63 - a] = (bitLength >> (a * 8)) & 0xFF;
	}
	//--------------------------------------------------------END OF PADDİNG--------------------------------------------------------)




	for (int a = 0; a < 16; a++)		//Generates W array's element
	{
		w[a] = (
			(uint32_t)padded[(a * 4)] << 24 |
			(uint32_t)padded[(a * 4) + 1] << 16 |
			(uint32_t)padded[(a * 4) + 2] << 8 |
			(uint32_t)padded[(a * 4) + 3]
			);

	}

	for (int a = 16; a < 64; a++)
	{
		w[a] = Q1(w[a - 2]) + w[a - 7] + Q0(w[a - 15]) + w[a - 16];
	}



	// Generating H constants
	uint32_t A = 0x6a09e667;
	uint32_t B = 0xbb67ae85;
	uint32_t C = 0x3c6ef372;
	uint32_t D = 0xa54ff53a;
	uint32_t E = 0x510e527f;
	uint32_t F = 0x9b05688c;
	uint32_t G = 0x1f83d9ab;
	uint32_t H = 0x5be0cd19;

	uint32_t T1, T2;

	uint32_t H0 = A;
	uint32_t H1 = B;
	uint32_t H2 = C;
	uint32_t H3 = D;
	uint32_t H4 = E;
	uint32_t H5 = F;
	uint32_t H6 = G;
	uint32_t H7 = H;

	for (int m = 0; m < 64; m++)
	{
		T1 = H + E1(E) + CH(E, F, G) + K[m] + w[m];
		T2 = E0(A) + MAJ(A, B, C);

		H = G;
		G = F;
		F = E;
		E = D + T1;
		D = C;
		C = B;
		B = A;
		A = T1 + T2;
	}

	H0 = H0 + A;
	H1 = H1 + B;
	H2 = H2 + C;
	H3 = H3 + D;
	H4 = H4 + E;
	H5 = H5 + F;
	H6 = H6 + G;
	H7 = H7 + H;



	hash[0] = (H0 >> 24) & 0xFF;
	hash[1] = (H0 >> 16) & 0xFF;
	hash[2] = (H0 >> 8) & 0xFF;
	hash[3] = (H0 >> 0) & 0xFF;

	hash[4] = (H1 >> 24) & 0xFF;
	hash[5] = (H1 >> 16) & 0xFF;
	hash[6] = (H1 >> 8) & 0xFF;
	hash[7] = (H1 >> 0) & 0xFF;

	hash[8] = (H2 >> 24) & 0xFF;
	hash[9] = (H2 >> 16) & 0xFF;
	hash[10] = (H2 >> 8) & 0xFF;
	hash[11] = (H2 >> 0) & 0xFF;

	hash[12] = (H3 >> 24) & 0xFF;
	hash[13] = (H3 >> 16) & 0xFF;
	hash[14] = (H3 >> 8) & 0xFF;
	hash[15] = (H3 >> 0) & 0xFF;

	hash[16] = (H4 >> 24) & 0xFF;
	hash[17] = (H4 >> 16) & 0xFF;
	hash[18] = (H4 >> 8) & 0xFF;
	hash[19] = (H4 >> 0) & 0xFF;

	hash[20] = (H5 >> 24) & 0xFF;
	hash[21] = (H5 >> 16) & 0xFF;
	hash[22] = (H5 >> 8) & 0xFF;
	hash[23] = (H5 >> 0) & 0xFF;

	hash[24] = (H6 >> 24) & 0xFF;
	hash[25] = (H6 >> 16) & 0xFF;
	hash[26] = (H6 >> 8) & 0xFF;
	hash[27] = (H6 >> 0) & 0xFF;

	hash[28] = (H7 >> 24) & 0xFF;
	hash[29] = (H7 >> 16) & 0xFF;
	hash[30] = (H7 >> 8) & 0xFF;
	hash[31] = (H7 >> 0) & 0xFF;

	control(hash, numberOfdigit, globalThreadID, c);	//we are calling control() method in sha256 method bc the hash array is in the threads local memory
}

//-----------------------------------SHA 256 TOOLS-------------------------------------------------------------)

__device__ uint32_t Q0(uint32_t x)						//Q0 references to small sigma0
{
	return ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3);
}

__device__ uint32_t Q1(uint32_t x)						//Q1 references to small sigma1
{
	return ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10);
}




__device__ uint32_t E0(uint32_t x)						//E0 references to big sigma0
{
	return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22);
}

__device__ uint32_t E1(uint32_t x)						//E0 references to big sigma1
{
	return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25);
}



__device__ uint32_t CH(uint32_t x, uint32_t y, uint32_t z)
{
	return (x & y) ^ (~x & z);
}

__device__ uint32_t MAJ(uint32_t x, uint32_t y, uint32_t z)
{
	return (x & y) ^ (x & z) ^ (y & z);
}



__device__ uint32_t ROTR(uint32_t x, uint8_t n)
{
	return (x >> n) | (x << (32 - n));
}

__device__ uint32_t SHR(uint32_t x, uint8_t n)
{
	return x >> n;
}


//------------------------------------------------------------------------------------------------------------)

__device__ void control(uint8_t* hash, int numberOfdigit, uint64_t globalThreadID, char* c)
{
	bool match = 1;
	char CORRECT_PASSWORD[32];

	for (int a = 0; a < 32; a++)
	{
		if (hash[a] != TARGET_HASH[a]) { match = false; break; }
	}

	if (match)
	{
		printf("🎯 Password found: ");
		for (int a = 0; a < numberOfdigit; ++a)
		{
			printf("%c", c[globalThreadID * numberOfdigit + a]);
		}
		printf("\n");

	}
}