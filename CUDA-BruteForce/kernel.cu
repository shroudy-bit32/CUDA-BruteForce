/*
////////////////////////SOME INFO'S ABOUT NVIDIA GPU ARCHITECTURE/////////////////////////////
//			MEMORY TYPES:
//		Register ->	Fastest memory units 	(There is 65536 Register unit in RTX4080 )
//		Shared Memory -> Second fastest memory unit (Sometimes its related to process u will make)
//		Constant Memory ->
//      L1 Cache
//		L2 Cache
//		Global memory -> Very slow
//
//
//-----------------------RTX 4080 MOBILE INFO (Reminding for my device)-----------------------------------------------
//		65536 register at shader memory	and 256KB per SM
//		7424 CUDA thread
//		64 KB Constant Memory
//		Shared Memory max 99KB(default) with some private config maybe max 163 kb per SM
////////////////////////////////////////////////////////////////////////////////////////
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <nvml.h>
#include <thread>
#include <Windows.h>

using namespace std;

#define chunksizeDEFİNED  29696
#define buffDEFİNED 50
#define threadPerBlockDEFİNED 256


__device__ __constant__ uint8_t TARGET_HASH[32] =
{
	0x38, 0x78, 0x22, 0x10, 0x12, 0xd3, 0x78, 0x5e, 0x4f, 0x21, 0xee, 0xf3, 0x71, 0x19, 0x41, 0x0a,
	0x7e, 0xd8, 0xeb, 0xb5, 0xde, 0x28, 0xef, 0x82, 0xc0, 0xca, 0xd4, 0x8d, 0x8c, 0xdc, 0x5d, 0x04
};
//dont forget GPT will probably hash it wrong so use web sites for encrypt

//void get_device_properties();						   //we deleted ts for now bc intellisense is dedecting nvmlInit but nvcc cant
													   // will be fixed after the debug


__global__ void kernel(int numberOfdigit, uint64_t offset, uint8_t* D_flags, char* __restrict__ D_CORRECT_PASSWORD);

__device__ void generatePassword(uint8_t* __restrict__ c, uint8_t numberOfdigit, uint64_t offset, uint32_t globalThreadID, uint8_t* __restrict__ D_flags, char* __restrict__ D_CORRECT_PASSWORD, uint64_t globalCombID);

__device__ void sha256(uint8_t* __restrict__ c, uint8_t numberOfdigit, uint32_t globalThreadID, uint8_t* __restrict__ D_flags, char* __restrict__ D_CORRECT_PASSWORD);
__device__ uint32_t Q0(uint32_t x);
__device__ uint32_t Q1(uint32_t x);
__device__ uint32_t E0(uint32_t x);
__device__ uint32_t E1(uint32_t x);
__device__ uint32_t CH(uint32_t x, uint32_t y, uint32_t z);
__device__ uint32_t MAJ(uint32_t x, uint32_t y, uint32_t z);
__device__ uint32_t ROTR(uint32_t x, uint8_t n);
__device__ uint32_t SHR(uint32_t x, uint8_t n);

__device__ void control(uint8_t* __restrict__ hash, uint8_t numberOfdigit, uint32_t globalThreadID, uint8_t* __restrict__ c, uint8_t* __restrict__ D_flags, char* __restrict__ D_CORRECT_PASSWORD);


int main()
{

	//uint64_t offset = 1;	   //we need to know which combination gpu stayed at then start from there to try combinations

	SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED);		//well well well welcome to dark side (Set's cpu to performance mode)


	uint64_t chunksize = chunksizeDEFİNED;
	int threadPerBlock = threadPerBlockDEFİNED;
	uint64_t blockPerGrid = (chunksize + threadPerBlock - 1) / threadPerBlock;

	char* D_CORRECT_PASSWORD;
	uint8_t* D_flags = 0;

	cudaHostAlloc(&D_CORRECT_PASSWORD, 32, cudaHostAllocMapped);
	cudaHostAlloc(&D_flags, 1, cudaHostAllocMapped);			// we didnt want to slow down kernel with copy paste process of flags, D_flags (cudaMemcpy)
	// so we said kernel to hold some place at RAM, Device and Host can acces this address
	// and somehow its not slow as normal RAM
	cudaEvent_t start, stop, startLoop, stopLoop;
	float elapsedTime, elapsedTimeLoop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&startLoop);
	cudaEventCreate(&stopLoop);

	cudaEventRecord(start);

	int numberOfdigit = 1;
	uint64_t offset = 0;

	for (; ((*D_flags & 1) == 0) && numberOfdigit < 6; numberOfdigit++)
	{
		offset = 0;  // Her digit için baştan başlanır

		uint64_t totalComb = 1;
		for (int i = 0; i < numberOfdigit; i++) totalComb *= 32;

		cudaEventRecord(startLoop);

		while (offset < totalComb && ((*D_flags & 1) == 0))
		{


			kernel << <blockPerGrid, threadPerBlock >> > (numberOfdigit, offset, D_flags, D_CORRECT_PASSWORD);
			cudaDeviceSynchronize();

			offset += chunksize * buffDEFİNED;

		}

		cudaEventRecord(stopLoop);
		cudaEventSynchronize(stopLoop);
		cudaEventElapsedTime(&elapsedTimeLoop, startLoop, stopLoop);
		printf("Digit %d ended\n\tPassedTime: %.4f ms\n", numberOfdigit, elapsedTimeLoop);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);


	cudaEventElapsedTime(&elapsedTime, start, stop);


	if (*D_flags == 1) {
		printf("Password found:\n\t%s\n", D_CORRECT_PASSWORD);
		printf("Passed time:  %.5f ms\n", elapsedTime);
	}

	system("pause");

	return 0;
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


__device__ __constant__ char D_charset[32] = { 'a','b','c','d','e','f','g','h',
											  'i','j','k','l','m','n','o','p',
											  'q','r','s','t','u','v','w','x',
											  'y','z', ' ', ' ', ' ', ' ', ' ' };
//we filled last 6 with spaces bc now we can use more bitwise operations which is so much faster
//and with this if u change Charset dont forget to change charset optimizations which is for 32 digit charset



/*
__device__ __constant__ char D_charset[26] =
{
	'a','b','c','d','e','f','g','h','i','j','k','l','m',
	'n','o','p','q','r','s','t','u','v','w','x','y','z'
};
*/

__device__ __constant__ int lengthSet = 32;
//__device__ __constant__ int chunksize = chunksizeDEFİNED;
__device__ __constant__ int buff = buffDEFİNED;

__global__ void kernel(int numberOfdigit, uint64_t offset, uint8_t* D_flags, char* __restrict__ D_CORRECT_PASSWORD)
{

	uint32_t globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;
	register uint8_t c[64];

	for (int k = 0; k < buff; k++)
	{
		//generate random passwords
		uint64_t globalCombID = offset + ((uint64_t)globalThreadID * buff) + k;
		generatePassword(c, numberOfdigit, offset, globalThreadID, D_flags, D_CORRECT_PASSWORD, globalCombID);
	}
}


__device__ void generatePassword(uint8_t* __restrict__ c, uint8_t numberOfdigit, uint64_t offset, uint32_t globalThreadID, uint8_t* __restrict__ D_flags, char* __restrict__ D_CORRECT_PASSWORD, uint64_t globalCombID)
{

#pragma unroll
	for (int a = numberOfdigit - 1; a >= 0; a--)
	{
		c[a] = D_charset[globalCombID & 31];				//if you want to change charset length change 31 and 5 with lengthSet
		globalCombID >>= 5;									//and change '&' with '%' and change '>>' with '/' 

	}

	sha256(c, numberOfdigit, globalThreadID, D_flags, D_CORRECT_PASSWORD);
}



__device__ void sha256(uint8_t* __restrict__ c, uint8_t numberOfdigit, uint32_t globalThreadID, uint8_t* __restrict__ D_flags, char* __restrict__ D_CORRECT_PASSWORD)
{

	uint16_t bitLength = numberOfdigit * 8;

	register uint8_t hash[32];
	register uint32_t w[64];


	//every password combination is separated to thread's local/ registry memory
	//--------------------------------------------------------SHA 256 PADDİNG-------------------------------------------------------)

	//-----------------------------------------------------------------------------------
	c[numberOfdigit] = 0x80;				//appending '1' bit to right of the password
	//-----------------------------------------------------------------------------------


#pragma unroll
	for (int a = numberOfdigit + 1; a < 56; a++)
	{
		c[a] = 0x00;						//appending 0 till it reaches 448 unit bit
	}

#pragma unroll
	for (int a = 0; a < 8; a++)
	{
		c[63 - a] = (bitLength >> (a * 8)) & 0xFF;
	}
	//--------------------------------------------------------END OF PADDİNG--------------------------------------------------------)

#pragma unroll
	for (int a = 0; a < 16; a++)		//Generates W array's element
	{
		w[a] = (
			(uint32_t)c[(a * 4)] << 24 |
			(uint32_t)c[(a * 4) + 1] << 16 |
			(uint32_t)c[(a * 4) + 2] << 8 |
			(uint32_t)c[(a * 4) + 3]
			);
	}

#pragma unroll	
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

#pragma unroll
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

	control(hash, numberOfdigit, globalThreadID, c, D_flags, D_CORRECT_PASSWORD);	//we are calling control() method in sha256 method bc the hash array is in the threads local memory

}

//-----------------------------------SHA 256 TOOLS-------------------------------------------------------------)

__device__ __forceinline__ uint32_t Q0(uint32_t x)						//Q0 references to small sigma0
{
	return ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3);
}

__device__ __forceinline__ uint32_t Q1(uint32_t x)						//Q1 references to small sigma1
{
	return ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10);
}


__device__ __forceinline__ uint32_t E0(uint32_t x)						//E0 references to big sigma0
{
	return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22);
}

__device__ __forceinline__ uint32_t E1(uint32_t x)						//E0 references to big sigma1
{
	return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25);
}



__device__ __forceinline__ uint32_t CH(uint32_t x, uint32_t y, uint32_t z)
{
	return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t MAJ(uint32_t x, uint32_t y, uint32_t z)
{
	return (x & y) ^ (x & z) ^ (y & z);
}



__device__ __forceinline__ uint32_t ROTR(uint32_t x, uint8_t n)
{
	return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t SHR(uint32_t x, uint8_t n)
{
	return x >> n;
}


//------------------------------------------------------------------------------------------------------------)

__device__ void control(uint8_t* __restrict__ hash, uint8_t numberOfdigit, uint32_t globalThreadID, uint8_t* __restrict__ c, uint8_t* __restrict__ D_flags, char* __restrict__ D_CORRECT_PASSWORD)
{


#pragma unroll
	for (int a = 0; a < 32; a++)
	{
		if (hash[a] != TARGET_HASH[a]) { return; }
	}

	for (int a = 0; a < numberOfdigit; ++a)
	{
		D_CORRECT_PASSWORD[a] = c[a];
	}
	D_CORRECT_PASSWORD[numberOfdigit] = '\0';
	*D_flags = 1;
}


