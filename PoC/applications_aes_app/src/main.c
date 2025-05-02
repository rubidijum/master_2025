#include <zephyr/kernel.h>
#include <stdio.h>
#include "aes.h"

uint8_t key[16] = {0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF};
uint8_t plaintext[16] __attribute__((section(".uninit")));
uint8_t ciphertext[16];

int main(void){
	struct AES_ctx ctx;
	AES_init_ctx(&ctx, key);


	memcpy(ciphertext, plaintext, 16);
	AES_ECB_encrypt(&ctx, ciphertext);

	if(ciphertext[0] == 0x42){
		printf("Magic byte\n");
	}

	return 0;	
}
