# Proof of concept

This directory serves as a starting point for thesis development. The goal is to set up a synthetic trace generation based on the compiled ZephyrOS binaries. After the trace generation, initial analysis on the generated dataset is performed, exploring the possibility of attacking the traces via SCA. Signal-to-noise ratio is used to inspect the leakage of the crypto implementation. After initial SNR analysis, a couple of attacks (including CPA, ML-based attacks) are performed on the traces to extract the secret key.

## Folder organization

Within `application_aes_app`, source code for zephyr app using tinyAES implementation of crypto is implemented. `dissassembly_tinyaes.txt` holds the result of `-objdump`.

## Attacks

### CPA

### ML

#### MLP

#### CNN

#### (Simplified) Transformer nets
