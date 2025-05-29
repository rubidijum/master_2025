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

### Building Zephyr samples

#### Prerequisites

Following the steps from [getting started instructions](https://docs.zephyrproject.org/latest/develop/getting_started/index.html) - after running apt update & upgrade,
run the following:

```bash
sudo apt install --no-install-recommends git cmake ninja-build gperf \
  ccache dfu-util device-tree-compiler wget \
  python3-dev python3-pip python3-setuptools python3-tk python3-wheel xz-utils file \
  make gcc gcc-multilib g++-multilib libsdl2-dev libmagic1
```

##### Setup the virtual environment

`sudo apt install python3-venv`

`python3 -m venv ~/zephyrproject/.venv`

`source ~/zephyrproject/.venv/bin/activate`

##### Install the west tool

`pip install west`

##### Setup the zephyr project

```bash
west init ~/zephyrproject
cd ~/zephyrproject
west update
```

`west zephyr-export`

`west packages pip --install`

##### Install the Zephyr SDK

```bash
cd ~/zephyrproject/zephyr
west sdk install
```
