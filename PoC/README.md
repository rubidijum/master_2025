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


###### Building the sample apps

In order to generate a zephyr .elf executable, the following steps should be run in sequence

1) Make sure that the virtual environment is activated (`source ..`)
2) `cd` into a `zephyrproject` directory (where the source code for ZephyrOS has been downloaded)
3) Run the west build command: `west build -p always -b qemu_cortex_m3 <path_to_the_sample_app>`

At the end of the successful build, the zephyr.elf will be generated in the build folder.

Alternatively, run the `get_zephyr_elf.sh` helper script to generate the executable automatically.

#### Dissasembling apps

For the accurate trace generation, analysis of the compiled zephyr binary needs to be performed. Rainbow tool provides funcion hooking mechanisms, memory access monitoring, register monitoring and more. For understanding where to hook into, dissassembly is necessary.

In order to dissasemble the binaries for targeted trace generation, ARM toolchain is required.
`sudo apt install gcc-arm-none-eabi`

Dissasembly output: `arm-none-eabi-objdump -d build/zephyr/zephyr.elf`

Getting the symbols: `arm-none-eabi-nm build/zephyr/zephyr.elf`