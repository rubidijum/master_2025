cmake_minimum_required(VERSION 3.13.1)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(bluetooth_rpa)
target_sources(app PRIVATE src/main.c src/aes.c)

add_compile_definitions(MBEDTLS_AES_MASKING=$<BOOL:${CONFIG_MBEDTLS_ENABLE_MASKING}>)