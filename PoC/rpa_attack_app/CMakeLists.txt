cmake_minimum_required(VERSION 3.13.1)

# Default config
if(NOT CONF_FILE)
  set(CONF_FILE prj.conf)
endif()

# Allow external config chaining
if(EXTRA_CONF_FILE)
  list(APPEND CONF_FILE ${EXTRA_CONF_FILE})
endif()

set(CONF_FILE "${CONF_FILE}" CACHE STRING "List of configuration files")

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(bluetooth_rpa)

target_sources(app PRIVATE src/main.c src/aes.c)

add_compile_definitions(MBEDTLS_ENABLE_MASKING=$<BOOL:${CONFIG_MBEDTLS_ENABLE_MASKING}>)

