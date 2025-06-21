"""!
    Utility script for checking the Unicorn installation.
"""

from capstone import *
from unicorn import *
from unicorn.arm_const import *

CODE = b"\x01\x30\x8f\xe2"

print("-- Capstone ARM32 --")
md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
for insn in md.disasm(CODE, 0x1000):
    print(f"{insn.address:#x}: {insn.mnemonic} {insn.op_str}")

print("\n-- Unicorn ARM32 ---")
ADDRESS = 0x1000
try:
    mu = Uc(UC_ARCH_ARM, UC_MODE_THUMB)
    mu.mem_map(ADDRESS, 0x1000)
    mu.mem_write(ADDRESS, CODE)
    mu.reg_write(UC_ARM_REG_R0, 0x1234)
    mu.emu_start(ADDRESS, ADDRESS + len(CODE))
    print("Unicorn ran successfully")
except Exception as e:
    print(f"Unicorn failed: {e}")
