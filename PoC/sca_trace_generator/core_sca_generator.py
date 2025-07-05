import numpy as np
import lascar
from lascar.tools.aes import sbox
from rainbow import TraceConfig, Print, HammingWeight
from rainbow.devices import rainbow_stm32f215
from abc import ABC, abstractmethod

class SideChannelTarget(ABC):
    def __init__(self, target_path, num_traces=5000, key=b"\xde\xad\xbe\xef"*4, verbose=False):
        self.num_traces = num_traces
        self.key = key
        self.traces = None
        self.labels = None
        self.plaintexts = None

        self.print_cfg = Print(0)

        if verbose is True:
            self.print_cfg=Print.Functions

        self.emu = rainbow_stm32f215(
            print_config=self.print_cfg,
            trace_config=TraceConfig(register=HammingWeight(), mem_value=HammingWeight())
        )

        self._load_target(target_path)

    @property
    @abstractmethod
    def name(self):
        """
        Unique name of the target.
        """
        pass

    @abstractmethod
    def _load_target(self, target_path):
        """
        Loads the specified .elf target
        """
        pass

    # @abstractmethod
    # def _setup_emu(self):
    #     """
    #     Performs emulator setup
    #     """

    @abstractmethod
    def _encrypt_step(self, plaintext: bytes, key: bytes):
        """
        Runs a single encryption operation

        Args:
            plaintext: 16-byte plaintext

        Returns:
            Numpy array containing ciphertext
        """
        pass

    @abstractmethod
    def _get_leakage(self) -> np.ndarray:
        """
        Gets leakage based on the single encryption operation
        """
        pass

    # class CortexMAesContainer(lascar.AbstractContainer):
    #     def generate_traces(self, idx):
    #         plaintext = np.random.randint(0, 256, (16,), np.uint8)
    #         self._encrypt_step(self.key, plaintext.tobytes())
    #         leakage = self._get_leakage()
    #         return lascar.Trace(leakage, plaintext)

    # def generate_traces(self):
    #     print(f"Generating {self.num_traces} traces for target '{self.name}'...")

    #     traces_list = []
    #     plaintexts_list = []

    #     for _ in range(self.num_traces):
    #         plaintext = np.random.randint(0, 256, (16,), np.uint8)
    #         # Update the internal emulator state after single encryption
    #         self._encrypt_step(plaintext.tobytes())
    #         leakage = self._get_leakage()
    #         traces_list.append(leakage)
    #         plaintexts_list.append(plaintext)

    #     self.traces = np.array(traces_list)
    #     self.plaintexts = np.array(plaintexts_list)

    #     key_array = np.array([byte for byte in self.key])
    #     self.labels = sbox([self.plaintexts ^ key_array])
    #     print("Trace generation complete")

    # def save_traces(self, filename=None):

    #     if self.traces is None:
    #         raise ValueError("Traces have not been generated yet. Call generate_traces to update the internal state first.")
        
    #     if filename is None:
    #         filename = f"traces_{self.name}_.npz"

    #     np.savez(
    #         filename,
    #         traces=self.traces,
    #         labels=self.labels,
    #         plaintexts=self.plaintexts,
    #         key=self.key
    #     )
