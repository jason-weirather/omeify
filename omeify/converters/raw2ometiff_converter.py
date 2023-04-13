import os
import tempfile
import subprocess
import logging
import shutil
import zarr
from os import PathLike
from zarr.hierarchy import Group

class Raw2OmeTiffConverter:
    def __init__(self, input_raw, log_level=logging.WARNING):
        self.input_raw = None
        self.log_level = log_level
        logging.basicConfig(level=log_level)

        if isinstance(input_raw, (str,PathLike)):
            self.input_raw = self.input_raw = zarr.open(input_raw, mode='r')
        elif isinstance(input_raw, Group):
            self.input_raw = input_raw
        else:
            raise ValueError("input_raw must be a directory path or a zarr.hierarchy.Group object.")

    def convert(self, output_path):
        output_path = self._run_raw2ometiff(output_path)

    def _run_raw2ometiff(self,output_path):
        cmd = [
            "raw2ometiff",
            "-p",
            self.input_raw.store.path,
            output_path,
            # Additional command line arguments can be added here
        ]
        cmd = [x for x in cmd if x is not None]

        if self.log_level >= logging.INFO:
            logging.info(f"{cmd}")

        stdout_setting = None if self.log_level <= logging.INFO else subprocess.PIPE
        stderr_setting = None if self.log_level <= logging.WARNING else subprocess.PIPE

        process = subprocess.run(cmd, check=True, stdout=stdout_setting, stderr=stderr_setting)

        if self.log_level > logging.WARNING and process.stderr:
            logging.error(process.stderr.decode("utf-8"))

        # You can check the output or handle errors here, if necessary
        # For example:
        # if process.returncode != 0:
        #     raise Exception(f"Error during raw2ometiff conversion: {process.stderr.decode('utf-8')}")
        return output_path
