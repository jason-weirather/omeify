import os
import tempfile
import subprocess
import logging
import shutil
import zarr
from os import PathLike
from zarr.hierarchy import Group

class Raw2OmeTiffConverter:
    def __init__(self, input_raw):
        self.input_raw = None
        self.logger = logging.getLogger(__name__)

        if isinstance(input_raw, (str,PathLike)):
            self.input_raw = self.input_raw = zarr.open(input_raw, mode='r')
        elif isinstance(input_raw, Group):
            self.input_raw = input_raw
        else:
            raise ValueError("input_raw must be a directory path or a zarr.hierarchy.Group object.")

    def convert(self, output_path, rgb = False):
        conversion = self._run_raw2ometiff(output_path,rgb)
        return conversion

    def _run_raw2ometiff(self,output_path,rgb):
        cmd = [
            "raw2ometiff",
            "-p" if self.logger.isEnabledFor(logging.INFO) else None,
            "--rgb" if rgb else None,
            self.input_raw.store.path,
            output_path,
            # Additional command line arguments can be added here
        ]
        cmd = [x for x in cmd if x is not None]

        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"{cmd}")

        stdout_setting = subprocess.PIPE
        stderr_setting = None if self.logger.isEnabledFor(logging.INFO) else subprocess.PIPE

        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info("Executing raw2ometiff conversion...")
        process = subprocess.run(cmd, check=True, stdout=stdout_setting, stderr=stderr_setting)

        if self.logger.isEnabledFor(logging.WARNING) and process.stderr:
            self.logger.error(process.stderr.decode("utf-8"))

        # You can check the output or handle errors here, if necessary
        # For example:
        # if process.returncode != 0:
        #     raise Exception(f"Error during raw2ometiff conversion: {process.stderr.decode('utf-8')}")
        return output_path
