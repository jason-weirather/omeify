import os, re
import tempfile
import subprocess
import logging
import shutil
import zarr
from os import PathLike
from zarr.hierarchy import Group

class Raw2OmeTiffConverter:
    # make a class level logger
    logger = logging.getLogger(__name__)

    def __init__(self, input_raw):
        self.input_raw = None
        self.logger = logging.getLogger(__name__)

        if isinstance(input_raw, (str,PathLike)):
            self.input_raw = self.input_raw = zarr.open(input_raw, mode='r')
        elif isinstance(input_raw, Group):
            self.input_raw = input_raw
        else:
            raise ValueError("input_raw must be a directory path or a zarr.hierarchy.Group object.")

    @classmethod
    def get_version(cls):
        cls.logger.info(f"Getting raw2ometiff version")
        try:
            result = subprocess.run(['raw2ometiff','--version'],
                capture_output=True,text=True,check=True)
            m = re.search('Version = (\S+)',result.stdout.strip())
            if not m:
                cls.logger.warning(f"Failed to extract version from raw2ometiff")
                return None
            else:
                return m.group(1)
        except subprocess.CalledProcessError as e:
            cls.logger.warning(f"Error occured while trying to get the version of raw2ometiff: {e}")
            return None

    def convert(self, output_path, rgb = False, compression = None):
        conversion = self._run_raw2ometiff(output_path,rgb,compression)
        return conversion

    def _run_raw2ometiff(self,output_path,rgb,compression):
        cmd = [
            "raw2ometiff",
            "-p" if self.logger.isEnabledFor(logging.INFO) else None,
            "--rgb" if rgb else None,
            f'--compression={compression}' if compression else None,
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
            
        #process = subprocess.run(cmd, check=True, stdout=stdout_setting, stderr=stderr_setting)
        try:
            process = subprocess.run(cmd, check=True, stdout=stdout_setting, stderr=stderr_setting)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}: {e.stderr}")

        if self.logger.isEnabledFor(logging.WARNING) and process.stderr:
            self.logger.error(process.stderr.decode("utf-8"))

        return output_path
