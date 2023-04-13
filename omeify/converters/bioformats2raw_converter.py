import os
import tempfile
import subprocess
import logging
import shutil
import zarr

class Bioformats2RawConverter:
    def __init__(self, input_image, log_level = logging.WARNING):
        self.input_image = input_image
        self.log_level = log_level
        self.raw = None
        self.raw_path = None
        logging.basicConfig(level = log_level)
    def print_raw_path(self,prefix = " "):
        print_directory_tree(self.raw_path,prefix)
    def convert(self,series = None):
        # Create a temporary directory for the intermediate raw files
        tmp_dir = tempfile.mkdtemp(suffix = ".zarr")

        try:

            self._run_bioformats2raw(tmp_dir,series)
            
            # Continue processing the raw data in tmp_dir, e.g., converting to OME-TIFF
            # ...
            self.raw = zarr.open(tmp_dir,mode='r')
            self.raw_path = tmp_dir
        except Exception as e:
            # clean up temp dir in case of error
            shutil.rmtree(tmp_dir)
            raise e
        return self.raw
    def cleanup(self):
        if self.raw is not None:
            self.raw.store.close()
        if self.raw_path is not None and os.path.exists(self.raw_path):
            shutil.rmtree(self.raw_path)
        self.raw = None
        self.raw_path = None
    def _run_bioformats2raw(self, output_location, series):
        # Adjust the arguments as needed for your use case
        cmd = [
            "bioformats2raw",
            "--overwrite",
            "-p" if self.log_level >= logging.INFO else None,
            "-s" if series is not None else None, 
            f"{series if isinstance(series, int) else ','.join(map(str, series))}" if series is not None else None,
            self.input_image,
            output_location,
            # Additional command line arguments can be added here
        ]
        cmd = [x for x in cmd if x is not None]
        #print(cmd)
        if self.log_level >= logging.INFO:
            logging.info(f"{cmd}")
        stdout_setting = None if self.log_level <= logging.INFO else subprocess.PIPE
        stderr_setting = None if self.log_level <= logging.WARNING else subprocess.PIPE

        # Run the command and wait for completion
        process = subprocess.run(cmd, check=True, stdout=stdout_setting,stderr=stderr_setting)

        if self.log_level > logging.WARNING and process.stderr:
            logging.error(process.stderr.decode("utf-8"))

        # You can check the output or handle errors here, if necessary
        # For example:
        # if process.returncode != 0:
        #     raise Exception(f"Error during bioformats2raw conversion: {process.stderr.decode('utf-8')}")
