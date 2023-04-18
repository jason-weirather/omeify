import os, re
import tempfile
import subprocess
import logging
import shutil
import zarr

class Bioformats2RawConverter:
    # make a class level logger
    logger = logging.getLogger(__name__)

    def __init__(self, input_image):
        self.input_image = input_image
        self.raw = None
        self.raw_path = None
        self.logger = logging.getLogger(__name__)
    @classmethod
    def get_version(cls):
        cls.logger.info(f"Getting bioformats2raw version")
        try:
            result = subprocess.run(['bioformats2raw','--version'],
                capture_output=True,text=True,check=True)
            m = re.search('Version = (\S+)',result.stdout.strip())
            if not m:
                cls.logger.warning(f"Failed to extract version from bioformats2raw")
                return None
            else:
                return m.group(1)
        except subprocess.CalledProcessError as e:
            cls.logger.warning(f"Error occured while trying to get the version of bioformats2raw: {e}")
            return None
    def print_raw_path(self,prefix = " "):
        print_directory_tree(self.raw_path,prefix)
    def get_ome_xml(self):
        if self.raw is None:
            raise ValueError("Need to convert first.")
        ome_xml = None
        with open(os.path.join(self.raw.store.path,'OME','METADATA.ome.xml'),'r') as inf:
            ome_xml = inf.read()
        return ome_xml
    def convert(self,series, cache_directory=None):
        # Create a temporary directory for the intermediate raw files
        tmp_dir = None
        if cache_directory is not None:
            if not os.path.exists(cache_directory):
                os.makedirs(cache_directory)
            tmp_dir = tempfile.mkdtemp(suffix=".zarr", dir=cache_directory)
        elif cache_directory is None:
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
            "-p" if self.logger.isEnabledFor(logging.INFO) else None,
            "-s" if series is not None else None, 
            f"{series if isinstance(series, int) else ','.join(map(str, series))}" if series is not None else None,
            self.input_image,
            output_location,
            # Additional command line arguments can be added here
        ]
        cmd = [x for x in cmd if x is not None]
        #print(cmd)
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"{cmd}")
        stdout_setting = subprocess.PIPE
        stderr_setting = None if self.logger.isEnabledFor(logging.INFO) else subprocess.PIPE

        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info("Executing bioformats2raw conversion command...")

        # Run the command and wait for completion
        process = subprocess.run(cmd, check=True, stdout=stdout_setting,stderr=stderr_setting)

        if self.logger.isEnabledFor(logging.WARNING) and process.stderr:
            self.logger.error(process.stderr.decode("utf-8"))

