"""File paths for deep_fim_hapsi.py"""

from os import getcwd, makedirs
from os.path import abspath, dirname, exists, join 
from shutil import rmtree


BASE_DIR: str = dirname(abspath(getcwd()))

# Possible additional folders and filenames are appended in the functions.
saved_data: str = join(BASE_DIR, "saved data")
saved_images: str = join(BASE_DIR, "plots")
saved_Rr_and_Lc: str = join(BASE_DIR, "Rr and Lc")
saved_errors: str = join(BASE_DIR, "errors")
bitflip_errors: str = join(BASE_DIR, "bitflips")
saved_logs: str = join(BASE_DIR, "logs")

SYNTH_DATA: str = join(BASE_DIR, "synth data") 
BASSO_DATA: str = join(BASE_DIR, "matrix market")
BASSO_OUT: str = join(BASE_DIR, "basso out")


def reset_directory(directory: str):
    """Deletes 'directory' and recreates it."""
    if exists(directory):
        rmtree(directory, ignore_errors=True)
    makedirs(directory, exist_ok=True)
