# install_gensim.py

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    # install("gensim")  # Removed gensim dependency
