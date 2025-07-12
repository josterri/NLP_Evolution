import subprocess
import sys

#subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])

import nltk
import os

# Define your custom data directory
nltk_data_path = r"D:\Joerg\Research\NLP\NLP_Evolution\nltk_data"

# Make sure the directory exists
os.makedirs(nltk_data_path, exist_ok=True)

# Download to that path
nltk.download('punkt', download_dir=nltk_data_path)

# Tell NLTK to look there for resources
nltk.data.path.append(nltk_data_path)

# Example usage
from nltk.tokenize import word_tokenize

text = "This is an example sentence."
tokens = word_tokenize(text)
print(tokens)
