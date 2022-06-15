# Simple script used to open gzip-compressed GoogleNews-vectors-negative300.bin.gz and copy its contents into a .txt file.
# Only used during initial setup when building own sentiment model utilizing word2vec.


import os, sys, gzip, shutil

# file paths
gz_file = "GoogleNews-vectors-negative300.bin.gz"
txt_file = "word2vec.txt"

# exit if word2vec data is not yet downloaded
if not os.path.exists(gz_file):
    print("word2vec data does not exist. Please download it first. Exiting ...")
    sys.exit()

print("Now copying ...")

# open gzip-compressed file and copy its contents to open .txt file
with gzip.open(gz_file, "rb") as fin:
    with open(txt_file, "wb") as fout:
        shutil.copyfileobj(fin, fout)

print("Success")