import os

def checked_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

