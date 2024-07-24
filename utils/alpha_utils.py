import os
import lzma
import dill as pickle


def load_pickle(path):
    with lzma.open(path, "rb") as fp:
        file = pickle.load(fp)
    return file


def save_pickle(path, obj):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with lzma.open(path, "wb") as fp:
        pickle.dump(obj, fp)
