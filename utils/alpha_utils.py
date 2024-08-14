import os
import lzma
import dill as pickle


def load_pickle(path: str):
    """
    Loads and returns a Python object from a pickled file.

    Args:
        path (str): The file path to the pickled file.

    Returns:
        Any: The Python object stored in the pickled file.
    """
    with lzma.open(path, "rb") as fp:
        file = pickle.load(fp)
    return file


def save_pickle(path: str, obj):
    """
    Saves a Python object to a file using pickle.

    Args:
        path (str): The file path where the object should be saved.
        obj (Any): The Python object to be pickled and saved.

    Returns:
        None
    """
    directory = os.path.dirname(path)

    if not os.path.exists(directory):
        os.makedirs(directory)
    with lzma.open(path, "wb") as fp:
        pickle.dump(obj, fp)
