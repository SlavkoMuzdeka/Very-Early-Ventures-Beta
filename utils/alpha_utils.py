import lzma
import dill as pickle
import scipy.stats as stats


def load_pickle(path):
    with lzma.open(path, "rb") as fp:
        file = pickle.load(fp)
    return file


def save_pickle(path, obj):
    with lzma.open(path, "wb") as fp:
        pickle.dump(obj, fp)


def zscore_to_percentage(z):
    return stats.norm.cdf(z) * 100
