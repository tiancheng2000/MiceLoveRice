from __future__ import absolute_import, division, print_function

import h5py
import numpy as np

__all__ = [
    "create_dataset",
    "append_dataset",
]


def create_dataset(group: h5py.Group, db_name: str, data: list or np.ndarray, extensible=True, compression="gzip"):
    if isinstance(data, list):
        data = np.array(data)
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be either np.ndarray or list")
    if extensible:
        maxshape = (None, *data.shape[1:])  # set axis 0 to unlimited (for expansion)
        return group.create_dataset(db_name, data=data, compression=compression, chunks=True, maxshape=maxshape)
    else:
        return group.create_dataset(db_name, data=data, compression=compression)


def append_dataset(group: h5py.Group, db_name: str, data: list or np.ndarray):
    if db_name not in group.keys():
        raise ValueError(f"db_name({db_name}) cannot be found in group keys")
    count = len(data)
    group[db_name].resize(group[db_name].shape[0] + count, axis=0)
    group[db_name][-count:] = data
