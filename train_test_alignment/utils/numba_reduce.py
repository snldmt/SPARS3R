# https://github.com/numba/numba/issues/1269
from numba import njit
import numpy as np


@njit(cache=True)
def np_mean_along_axis(arr: np.ndarray, axis: int):
    assert arr.ndim == 2  # only supports 2D arrays
    assert axis in (0, 1)
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.mean(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.mean(arr[i, :])
    return result


@njit(cache=True)
def np_std_along_axis(arr: np.ndarray, axis: int):
    assert arr.ndim == 2  # only supports 2D arrays
    assert axis in (0, 1)
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.std(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.std(arr[i, :])
    return result