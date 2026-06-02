from __future__ import annotations

import collections
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np


class _StorageType:
    def __init__(self, dtype_name: str):
        self.dtype_name = dtype_name


class _StorageRef:
    def __init__(self, key: str, dtype_name: str):
        self.key = key
        self.dtype_name = dtype_name


class _TensorRef:
    def __init__(self, storage: _StorageRef, offset: int, shape, stride):
        self.storage = storage
        self.offset = offset
        self.shape = tuple(shape)
        self.stride = tuple(stride)


def _rebuild_tensor_v2(storage, offset, shape, stride, requires_grad, hooks):
    return _TensorRef(storage, offset, shape, stride)


def _rebuild_parameter(data, requires_grad, backward_hooks):
    return data


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")

    def persistent_load(self, pid):
        storage_tag, storage_type, key, _location, _size = pid
        if storage_tag != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id: {pid}")
        return _StorageRef(key, storage_type.dtype_name)


def _dtype_for_storage(dtype_name: str):
    if dtype_name == "FloatStorage":
        return np.float32
    if dtype_name == "LongStorage":
        return np.int64
    raise ValueError(f"Unsupported storage dtype: {dtype_name}")


def _load_checkpoint_without_torch(path: Path):
    with zipfile.ZipFile(path) as archive:
        root = archive.namelist()[0].split("/", 1)[0]
        checkpoint = _CheckpointUnpickler(io.BytesIO(archive.read(f"{root}/data.pkl"))).load()

        arrays = {}
        for name, tensor in checkpoint["model_state"].items():
            dtype = _dtype_for_storage(tensor.storage.dtype_name)
            raw = archive.read(f"{root}/data/{tensor.storage.key}")
            itemsize = np.dtype(dtype).itemsize
            strides = tuple(s * itemsize for s in tensor.stride)
            array = np.ndarray(
                shape=tensor.shape,
                dtype=dtype,
                buffer=raw,
                offset=tensor.offset * itemsize,
                strides=strides if strides else None,
            ).copy()
            arrays[name] = array

    return arrays, checkpoint["classes"]


def _conv2d(x, weight, stride=1, padding=0):
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    k_h, k_w = weight.shape[2], weight.shape[3]
    windows = np.lib.stride_tricks.sliding_window_view(x, (k_h, k_w), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride, :, :]
    out = np.tensordot(windows, weight, axes=([1, 4, 5], [1, 2, 3]))
    return np.moveaxis(out, -1, 1).astype(np.float32, copy=False)


def _batch_norm(x, weight, bias, running_mean, running_var, eps=1e-5):
    scale = weight.reshape(1, -1, 1, 1) / np.sqrt(running_var.reshape(1, -1, 1, 1) + eps)
    shift = bias.reshape(1, -1, 1, 1) - running_mean.reshape(1, -1, 1, 1) * scale
    return x * scale + shift


def _relu(x):
    return np.maximum(x, 0, out=x)


def _max_pool2d(x, kernel_size=3, stride=2, padding=1):
    if padding:
        x = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=-np.inf,
        )
    windows = np.lib.stride_tricks.sliding_window_view(x, (kernel_size, kernel_size), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride, :, :]
    return windows.max(axis=(-1, -2))


class NumpyResNet18:
    def __init__(self, checkpoint_path: Path):
        self.state, self.classes = _load_checkpoint_without_torch(checkpoint_path)

    def _bn(self, x, prefix):
        return _batch_norm(
            x,
            self.state[f"{prefix}.weight"],
            self.state[f"{prefix}.bias"],
            self.state[f"{prefix}.running_mean"],
            self.state[f"{prefix}.running_var"],
        )

    def _block(self, x, prefix, stride=1, downsample=False):
        identity = x
        out = _conv2d(x, self.state[f"{prefix}.conv1.weight"], stride=stride, padding=1)
        out = _relu(self._bn(out, f"{prefix}.bn1"))
        out = _conv2d(out, self.state[f"{prefix}.conv2.weight"], stride=1, padding=1)
        out = self._bn(out, f"{prefix}.bn2")

        if downsample:
            identity = _conv2d(identity, self.state[f"{prefix}.downsample.0.weight"], stride=stride, padding=0)
            identity = self._bn(identity, f"{prefix}.downsample.1")

        out = out + identity
        return _relu(out)

    def predict_logits(self, x):
        x = _conv2d(x, self.state["conv1.weight"], stride=2, padding=3)
        x = _relu(self._bn(x, "bn1"))
        x = _max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self._block(x, "layer1.0")
        x = self._block(x, "layer1.1")
        x = self._block(x, "layer2.0", stride=2, downsample=True)
        x = self._block(x, "layer2.1")
        x = self._block(x, "layer3.0", stride=2, downsample=True)
        x = self._block(x, "layer3.1")
        x = self._block(x, "layer4.0", stride=2, downsample=True)
        x = self._block(x, "layer4.1")

        x = x.mean(axis=(2, 3))
        return x @ self.state["fc.weight"].T + self.state["fc.bias"]


def save_classes_json(checkpoint_path: Path, classes_path: Path):
    _state, classes = _load_checkpoint_without_torch(checkpoint_path)
    classes_path.write_text(json.dumps(classes), encoding="utf-8")
    return classes
