"""Wire protocol for the LeapBot-CE cloud-edge deployment services.

Framing: 4-byte magic ``LBC1`` | 4-byte little-endian JSON header length |
UTF-8 JSON header | raw tensor bytes concatenated in header order.

Tensor bytes are the raw C-contiguous little-endian buffers (non-uint8
dtypes are reinterpreted through ``torch.view(torch.uint8)``), so the
0.34 MB bf16 planning payload crosses the wire without base64 inflation and
the receiver reconstructs exact bits.  The header carries per-tensor
``shape``/``dtype`` plus arbitrary JSON fields (timestamps, timings, session
ids).
"""

from __future__ import annotations

import json
import struct

import numpy as np
import torch

MAGIC = b"LBC1"

_DTYPE_WIDTHS = {
    "uint8": 1,
    "int8": 1,
    "bool": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}


def pack_message(header: dict, tensors: list[torch.Tensor]) -> bytes:
    """Serialize a JSON header plus raw tensor payloads into one body."""
    meta, chunks = [], []
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            else:
                raise TypeError(f"protocol carries torch/numpy arrays, got {type(tensor)!r}")
        flat = tensor.detach().cpu().contiguous()
        dtype_name = str(flat.dtype).replace("torch.", "")
        if dtype_name not in _DTYPE_WIDTHS:
            raise ValueError(f"unsupported wire dtype {dtype_name}")
        raw = flat if flat.dtype == torch.uint8 else flat.view(torch.uint8)
        meta.append({"shape": list(flat.shape), "dtype": dtype_name})
        chunks.append(raw.numpy().tobytes())
    head = json.dumps({"tensors": meta, **header}).encode("utf-8")
    return MAGIC + struct.pack("<I", len(head)) + head + b"".join(chunks)


def unpack_message(buf: bytes) -> tuple[dict, list[torch.Tensor]]:
    """Inverse of :func:`pack_message`; returns (header without "tensors", tensors)."""
    if len(buf) < 8 or buf[:4] != MAGIC:
        raise ValueError(f"bad frame magic: {buf[:4]!r}")
    (hlen,) = struct.unpack("<I", buf[4:8])
    header = json.loads(buf[8 : 8 + hlen].decode("utf-8"))
    meta = header.pop("tensors")
    tensors: list[torch.Tensor] = []
    offset = 8 + hlen
    for entry in meta:
        count = 1
        for dim in entry["shape"]:
            count *= int(dim)
        width = _DTYPE_WIDTHS[entry["dtype"]]
        blob = buf[offset : offset + count * width]
        if len(blob) != count * width:
            raise ValueError(
                f"truncated frame: tensor {entry} wants {count * width} bytes, "
                f"got {len(blob)}"
            )
        offset += count * width
        tensor = torch.frombuffer(bytearray(blob), dtype=getattr(torch, entry["dtype"]))
        tensors.append(tensor.reshape(entry["shape"]))
    return header, tensors
