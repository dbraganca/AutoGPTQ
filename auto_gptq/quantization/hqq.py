import torch
from torch import uint8, int32, Tensor
import numpy as np


# Bit packing logic. format: pack/unpack_nBits_target-<uint8 or int32>
class BitPack:
    # 8-bit
    ################################################
    @staticmethod
    def pack_8bit_u8(W_q: Tensor) -> Tensor:
        return W_q.to(uint8)

    @staticmethod
    def unpack_8bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        return W_q.to(dtype)

    # 4-bit
    ################################################
    @staticmethod
    def pack_4bit_u8(W_q: Tensor) -> Tensor:  # uint8 > uint8/2
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 2)

        return (W_q[:_step] << 4) | W_q[_step:]

    @staticmethod
    def unpack_4bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:  # uint8/2 > uint8
        _step = W_q.shape[0]
        tmp = torch.empty([2 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)

        tmp[:_step] = (W_q & 0b11110000) >> 4
        tmp[_step:] = W_q & 0b00001111

        return tmp

    # 2-bit
    ################################################
    @staticmethod
    def pack_2bit_u8(W_q: Tensor) -> Tensor:  # uint8 > uint8/4
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 4)

        return (
            W_q[:_step] << 6
            | W_q[_step : 2 * _step] << 4
            | W_q[2 * _step : 3 * _step] << 2
            | W_q[3 * _step :]
        )

    @staticmethod
    def unpack_2bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        _step = W_q.shape[0]
        tmp = torch.empty([4 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)

        tmp[0 * _step : 1 * _step] = (W_q & 0b11000000) >> 6
        tmp[1 * _step : 2 * _step] = (W_q & 0b00110000) >> 4
        tmp[2 * _step : 3 * _step] = (W_q & 0b00001100) >> 2
        tmp[3 * _step : 4 * _step] = W_q & 0b00000011

        return tmp

    # 3-bit
    ################################################
    @staticmethod
    def pack_3bit_32(W_q_in: Tensor) -> Tensor:
        W_q = torch.zeros(
            [int(10 * np.ceil(W_q_in.shape[0] / 10.0)), W_q_in.shape[1]],
            device=W_q_in.device,
            dtype=int32,
        )
        W_q[: len(W_q_in)] = W_q_in
        _step = int(len(W_q) / 10)

        W_q = (
            (W_q[:_step] << 27)
            | (W_q[1 * _step : 2 * _step] << 24)
            | (W_q[2 * _step : 3 * _step] << 21)
            | (W_q[3 * _step : 4 * _step] << 18)
            | (W_q[4 * _step : 5 * _step] << 15)
            | (W_q[5 * _step : 6 * _step] << 12)
            | (W_q[6 * _step : 7 * _step] << 9)
            | (W_q[7 * _step : 8 * _step] << 6)
            | (W_q[8 * _step : 9 * _step] << 3)
            | (W_q[9 * _step : 10 * _step])
        )

        return W_q

    # A bit faster than _cat version
    @staticmethod
    def unpack_3bit_32(W_q: Tensor, dtype=uint8) -> Tensor:
        _step = W_q.shape[0]
        tmp = torch.empty([10 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)

        tmp[0 * _step : 1 * _step] = (W_q & 0b00111000000000000000000000000000) >> 27
        tmp[1 * _step : 2 * _step] = (W_q & 0b00000111000000000000000000000000) >> 24
        tmp[2 * _step : 3 * _step] = (W_q & 0b00000000111000000000000000000000) >> 21
        tmp[3 * _step : 4 * _step] = (W_q & 0b00000000000111000000000000000000) >> 18
        tmp[4 * _step : 5 * _step] = (W_q & 0b00000000000000111000000000000000) >> 15
        tmp[5 * _step : 6 * _step] = (W_q & 0b00000000000000000111000000000000) >> 12
        tmp[6 * _step : 7 * _step] = (W_q & 0b00000000000000000000111000000000) >> 9
        tmp[7 * _step : 8 * _step] = (W_q & 0b00000000000000000000000111000000) >> 6
        tmp[8 * _step : 9 * _step] = (W_q & 0b00000000000000000000000000111000) >> 3
        tmp[9 * _step : 10 * _step] = W_q & 0b00000000000000000000000000000111

        return tmp

    # 1-bit
    ################################################
    @staticmethod
    def pack_1bit_u8(W_q: Tensor) -> Tensor:
        W_q = W_q.to(uint8)
        _step = int(len(W_q) / 8)

        return (
            W_q[:_step] << 7
            | W_q[1 * _step : 2 * _step] << 6
            | W_q[2 * _step : 3 * _step] << 5
            | W_q[3 * _step : 4 * _step] << 4
            | W_q[4 * _step : 5 * _step] << 3
            | W_q[5 * _step : 6 * _step] << 2
            | W_q[6 * _step : 7 * _step] << 1
            | W_q[7 * _step : 8 * _step]
        )

    @staticmethod
    def unpack_1bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        _step = W_q.shape[0]
        tmp = torch.empty([8 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)

        tmp[0 * _step : 1 * _step] = (W_q & 0b10000000) >> 7
        tmp[1 * _step : 2 * _step] = (W_q & 0b01000000) >> 6
        tmp[2 * _step : 3 * _step] = (W_q & 0b00100000) >> 5
        tmp[3 * _step : 4 * _step] = (W_q & 0b00010000) >> 4
        tmp[4 * _step : 5 * _step] = (W_q & 0b00001000) >> 3
        tmp[5 * _step : 6 * _step] = (W_q & 0b00000100) >> 2
        tmp[6 * _step : 7 * _step] = (W_q & 0b00000010) >> 1
        tmp[7 * _step : 8 * _step] = W_q & 0b00000001

        return tmp



unpack_view_dtype = {
    "8bit_u8": uint8,
    "4bit_u8": uint8,
    "3bit_32": int32,
    "2bit_u8": uint8,
    "1bit_u8": uint8,
}

pack = {
    "8bit_u8": BitPack.pack_8bit_u8,
    "4bit_u8": BitPack.pack_4bit_u8,
    "3bit_32": BitPack.pack_3bit_32,
    "2bit_u8": BitPack.pack_2bit_u8,
    "1bit_u8": BitPack.pack_1bit_u8,
}

unpack = {
    "8bit_u8": BitPack.unpack_8bit_u8,
    "4bit_u8": BitPack.unpack_4bit_u8,
    "3bit_32": BitPack.unpack_3bit_32,
    "2bit_u8": BitPack.unpack_2bit_u8,
    "1bit_u8": BitPack.unpack_1bit_u8,
}

bit_to_packing = {
    8: "8bit_u8",
    4: "4bit_u8",
    3: "3bit_32",
    2: "2bit_u8",
    1: "1bit_u8",
}

import torch
import numpy as np
from torch import uint8, int32, float16, float32, nn, Tensor
from typing import Union


def is_divisible(val1: int, val2: int) -> bool:
    return int(val2 * np.ceil(val1 / val2)) == val1


def shrink_lp_op(x: Tensor, beta: float, lp_norm: float) -> Tensor:
    if lp_norm == 1:
        return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
    else:
        return torch.sign(x) * torch.nn.functional.relu(
            torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1)
        )

def optimize_weights(
    tensor: Tensor,
    scale: Tensor,
    zero: Tensor,
    min_max: list,
    axis: int = 0,
    device: Union[str, None] = None,
    opt_params: dict = {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20},
    verbose: bool = False,
) -> tuple:
    lp_norm, beta, kappa, iters = (
        opt_params["lp_norm"],
        opt_params["beta"],
        opt_params["kappa"],
        opt_params["iters"],
    )

    if device is None:
        device = tensor.device
    else:
        device = torch.device(device)

    dtype = float16 if (device.type == "cuda") else float32
    W_f = tensor.to(dtype=dtype, device=device)
    scale = scale.to(dtype=dtype, device=device)
    zero = zero.to(dtype=dtype, device=device)

    best_error = 1e4
    for i in range(iters):
        W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale
        W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
        beta *= kappa

        current_error = float(torch.abs(W_f - W_r).mean())
        if verbose:
            print(i, np.round(current_error, 6))
        if current_error < best_error:
            best_error = current_error
        else:
            break

    scale = scale.to(tensor.device)
    zero = zero.to(tensor.device)
    del W_f, W_q, W_r, W_e
    torch.cuda.empty_cache()

    W_q = torch.round(tensor * scale + zero).clamp(min_max[0], min_max[1])
    return W_q, scale, zero

def quantize_algo(
    tensor: Tensor,
    nbits: int = 4,
    channel_wise: bool = True,
    group_size: int = 64,
    optimize: bool = False,
    round_zero: bool = False,
    axis: int = 0,
    compute_dtype: Union[torch.dtype, None] = None,
    view_as_float: bool = False,
    device: str = "cuda",
) -> tuple:
    assert nbits in [8, 4, 3, 2, 1], (
        "nbits=" + str(nbits) + " not supported."
    )
    assert axis in [0, 1], "axis should be either 0 or 1"
    if group_size is not None:
        assert is_divisible(tensor.numel(), group_size), (
            "group_size should be divisble by the total tensor dimensions. shape: "
            + str(tensor.shape)
            + ", group_size: "
            + str(group_size)
        )

    W = tensor.float()
    shape = W.shape

    # Reshape for grouping
    if (group_size is not None) and channel_wise:
        W = (
            W.reshape([-1, group_size])
            if (axis == 1)
            else W.reshape([group_size, -1])
        )

    # Get min/max values
    if not channel_wise:
        _min, _max = W.min(), W.max()
        optimize = False
    else:
        _min = W.min(axis=axis, keepdim=True)[0]
        _max = W.max(axis=axis, keepdim=True)[0]

    max_v = 2**nbits - 1
    min_v = 0
    min_max = [min_v, max_v]

    # Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero, the scale is inverted later on.
    scale = (max_v / (_max - _min)).clamp(
        max=2e4
    )  # clamp to avoid half-precision problems
    zero = -_min * scale

    # Round zero as in: https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py#L42C9-L42C14
    if round_zero:
        zero = torch.round(zero)

    # Fine-tune weights
    if optimize:
        W_q, scale, zero = optimize_weights(
            tensor=W,
            scale=scale,
            zero=zero,
            min_max=min_max,
            axis=axis,
            device=device,
        )
    else:
        W_q = torch.round(W * scale + zero).clamp(min_max[0], min_max[1])

    # Store meta-data (we invert the scale for dequantization)
    meta = {
        "nbits": nbits,
        "group_size": group_size,
        "shape": shape,
        "scale": 1.0 / scale,
        "zero": zero,
        "axis": axis,
        "packing": bit_to_packing[nbits],
    }
    meta["unpack_view_dtype"] = unpack_view_dtype[meta["packing"]]

    # Pack bits
    meta["view_as_float"] = view_as_float

    W_q = pack[meta["packing"]](W_q)

    # cleanup
    del W, _min, _max
    torch.cuda.empty_cache()

    return W_q, meta

def quantize(
    W: Tensor,
    weight_quant_params: dict,
    scale_quant_params: dict,
    zero_quant_params: dict,
) -> None:
    quant_scale = scale_quant_params is not None
    quant_zero = zero_quant_params is not None

    # Quantize
    W_q, meta = quantize_algo(
        W,
        device="cpu",
        compute_dtype=torch.float16,
        **weight_quant_params,
    )
    meta.update({"quant_scale": quant_scale, "quant_zero": quant_zero})

    return W_q, meta


def dequantize(W_q: Tensor, meta: dict) -> Tensor:
    compute_dtype = meta["compute_dtype"] if ("compute_dtype" in meta) else float16
    if meta["packing"]:
        if meta["view_as_float"]:
            W_q = W_q.view(meta["unpack_view_dtype"])
        W_r = unpack[meta["packing"]](W_q, dtype=compute_dtype)
        if (meta["group_size"] is not None) and (meta["nbits"] == 3):
            W_r = (
                W_r[: meta["group_size"]]
                if (meta["axis"] == 0)
                else W_r[:, : meta["group_size"]]
            )
    else:
        W_r = W_q.to(compute_dtype)
    W_r = ((W_r - meta["zero"]) * meta["scale"]).reshape(meta["shape"])
    return W_r

if __name__ == "__main__":
    quant_config = {
        'weight_quant_params': {'nbits': 4, 'channel_wise': True, 'group_size': 64, 'optimize': True, 'round_zero': True, 'axis': 0, 'view_as_float': False},
        'scale_quant_params': None,
        'zero_quant_params': None}
    
    W = torch.load("W_0.pt")
    print(f"Original: {W}")

    W_q, meta = quantize(W, **quant_config)

    #print(f"Quantized: {W_q}")
    #print(meta)

    W_r = dequantize(W_q, meta)
    print(f"Dequantized: {W_r}")
    torch.save(W_r, 'HQQ_W.pt')