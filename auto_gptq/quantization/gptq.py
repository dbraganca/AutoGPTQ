import math
import os
import time
from logging import getLogger

import torch
import torch.nn as nn
import transformers

from .quantizer import Quantizer

from .hqq import quantize as hqq_quantize
from .hqq import dequantize as hqq_dequantize

logger = getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.pytorch_utils.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = Quantizer()

    def add_batch(self, inp, out):
        if os.environ.get("DEBUG"):
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        group_size=-1,
        actorder=False,
        static_groups=False,
        L = 0.1
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        hqq_group_size = group_size if group_size != -1 else None

        hqq_quant_config = {
            'weight_quant_params': {'nbits': self.quantizer.bits, 'channel_wise': self.quantizer.perchannel, 'group_size': hqq_group_size, 'optimize': True, 'round_zero': False, 'axis': 1, 'view_as_float': False},
            'scale_quant_params': None,
            'zero_quant_params': None}
        
        Q_hqq, meta_hqq = hqq_quantize(W, **hqq_quant_config)

        W_hqq = hqq_dequantize(Q_hqq, meta_hqq)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        g_idx = []
        scale = []
        zero = []
        now_idx = 1

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + group_size)], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            W1_hqq = W_hqq[:, i1:i2].clone()
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                w_hqq = W1_hqq[:, i]
                d = Hinv1[i, i]

                if group_size != -1:
                    if not static_groups:
                        if (i1 + i) % group_size == 0:
                            self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + group_size)], weight=True)

                        if ((i1 + i) // group_size) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // group_size]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = ((w - w_hqq) * L + (w - q) * (1 - L)) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if os.environ.get("DEBUG"):
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                logger.debug(torch.sum(Losses))

        torch.cuda.synchronize()
        logger.info(f"duration: {(time.time() - tick)}")
        logger.info(f"avg loss: {torch.sum(Losses).item() / self.nsamples}")

        group_size = group_size if group_size != -1 else self.columns
        if static_groups and actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        logger.debug(f"hqq scale/zero shape: {meta_hqq['scale'].shape} hqq_zero.shape: {meta_hqq['zero'].shape}")
        logger.debug(f"scale shape: {self.quantizer.scale.shape} zero shape: {self.quantizer.zero.shape}")  

        #reshape hqq scale and zero to match the shape of the quantizer
        #meta_hqq["scale"] = meta_hqq["scale"].reshape(self.quantizer.scale.shape)
        #meta_hqq["zero"] = meta_hqq["zero"].reshape(self.quantizer.zero.shape)       

        logger.info(f"Prev self.quantizer.scale: {self.quantizer.scale[:5]} ")
        logger.info(f"Prev self.quantizer.zero: {self.quantizer.zero[:5]} ")
        logger.info(f"Prev Q: {finalQ[0][:5]} ")

        self.quantizer.scale = L*meta_hqq["scale"] + (1-L)*self.quantizer.scale
        self.quantizer.zero =L*meta_hqq["zero"] + (1-L)*self.quantizer.zero
        finalQ = L*W_hqq + (1-L)*Q

        logger.info(f"Final  self.quantizer.scale: {self.quantizer.scale[:5]} ")
        logger.info(f"Final  self.quantizer.zero: {self.quantizer.zero[:5]} ")
        logger.info(f"Final Q: {finalQ[0][:5]} ")

        self.layer.weight.data = finalQ.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)

        if os.environ.get("DEBUG"):
            logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        return scale, zero, g_idx

    def free(self):
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


__all__ = ["GPTQ"]
