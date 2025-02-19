import torch
import math
from math import pi

import argparse

# global variable
dtype = torch.float64


def Fourier(N):
    i = torch.complex(torch.tensor(0, dtype=dtype), torch.tensor(1, dtype=dtype))
    w = torch.exp(-2 * pi * i / N)
    tmp = torch.arange(0, N)
    matrix = tmp[:, None] * tmp[None]
    F = w ** matrix
    return F.cuda()


def Butterworth(T, H, W, d_s=0.25, d_t=0.25, n=4):
    mask = torch.zeros((T, H, W), dtype=dtype)
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = ((d_s/d_t) * (2*t/T - 1))**2 + (2*h/H - 1) ** 2 + (2*w/W - 1) ** 2
                mask[t, h, w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask.cuda()


def Gaussian(T, H, W, d_s=0.25, d_t=0.25):
    mask = torch.zeros((T, H, W), dtype=dtype)
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = ((d_s/d_t) * (2*t/T - 1))**2 + (2*h/H - 1) ** 2 + (2*w/W - 1) ** 2
                mask[t, h, w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask.cuda()


def covariance_error(args):
    T, H, W = args.shape

    # 1. Discrete Fourier Matrix 
    F_T, F_H, F_W = Fourier(T), Fourier(H), Fourier(W)
    F = torch.kron(torch.kron(F_T, F_H), F_W)
    A, B = F.real, F.imag
    del F_T, F_H, F_W, F

    # 2. Diagonal Matrix corresponding to the Filter
    # 2.1 filter
    if args.filter == "gaussian":
        filter = Gaussian(T, H, W)
    elif args.filter == "butterworth":
        filter = Butterworth(T, H, W)
    else:
        raise ValueError(f"Expected: gaussian or butterworth. But got {args.filter}!")
    # 2.2 filter -> diagonal matrix (vector form)
    L = torch.fft.ifftshift(filter, dim=(0, 1, 2)).flatten()


    # 3. Corvariance Error
    # 3.1 PYoCo mixed noise prior
    pyoco = 0.5 * ((T-1) * T * H * W) ** 0.5
    # 3.2 FreeInit 
    P = (A * L[None] @ A + B * L[None] @ B) / torch.tensor(T * H * W, dtype=dtype)
    freeinit = torch.norm( 2 * (P - P @ P), "fro")
    # 3.3 Ours (we set "\cos\theta =1", which is the upper bound)
    Q = (A * L[None] @ B + B * L[None] @ A) / torch.tensor(T * H * W, dtype=dtype)
    freqprior = torch.norm( Q @ Q, "fro")

    print(f"Settings:\nShape:{(T, H, W)},  Filter type: {args.filter}")
    print(f"Corvariance error:")
    print(f"PYoCo:    \t{pyoco}")
    print(f"FreeInit: \t{freeinit}")
    print(f"FreqPrior:\t{freqprior}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", 
        type=str, 
        default="gaussian", 
        choices=["gaussian", "butterworth"],
        help="The type of low-pass filter"
    )
    parser.add_argument(
        "--shape",  
        type=int, 
        nargs=3, 
        required=True,
        help="The shape of latent: (frames, height, width)."
    )
    args = parser.parse_args()
    covariance_error(args)
