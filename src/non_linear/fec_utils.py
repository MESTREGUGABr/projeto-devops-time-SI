import torch

def encode_repetition(bits, n=3):
    return bits.repeat_interleave(n, dim=1)

def decode_repetition(bits, n=3):
    batch_size = bits.shape[0]
    reshaped = bits.view(batch_size, -1, n)
    return (reshaped.sum(dim=2) > (n / 2)).float()