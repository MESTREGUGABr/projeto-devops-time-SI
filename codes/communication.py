import torch
import numpy as np

class CommunicationSystem:
    def __init__(self, msg_len, device):
        self.msg_len = msg_len
        self.device = device

    def generate_bits(self, batch_size):
        return torch.randint(0, 2, (batch_size, self.msg_len)).float().to(self.device)

    def modulate_bpsk(self, bits):
        return 2 * bits - 1

    def apply_v2x_channel(self, tx_signal, snr_db):
        batch, length = tx_signal.shape
        h_r = torch.randn(batch, length).to(self.device)
        h_i = torch.randn(batch, length).to(self.device)
        factor = torch.sqrt(h_r**2 + h_i**2)
        h_r, h_i = h_r / factor, h_i / factor
        
        rx_r = tx_signal * h_r
        rx_i = tx_signal * h_i
        
        snr_lin = 10 ** (snr_db / 10.0)
        std = np.sqrt(1.0 / (2 * snr_lin))
        
        y_r = rx_r + torch.randn_like(rx_r) * std
        y_i = rx_i + torch.randn_like(rx_i) * std
        return y_r, y_i, h_r, h_i

def decode_zf(rx_r, rx_i, h_r, h_i):
    den = h_r**2 + h_i**2 + 1e-12
    eq_r = (rx_r * h_r + rx_i * h_i) / den
    return (eq_r > 0).float()