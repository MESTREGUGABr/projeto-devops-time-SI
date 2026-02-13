import torch
import numpy as np

class CommunicationSystem:
    def __init__(self, msg_len, device):
        self.msg_len = msg_len
        self.device = device
        self.mapping = {
            (0,0,0,0): complex(-3, -3), (0,0,0,1): complex(-3, -1),
            (0,0,1,1): complex(-3, 1),  (0,0,1,0): complex(-3, 3),
            (0,1,1,0): complex(-1, 3),  (0,1,1,1): complex(-1, 1),
            (0,1,0,1): complex(-1, -1), (0,1,0,0): complex(-1, -3),
            (1,1,0,0): complex(1, -3),  (1,1,0,1): complex(1, -1),
            (1,1,1,1): complex(1, 1),   (1,1,1,0): complex(1, 3),
            (1,0,1,0): complex(3, 3),   (1,0,1,1): complex(3, 1),
            (1,0,0,1): complex(3, -1),  (1,0,0,0): complex(3, -3)
        }

    def generate_bits(self, batch_size):
        return torch.randint(0, 2, (batch_size, self.msg_len)).float().to(self.device)

    def modulate_16qam(self, bits):
        batch_size = bits.shape[0]
        bits_cpu = bits.view(batch_size, -1, 4).cpu().numpy()
        syms = np.zeros((batch_size, bits_cpu.shape[1]), dtype=complex)
        for i in range(batch_size):
            for j in range(bits_cpu.shape[1]):
                syms[i,j] = self.mapping[tuple(bits_cpu[i,j].astype(int))]
        syms = syms / np.sqrt(10.0)
        return torch.tensor(syms.real, dtype=torch.float32).to(self.device), \
               torch.tensor(syms.imag, dtype=torch.float32).to(self.device)

    def apply_v2x_channel(self, tx_r, tx_i, snr_db):
        h_r = torch.randn_like(tx_r); h_i = torch.randn_like(tx_i)
        f = torch.sqrt(h_r**2 + h_i**2 + 1e-12); h_r, h_i = h_r/f, h_i/f
        rx_r = tx_r * h_r - tx_i * h_i
        rx_i = tx_r * h_i + tx_i * h_r
        snr_lin = 10 ** (snr_db / 10.0)
        std = np.sqrt(1.0 / (2 * snr_lin))
        return rx_r + torch.randn_like(rx_r)*std, rx_i + torch.randn_like(rx_i)*std, h_r, h_i

def apply_hpa_non_linearity(rx, ix, alpha=0.15):
    mag = torch.sqrt(rx**2 + ix**2 + 1e-12)
    dist = mag - alpha * (mag ** 3)
    return rx * (dist/mag), ix * (dist/mag)

def decode_zf_16qam(rx_r, rx_i, h_r, h_i):
    den = h_r**2 + h_i**2 + 1e-12
    eq_r = (rx_r * h_r + rx_i * h_i) / den * np.sqrt(10.0)
    eq_i = (rx_i * h_r - rx_r * h_i) / den * np.sqrt(10.0)
    r0, r1 = (eq_r > 0).float(), (torch.abs(eq_r) < 2).float()
    i0, i1 = (eq_i > 0).float(), (torch.abs(eq_i) < 2).float()
    return torch.stack((r0, r1, i0, i1), dim=2).view(rx_r.shape[0], -1)

def decode_ml_16qam(rx_r, rx_i, h_r, h_i, device):
    ref_points = [
        (-3, -3), (-3, -1), (-3, 1), (-3, 3),
        (-1, 3), (-1, 1), (-1, -1), (-1, -3),
        (1, -3), (1, -1), (1, 1), (1, 3),
        (3, 3), (3, 1), (3, -1), (3, -3)
    ]
    ref_bits = [
        [0,0,0,0], [0,0,0,1], [0,0,1,1], [0,0,1,0],
        [0,1,1,0], [0,1,1,1], [0,1,0,1], [0,1,0,0],
        [1,1,0,0], [1,1,0,1], [1,1,1,1], [1,1,1,0],
        [1,0,1,0], [1,0,1,1], [1,0,0,1], [1,0,0,0]
    ]
    
    p_real = torch.tensor([p[0] for p in ref_points], device=device).float() / np.sqrt(10.0)
    p_imag = torch.tensor([p[1] for p in ref_points], device=device).float() / np.sqrt(10.0)
    
    p_dist_r, p_dist_i = apply_hpa_non_linearity(p_real, p_imag)
    
    y_r = rx_r.view(-1, 1)
    y_i = rx_i.view(-1, 1)
    H_r = h_r.view(-1, 1)
    H_i = h_i.view(-1, 1)
    
    C_r = p_dist_r.view(1, -1)
    C_i = p_dist_i.view(1, -1)
    
    HC_r = H_r * C_r - H_i * C_i
    HC_i = H_r * C_i + H_i * C_r
    
    dist = (y_r - HC_r)**2 + (y_i - HC_i)**2
    
    min_idx = torch.argmin(dist, dim=1)
    
    bits_tensor = torch.tensor(ref_bits, device=device).float()
    decoded_bits = bits_tensor[min_idx]
    
    return decoded_bits.view(rx_r.shape[0], -1)