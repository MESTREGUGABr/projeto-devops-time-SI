import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models import PLS_DNN
from communication import CommunicationSystem, decode_zf

SNR_FIXA = 10
CSI_ERROR_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_stress_test():
    os.makedirs('results', exist_ok=True)
    comm = CommunicationSystem(2100, DEVICE)
    
    model = PLS_DNN().to(DEVICE)
    model.load_state_dict(torch.load('results/bob_model.pth'))
    model.eval()

    ber_dnn, ber_zf = [], []

    print(f"--- STRESS TEST: IA vs Zero-Forcing (SNR: {SNR_FIXA}dB) ---")
    print(f"{'Erro CSI':>10} | {'BER DNN':>10} | {'BER ZF':>10}")
    print("-" * 40)

    for error_std in CSI_ERROR_LEVELS:
        e_dnn, e_zf, total = 0, 0, 0
        for _ in range(100):
            with torch.no_grad():
                bits = comm.generate_bits(100)
                tx = comm.modulate_bpsk(bits)
                rx_r, rx_i, hr, hi = comm.apply_v2x_channel(tx, SNR_FIXA)

                hr_err = hr + torch.randn_like(hr) * error_std
                hi_err = hi + torch.randn_like(hi) * error_std
                
                inp = torch.stack((rx_r, rx_i, hr_err, hi_err), dim=2).view(-1, 4)
                
                d_dnn = (model(inp).view_as(bits) > 0.5).float()
                d_zf = decode_zf(rx_r, rx_i, hr_err, hi_err)

                e_dnn += torch.sum(torch.abs(d_dnn - bits)).item()
                e_zf += torch.sum(torch.abs(d_zf - bits)).item()
                total += bits.numel()

        ber_dnn.append(e_dnn/total)
        ber_zf.append(e_zf/total)
        print(f"{error_std:10.2f} | {ber_dnn[-1]:10.5f} | {ber_zf[-1]:10.5f}")

    plt.figure(figsize=(10, 6))
    plt.plot(CSI_ERROR_LEVELS, ber_dnn, 'b-o', label='Bob (IA)')
    plt.plot(CSI_ERROR_LEVELS, ber_zf, 'k--s', label='Bob (Zero-Forcing)')
    plt.yscale('log'); plt.grid(True, which="both", ls="--")
    plt.xlabel("Nível de Ruído no CSI"); plt.ylabel("BER"); plt.legend()
    plt.title("Análise de Robustez: IA vs Equalização Clássica")
    plt.savefig("results/stress_test_zf.png")

if __name__ == "__main__":
    run_stress_test()