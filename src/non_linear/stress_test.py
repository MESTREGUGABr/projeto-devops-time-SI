import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models import PLS_DNN
from communication import CommunicationSystem, decode_zf_16qam, apply_hpa_non_linearity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SNR_FIXA = 15
CSI_ERROR_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_stress_test():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    comm = CommunicationSystem(2400, DEVICE)
    
    model = PLS_DNN().to(DEVICE)
    model_path = os.path.join(MODELS_DIR, 'bob_model_nl.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    ber_dnn, ber_zf = [], []

    print(f"--- STRESS TEST 16-QAM: IA vs Zero-Forcing (SNR: {SNR_FIXA}dB) ---")
    for error_std in CSI_ERROR_LEVELS:
        e_dnn, e_zf, total = 0, 0, 0
        for _ in range(100):
            with torch.no_grad():
                bits = comm.generate_bits(100)
                tx_r, tx_i = apply_hpa_non_linearity(*comm.modulate_16qam(bits))
                rx_r, rx_i, hr, hi = comm.apply_v2x_channel(tx_r, tx_i, SNR_FIXA)

                hr_err = hr + torch.randn_like(hr) * error_std
                hi_err = hi + torch.randn_like(hi) * error_std
                
                inp = torch.stack((rx_r, rx_i, hr_err, hi_err), dim=2).view(-1, 4)
                d_dnn = (model(inp).view_as(bits) > 0.5).float()
                d_zf = decode_zf_16qam(rx_r, rx_i, hr_err, hi_err)

                e_dnn += torch.sum(torch.abs(d_dnn - bits)).item()
                e_zf += torch.sum(torch.abs(d_zf - bits)).item()
                total += bits.numel()

        ber_dnn.append(e_dnn/total); ber_zf.append(e_zf/total)
        print(f"CSI Error {error_std:.2f} | DNN: {ber_dnn[-1]:.5f} | ZF: {ber_zf[-1]:.5f}")

    plt.figure(figsize=(10, 6))
    plt.plot(CSI_ERROR_LEVELS, ber_dnn, 'g-o', label='Bob (DNN NL)')
    plt.plot(CSI_ERROR_LEVELS, ber_zf, 'r--s', label='Bob (Zero-Forcing)')
    plt.yscale('log'); plt.grid(True, which="both", ls="--")
    plt.xlabel("Nível de Ruído no CSI"); plt.ylabel("BER"); plt.legend()
    
    plot_path = os.path.join(RESULTS_DIR, 'non_linear_stress_test_zf.png')
    plt.savefig(plot_path)

if __name__ == "__main__":
    run_stress_test()