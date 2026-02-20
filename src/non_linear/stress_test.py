import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models import PLS_DNN
from communication import CommunicationSystem, decode_zf_16qam, decode_ml_16qam, apply_hpa_non_linearity

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

    ber_dnn, ber_zf, ber_ml, ber_zf_lin = [], [], [], []

    print(f"--- STRESS TEST 16-QAM: IA vs ZF vs ML (SNR: {SNR_FIXA}dB) ---")
    print(f"{'CSI Erro':>10} | {'DNN (NL)':>10} | {'ML (NL)':>10} | {'ZF (NL)':>10} | {'ZF (Lin)':>10}")
    print("-" * 65)

    for error_std in CSI_ERROR_LEVELS:
        e_dnn, e_zf, e_ml, e_zf_lin, total = 0, 0, 0, 0, 0
        for _ in range(100):
            with torch.no_grad():
                bits = comm.generate_bits(100)
                
                tx_r_nl, tx_i_nl = apply_hpa_non_linearity(*comm.modulate_16qam(bits))
                rx_r_nl, rx_i_nl, hr_nl, hi_nl = comm.apply_v2x_channel(tx_r_nl, tx_i_nl, SNR_FIXA)

                tx_r_lin, tx_i_lin = apply_hpa_non_linearity(*comm.modulate_16qam(bits), alpha=0.0)
                rx_r_lin, rx_i_lin, hr_lin, hi_lin = comm.apply_v2x_channel(tx_r_lin, tx_i_lin, SNR_FIXA)

                hr_err_nl = hr_nl + torch.randn_like(hr_nl) * error_std
                hi_err_nl = hi_nl + torch.randn_like(hi_nl) * error_std
                
                hr_err_lin = hr_lin + torch.randn_like(hr_lin) * error_std
                hi_err_lin = hi_lin + torch.randn_like(hi_lin) * error_std
                
                inp_nl = torch.stack((rx_r_nl, rx_i_nl, hr_err_nl, hi_err_nl), dim=2).view(-1, 4)
                
                d_dnn = (model(inp_nl).view_as(bits) > 0.5).float()
                d_zf = decode_zf_16qam(rx_r_nl, rx_i_nl, hr_err_nl, hi_err_nl)
                d_ml = decode_ml_16qam(rx_r_nl, rx_i_nl, hr_err_nl, hi_err_nl, DEVICE)
                
                d_zf_lin = decode_zf_16qam(rx_r_lin, rx_i_lin, hr_err_lin, hi_err_lin)

                e_dnn += torch.sum(torch.abs(d_dnn - bits)).item()
                e_zf += torch.sum(torch.abs(d_zf - bits)).item()
                e_ml += torch.sum(torch.abs(d_ml - bits)).item()
                e_zf_lin += torch.sum(torch.abs(d_zf_lin - bits)).item()
                total += bits.numel()

        ber_dnn.append(e_dnn/total)
        ber_zf.append(e_zf/total)
        ber_ml.append(e_ml/total)
        ber_zf_lin.append(e_zf_lin/total)
        
        print(f"{error_std:10.2f} | {ber_dnn[-1]:10.5f} | {ber_ml[-1]:10.5f} | {ber_zf[-1]:10.5f} | {ber_zf_lin[-1]:10.5f}")

    plt.figure(figsize=(10, 6))
    plt.plot(CSI_ERROR_LEVELS, ber_dnn, 'g-o', linewidth=2, label='Bob (DNN - Proposed)')
    plt.plot(CSI_ERROR_LEVELS, ber_ml, 'k--^', label='Bob (ML - Optimal bound)')
    plt.plot(CSI_ERROR_LEVELS, ber_zf_lin, 'c--s', alpha=0.8, label='ZF (Linear Ideal Baseline)')
    plt.plot(CSI_ERROR_LEVELS, ber_zf, 'r--x', alpha=0.6, label='Bob (ZF - Hardware Impaired)')
    
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel("Nível de Ruído na Estimação de Canal (CSI Error)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"Robustez sob Distorção Não-Linear (16-QAM, SNR={SNR_FIXA}dB)")
    plt.legend()
    
    plot_path = os.path.join(RESULTS_DIR, 'non_linear_stress_test_zf.png')
    plt.savefig(plot_path)
    print(f"\n[SUCESSO] Gráfico salvo em: {plot_path}")

if __name__ == "__main__":
    run_stress_test()