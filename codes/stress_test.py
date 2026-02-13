import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models import PLS_DNN
from communication import CommunicationSystem

# Configurações do teste de estresse
SNR_FIXA = 10  # Testaremos o impacto do erro em um sinal teoricamente "bom"
CSI_ERROR_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5] # % de ruído no CSI
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_stress_test():
    os.makedirs('results', exist_ok=True)
    
    # Carregar modelo e sistema (Assumindo MESSAGE_LEN=2100)
    comm_sys = CommunicationSystem(2100, DEVICE)
    model = PLS_DNN().to(DEVICE) 
    model.load_state_dict(torch.load('results/bob_model.pth'))
    model.eval()

    ber_per_csi_error = []

    print(f"--- INICIANDO STRESS TEST (SNR Fixa: {SNR_FIXA}dB) ---")
    print(f"{'Erro CSI':>10} | {'BER Bob':>10}")
    print("-" * 25)

    for error_std in CSI_ERROR_LEVELS:
        total_err, total_bits = 0, 0
        
        for _ in range(100):
            with torch.no_grad():
                bits = comm_sys.generate_bits(100)
                tx = comm_sys.modulate_bpsk(bits)
                rx_r, rx_i, h_r, h_i = comm_sys.apply_v2x_channel(tx, SNR_FIXA)

                # INJEÇÃO DE STRESS: Sujar o CSI que o Bob recebe
                h_r_noisy = h_r + torch.randn_like(h_r) * error_std
                h_i_noisy = h_i + torch.randn_like(h_i) * error_std

                inputs = torch.stack((rx_r, rx_i, h_r_noisy, h_i_noisy), dim=2).view(-1, 4)
                preds = model(inputs).view_as(bits)
                decisao = (preds > 0.5).float()
                
                total_err += torch.sum(torch.abs(decisao - bits)).item()
                total_bits += bits.numel()

        ber = total_err / total_bits
        ber_per_csi_error.append(ber)
        print(f"{error_std:10.2f} | {ber:10.5f}")

    # Plotagem do Stress Test
    plt.figure(figsize=(8, 5))
    plt.plot(CSI_ERROR_LEVELS, ber_per_csi_error, 'g-s', linewidth=2)
    plt.title(f"Resiliência da DNN a Erros de Estimação de Canal (SNR={SNR_FIXA}dB)")
    plt.xlabel("Nível de Ruído no CSI (Desvio Padrão)")
    plt.ylabel("Taxa de Erro de Bit (BER)")
    plt.grid(True, ls="--", alpha=0.7)
    plt.savefig("results/stress_test_csi.png")
    print("\nGráfico de stress salvo em 'results/stress_test_csi.png'")

if __name__ == "__main__":
    run_stress_test()