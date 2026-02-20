import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from models import PLS_DNN
from communication import CommunicationSystem, apply_hpa_non_linearity
from fec_utils import encode_repetition, decode_repetition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MESSAGE_LEN = 2400 
BATCH_SIZE = 128        
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
SNR_LIST = [-5, 0, 5, 10, 15, 20, 25] 
FEC_N = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_simulation():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    comm_sys = CommunicationSystem(MESSAGE_LEN, DEVICE)
    model = PLS_DNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Hardware: {DEVICE}")
    print("\n--- TREINAMENTO (16-QAM + HPA) ---")
    model.train()
    for epoch in range(NUM_EPOCHS):
        bits = comm_sys.generate_bits(BATCH_SIZE)
        tx_r, tx_i = comm_sys.modulate_16qam(bits)
        tx_r, tx_i = apply_hpa_non_linearity(tx_r, tx_i)
        rx_r, rx_i, h_r, h_i = comm_sys.apply_v2x_channel(tx_r, tx_i, snr_db=15)
        inputs = torch.stack((rx_r, rx_i, h_r, h_i), dim=2).view(-1, 4)
        targets = bits.view(-1, 4)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss.item():.6f}")

    print("\n--- SIMULAÇÃO COMPARATIVA NÃO-LINEAR ---")
    ber_bob, ber_fec, ber_eve = [], [], []
    model.eval()
    
    for snr in SNR_LIST:
        e_b, e_f, e_e, t_raw, t_fec = 0, 0, 0, 0, 0
        for _ in range(50):
            with torch.no_grad():
                bits_orig = torch.randint(0, 2, (100, MESSAGE_LEN//FEC_N)).float().to(DEVICE)
                bits_coded = encode_repetition(bits_orig, n=FEC_N)
                tx_r, tx_i = apply_hpa_non_linearity(*comm_sys.modulate_16qam(bits_coded))
                rx_r, rx_i, h_r, h_i = comm_sys.apply_v2x_channel(tx_r, tx_i, snr)
                inputs = torch.stack((rx_r, rx_i, h_r, h_i), dim=2).view(-1, 4)
                
                preds = model(inputs).view_as(bits_coded)
                d_bob = (preds > 0.5).float()
                
                eve_r0 = (rx_r > 0).float()
                eve_r1 = (torch.abs(rx_r) < 2).float()
                eve_i0 = (rx_i > 0).float()
                eve_i1 = (torch.abs(rx_i) < 2).float()
                d_eve = torch.stack((eve_r0, eve_r1, eve_i0, eve_i1), dim=2).view_as(bits_coded)
                
                e_b += torch.sum(torch.abs(d_bob - bits_coded)).item()
                e_e += torch.sum(torch.abs(d_eve - bits_coded)).item()
                t_raw += bits_coded.numel()
                
                f_bob = decode_repetition(d_bob, n=FEC_N)
                e_f += torch.sum(torch.abs(f_bob - bits_orig)).item()
                t_fec += bits_orig.numel()

        ber_bob.append(e_b/t_raw)
        ber_fec.append(e_f/t_fec)
        ber_eve.append(e_e/t_raw)
        print(f"{snr:3d}dB | Bob: {ber_bob[-1]:.5f} | FEC: {ber_fec[-1]:.5f} | Eve: {ber_eve[-1]:.5f}")
    
    model_path = os.path.join(MODELS_DIR, 'bob_model_nl.pth')
    torch.save(model.state_dict(), model_path)
    
    plt.figure(figsize=(10, 7))
    plt.semilogy(SNR_LIST, ber_bob, 'b--', label='Bob (DNN NL)')
    plt.semilogy(SNR_LIST, ber_fec, 'b-o', linewidth=2, label='Bob (DNN NL + FEC)')
    plt.semilogy(SNR_LIST, ber_eve, 'r--', label='Eve (Passivo)')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.legend()
    
    plot_path = os.path.join(RESULTS_DIR, 'non_linear_comparative_result.png')
    plt.savefig(plot_path)

if __name__ == "__main__":
    run_simulation()