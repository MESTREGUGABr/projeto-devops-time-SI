import subprocess
import sys
import os

def run_experiment(script_path):
    print(f"\n" + "="*50)
    print(f"INICIANDO: {script_path}")
    print("="*50)
    
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)
    
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=script_dir,
        capture_output=False
    )
    
    if result.returncode == 0:
        print(f"\n[SUCESSO] {script_name} finalizado.")
    else:
        print(f"\n[ERRO] Falha na execução de {script_name}.")

if __name__ == "__main__":
    # Caminhos relativos a partir de src/
    experiments = [
        "linear/main_linear.py",
        "linear/stress_test.py",
        "non_linear/main_nl.py",
        "non_linear/stress_test.py"
    ]

    # Garante que o diretório de trabalho é o local do script
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    for exp in experiments:
        full_path = os.path.join(root_dir, exp)
        if os.path.exists(full_path):
            run_experiment(full_path)
        else:
            print(f"Aviso: Arquivo não encontrado: {full_path}")