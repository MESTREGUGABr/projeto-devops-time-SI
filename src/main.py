import subprocess
import sys
import os

def run_experiment(script_path):
    print(f"\n" + "="*60)
    print(f"EXECUTANDO: {script_path}")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(script_path))
    script_name = os.path.basename(script_path)
    
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=script_dir,
        capture_output=False
    )
    
    if result.returncode == 0:
        print(f"\n[OK] {script_name} concluído com sucesso.")
    else:
        print(f"\n[FALHA] {script_name} retornou erro {result.returncode}.")

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    experiments = [
        "linear/linear_main.py",
        "linear/stress_test.py",
        "non_linear/nl_main.py",
        "non_linear/stress_test.py"
    ]

    for exp in experiments:
        full_path = os.path.join(root_dir, exp)
        if os.path.exists(full_path):
            run_experiment(full_path)
        else:
            print(f"\n[AVISO] Arquivo não encontrado: {full_path}")

    print("\n" + "="*60)
    print("PIPELINE DE EXPERIMENTOS FINALIZADA")
    print("Verifique a pasta 'results/' na raiz do projeto para os gráficos.")
    print("="*60)