# Estrutura dos Slides - Defesa Pública do Artigo

> **Tema:** Resiliência de Camada Física em Redes 6G: Mitigação Neural de Distorções Não-Lineares sob Incerteza de CSI
>
> **Estilo visual:** Seguir o modelo LAPA (fundo bege claro #F5F0EB, títulos em fonte monospace/bold preta, logos UFAPE no canto inferior direito, elementos decorativos geométricos nos cantos)
>
> **Tempo estimado:** 15-20 minutos de apresentação

---

## SLIDE 1 - CAPA

**Título:**
Resiliência de Camada Física em Redes 6G: Mitigação Neural de Distorções Não-Lineares sob Incerteza de CSI

**Subtítulo:**
Disciplina de Segurança da Informação - 2025.2

**Autores:**
Fernando Emidio, Emanuel Reino, Pedro William, Gustavo Wanderley, Pedro José

**Instituição:**
Universidade Federal do Agreste de Pernambuco (UFAPE) - Garanhuns, PE

**Orientador:**
Prof. Sergio Mendonça

**Logos:** UFAPE (canto inferior direito)

---

## SLIDE 2 - SUMÁRIO

**Título:** SUMÁRIO

**Itens (lista com bullets):**
- Introdução e Motivação
- Objetivo e Contribuições
- Trabalhos Relacionados
- Modelagem do Sistema
- Arquitetura de Detecção (DNN)
- Resultados Experimentais
- Conclusão e Trabalhos Futuros
- Demonstração do Repositório

---

## SLIDE 3 - INTRODUÇÃO E MOTIVAÇÃO

**Título:** INTRODUÇÃO

**Conteúdo (texto + diagrama conceitual):**

Lado esquerdo - Texto em bullet points:
- Redes 6G exigem modulações densas (16-QAM) para alta eficiência espectral
- Amplificadores de Potência (HPA) operam perto da saturação → distorções AM/AM severas
- Cenários veiculares (V2X) com hipermobilidade → estimativa de canal (CSI) imperfeita
- Receptores lineares clássicos (Zero-Forcing) colapsam nesse cenário

Lado direito - Diagrama simplificado:
```
[Alice (TX)] --HPA distortion--> [Canal V2X] --CSI imperfeita--> [Bob (RX)]
                                       |
                                  [Eve (Espião)]
```

---

## SLIDE 4 - O PROBLEMA

**Título:** O PROBLEMA

**Conteúdo visual - 3 colunas mostrando a cadeia de problemas:**

| Distorção HPA (AM/AM) | + Incerteza de CSI | = Colapso dos Receptores |
|---|---|---|
| HPA satura e comprime a constelação 16-QAM | Alta mobilidade V2X gera estimativas ruidosas do canal | ZF atinge piso de erro; ML degrada exponencialmente |

**Pergunta central (destaque):**
"Como garantir comunicação segura e confiável sob essas condições hostis?"

---

## SLIDE 5 - OBJETIVO E CONTRIBUIÇÕES

**Título:** OBJETIVO E CONTRIBUIÇÕES

**Objetivo:**
Propor uma arquitetura de Segurança de Camada Física (PLS) baseada em Redes Neurais Profundas (DNN) para cenários 6G com distorção não-linear e CSI imperfeita.

**4 Contribuições principais (cards/caixas):**

1. **Modelagem HPA + CSI Imperfeita**
   Simulação em banda base com distorção AM/AM e incerteza estocástica de canal

2. **Soft-Demapping Neural Cego**
   DNN para demapeamento bit-a-bit de 16-QAM sem conhecimento do amplificador

3. **Prova de Resiliência (Stress Test)**
   DNN supera ML sob alta variância no erro de estimação de canal

4. **Consolidação do Secrecy Gap**
   BER zero para Bob + BER 50% para Eve com FEC simples

---

## SLIDE 6 - TRABALHOS RELACIONADOS

**Título:** TRABALHOS RELACIONADOS

**Timeline/fluxo visual:**

- **Shannon (1948)** → Teoria fundamental de canais ruidosos
- **Bloch & Barros (2011)** → Formalização da Segurança de Camada Física (PLS)
- **O'Shea & Hoydis (2017)** → Deep Learning na camada física (Autoencoders neurais)
- **Ye et al. (2018)** → DNN supera métodos lineares em estimação de canal OFDM
- **Felix et al. (2018) / Kim et al. (2020)** → DL para mitigação de não-linearidades HPA
- **Ara & Kelley (2024)** → PLS nativo com demapeadores neurais (limitado a BPSK)

**Lacuna identificada (destaque em vermelho/negrito):**
"Nenhum trabalho estressou a arquitetura neural sob a tripla convergência: 16-QAM + distorção HPA + degradação de CSI"

---

## SLIDE 7 - MODELAGEM DO SISTEMA (Visão Geral)

**Título:** MODELAGEM DO SISTEMA E AMEAÇAS

**Diagrama de blocos do sistema Wiretap Channel:**

```
[Alice] → [Modulador 16-QAM] → [HPA (α=0.15)] → [Canal hB] → [Bob (DNN + FEC)]
                                        |
                                   [Canal hE]
                                        |
                                   [Eve (cega)]
```

**Equação do sinal recebido por Bob:**
yB = hB · xnl + nB

**Equação da CSI imperfeita:**
ĥB = hB + ecsi,  onde ecsi ~ CN(0, σ²e)

---

## SLIDE 8 - DISTORÇÃO DO HPA

**Título:** TRANSMISSOR E DISTORÇÃO NÃO-LINEAR (HPA)

**Equação central (destaque):**
xnl = x(1 - α|x|²),  α = 0.15

**Explicação visual:**
- Mostrar constelação 16-QAM ideal (pontos uniformes) vs. constelação distorcida pelo HPA (pontos comprimidos nas bordas)
- Símbolos da extremidade colidem com o teto de saturação
- Símbolos internos operam em regime quase-linear
- "A deformação assimétrica quebra a premissa de distâncias equidistantes dos receptores convencionais"

---

## SLIDE 9 - MODELO DE AMEAÇA (WIRETAP CHANNEL)

**Título:** O MODELO DE AMEAÇA

**Diagrama visual Alice-Bob-Eve:**

| Entidade | Papel | CSI |
|---|---|---|
| **Alice** | Transmissor | - |
| **Bob** | Receptor legítimo | Possui ĥB (estimativa ruidosa) |
| **Eve** | Interceptador passivo | Sem CSI → cega à fase do canal |

**Ponto chave (destaque):**
"A vantagem de Bob reside na assimetria de CSI. Eve sofre de ambiguidade polar múltipla e é incapaz de decodificar constelações densas."

---

## SLIDE 10 - LIMITAÇÕES DOS DETECTORES CLÁSSICOS

**Título:** LIMITAÇÕES DOS DETECTORES CLÁSSICOS

**Duas colunas:**

**Zero-Forcing (ZF):**
- Assume canal afim: x̂zf = y / ĥ
- Dupla penalidade: amplifica ruído + cego à distorção AM/AM
- Converge para Error Floor prematuro

**Maximum Likelihood (ML):**
- Fronteira teórica ótima: x̂ml = arg min |y - ĥ·Φ(x)|²
- Exige conhecimento exato da função de transferência do HPA
- Degradação exponencial quando CSI degrada (rigidez geométrica)

**Conclusão visual (seta apontando para baixo):**
"Ambos falham no cenário 6G → necessidade de abordagem data-driven"

---

## SLIDE 11 - ARQUITETURA DA DNN (Slide Principal)

**Título:** O DEMAPEADOR NEURAL PROFUNDO (DNN)

**Diagrama da rede neural (fluxo horizontal):**

```
ENTRADA (4 neurônios)          CAMADAS OCULTAS              SAÍDA (4 neurônios)
┌─────────────────┐     ┌───────┐  ┌───────┐  ┌──────┐    ┌──────────────┐
│ Re{y}           │     │  256  │  │  128  │  │  64  │    │ P(b1=1|y,ĥ) │
│ Im{y}           │ ──→ │ ReLU  │─→│ ReLU  │─→│ ReLU │──→ │ P(b2=1|y,ĥ) │
│ Re{ĥ}           │     │       │  │       │  │      │    │ P(b3=1|y,ĥ) │
│ Im{ĥ}           │     └───────┘  └───────┘  └──────┘    │ P(b4=1|y,ĥ) │
└─────────────────┘                                        │  (Sigmoid)   │
                                                           └──────────────┘
```

**Pontos chave:**
- Entrada tetradimensional: sinal recebido + CSI estimada
- Saída: probabilidade a posteriori de cada bit (Soft-Demapping)
- Otimizador: Adam, Loss: Binary Cross-Entropy (BCE)
- Treinamento a SNR fixa de 15 dB (evita virar filtro de ruído)

---

## SLIDE 12 - FEC E SECRECY GAP

**Título:** CORREÇÃO DE ERRO DIRETA (FEC)

**Conteúdo:**
- Código de repetição majoritário R = 1/3
- Propósito: provar que a linearização neural é suficiente para que um FEC rudimentar elimine erros residuais
- Se a DNN+FEC simples atinge BER=0, esquemas modernos (LDPC) garantiriam performance superior

**Diagrama de contraste:**

| | Bob (com CSI) | Eve (sem CSI) |
|---|---|---|
| DNN | Corrige distorção → BER baixo | Não consegue compensar fase |
| FEC R=1/3 | Votação majoritária elimina erros residuais → BER = 0 | Vetores de erro aleatórios → votação falha → BER ≈ 50% |

---

## SLIDE 13 - PARÂMETROS DA SIMULAÇÃO

**Título:** METODOLOGIA DE SIMULAÇÃO

**Tabela de parâmetros (reproduzir a Tabela 1 do artigo):**

| Parâmetro | Valor |
|---|---|
| Modulação | 16-QAM (Gray Code) |
| Comprimento da Mensagem | 2400 bits |
| Batch Size | 128 |
| Fator de Saturação HPA (α) | 0.15 |
| Código de Canal (FEC) | Repetição (R = 1/3) |
| Topologia DNN (Ocultas) | 256, 128, 64 neurônios |
| Ativação (Ocultas / Saída) | ReLU / Sigmoid |
| Otimizador / Learning Rate | Adam / 0.001 |
| SNR de Treinamento | 15 dB |
| Épocas | 1000 |

**Ferramentas:** PyTorch + Simulações de Monte Carlo

---

## SLIDE 14 - RESULTADO 1: ROBUSTEZ SOB INCERTEZA DE CSI

**Título:** ANÁLISE DE ROBUSTEZ SOB INCERTEZA DE CSI

**Inserir a Figura 1 do artigo (gráfico `non_linear_stress_test_zf.png`)**

**Destaques ao lado do gráfico:**
- ZF Hardware Impaired: pior desempenho em todo o espectro (Error Floor)
- ZF Linear Ideal: baseline utópico sem HPA (α=0.0)
- Sob CSI perfeita (σe=0.0): **DNN (BER=2.07%) ≈ ML (BER=2.04%)**
- Sob alta incerteza (σe ≥ 0.3): **DNN SUPERA o ML**
- A DNN é o único algoritmo tolerante à tripla convergência de desafios

---

## SLIDE 15 - RESULTADO 2: ISOLAMENTO CRIPTOGRÁFICO

**Título:** ISOLAMENTO CRIPTOGRÁFICO (SECRECY GAP)

**Inserir a Figura 2 do artigo (gráfico `non_linear_comparative_result.png`)**

**Destaques ao lado do gráfico:**
- **Eve (linha tracejada vermelha):** BER estagnado em ~50% em todo o espectro → sigilo perfeito
- **Bob DNN NL (linha azul tracejada):** BER decai com SNR
- **Bob DNN NL + FEC (linha azul sólida):** BER → 0 a partir de 20 dB
- O Secrecy Gap é mantido em todas as faixas de SNR
- "PLS validado como blindagem criptográfica autossuficiente na Camada 1"

---

## SLIDE 16 - CENÁRIO LINEAR (BASELINE BPSK)

**Título:** VALIDAÇÃO - CENÁRIO LINEAR (BASELINE)

**Inserir o gráfico `linear_comparative_result.png`**

**Explicação:**
- Cenário BPSK sem distorção HPA (validação de referência)
- DNN empata com ZF em condições ideais
- Confirma que a rede neural não degrada performance em cenários simples
- Eve mantida em BER ~50% (sigilo preservado)

---

## SLIDE 17 - CONCLUSÃO

**Título:** CONCLUSÃO

**3 pontos principais (cards/caixas com ícones):**

1. **Vulnerabilidade Comprovada**
   Receptores clássicos (ZF e ML) falham sob HPA + CSI imperfeita em cenários 6G

2. **DNN como Solução Definitiva**
   Soft-Demapping neural empata com ML sob CSI perfeita e SUPERA sob alta incerteza

3. **Secrecy Gap Consolidado**
   DNN + FEC: BER = 0 para Bob | BER = 50% para Eve
   → PLS viável como segurança nativa na Camada 1

---

## SLIDE 18 - TRABALHOS FUTUROS

**Título:** TRABALHOS FUTUROS

**3 linhas de investigação (com ícones):**

1. **Expansão para MIMO Massivo**
   Avaliação da compensação espacial de redes neurais sob beamforming e interferências correlacionadas

2. **Complexidade Temporal (Série de Volterra)**
   Arquiteturas recorrentes para compensação de ISI em circuitos de ultra-banda-larga

3. **Resiliência Contra Adversários Ativos**
   Stress tests com Smart Jamming e Spoofing de pilotos CSI no ambiente V2X

---

## SLIDE 19 - REPOSITÓRIO E REPRODUTIBILIDADE

**Título:** CÓDIGO-FONTE E REPRODUTIBILIDADE

**Conteúdo:**

**Repositório GitHub:**
`github.com/Fernando7492/projeto-devops-time-SI`

**Stack Tecnológica:**
- Python 3.10+ | PyTorch | NumPy | SciPy | Matplotlib

**Como reproduzir:**
```bash
git clone https://github.com/Fernando7492/projeto-devops-time-SI.git
cd projeto-devops-time-SI
pip install -r requirements.txt
python src/main.py
# Resultados em results/
```

**Estrutura modular:**
- `src/linear/` → Cenário BPSK (baseline)
- `src/non_linear/` → Cenário 16-QAM + HPA (desafio 6G)
- `results/` → Gráficos gerados automaticamente

**Tag de versão final:** `v1.0-final`

---

## SLIDE 20 - REFERÊNCIAS

**Título:** REFERÊNCIAS

**Listar as principais (formato compacto):**
- [1] 3GPP TR 37.885, V2X, 2019
- [2] Abdel Hakeem et al., Security 6G, Electronics, 2022
- [4] Kim et al., DL Signal Detection + HPA, IEEE WCL, 2020
- [5] Bloch & Barros, Physical-Layer Security, 2011
- [7] O'Shea & Hoydis, DL for Physical Layer, IEEE TCCN, 2017
- [8] Ye et al., DL Channel Estimation OFDM, IEEE WCL, 2018
- [13] Ara & Kelley, PLS for 6G Layer-1, IEEE Access, 2024

---

## SLIDE 21 - AGRADECIMENTOS / PERGUNTAS

**Título:** Agradecimentos

**Conteúdo:**
- Universidade Federal do Agreste de Pernambuco (UFAPE)
- Prof. Sergio Mendonça (Orientador)
- Equipe: Fernando Emidio, Emanuel Reino, Pedro William, Gustavo Wanderley, Pedro José

**Texto central:**
"Obrigado! Perguntas?"

**Logos:** UFAPE
