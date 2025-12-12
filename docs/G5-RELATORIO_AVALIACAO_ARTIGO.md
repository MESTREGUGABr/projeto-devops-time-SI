# RELATÓRIO DE AVALIAÇÃO DO ARTIGO CIENTÍFICO - GRUPO 1

**Título:** Segurança na Camada Física em Redes 6G com Deep Learning: Análise de Lacuna e Proposta Metodológica

**Avaliador:** Mariana Costa / Sérgio Mendonça  
**Data:** 09/12/2025

---

## 1. ADEQUAÇÃO AO ESCOPO

**Avaliação:** ✅ **ADEQUADO**

O artigo enquadra-se perfeitamente na área de Redes de Computadores e Telecomunicações, com foco em Segurança da Informação para a futura geração 6G. O tema abordado — Segurança na Camada Física (PLS) utilizando Deep Learning (DL) em cenários de alta mobilidade (V2X) — é de altíssima relevância técnica e científica, dado que o suporte a cenários de hipermobilidade é um requisito central do 6G.

**Pontos Fortes:**
- Foco em tecnologias emergentes (6G e Deep Learning).
- Abordagem de um problema específico e complexo: segurança em canais de desvanecimento rápido (*fast-fading*).
- Integração de conceitos de IA na camada física (*Physical Layer AI*).

---

## 2. VERIFICAÇÃO DE ORIGINALIDADE

**Avaliação:** ✅ **ORIGINAL**

O trabalho demonstra originalidade ao identificar uma lacuna específica na literatura recente (trabalho de Ara e Kelley, 2024).

**Indícios de Originalidade:**
- Identificação de que modelos atuais (como PLS-DNN) foram validados apenas em modelos de canal estacionários (COST 259).
- Proposta inédita de validação deste modelo em cenários V2X com desvanecimento Rayleigh rápido.
- Criação de uma metodologia comparativa para medir a "lacuna de segurança" entre receptor legítimo e espião em alta mobilidade.

**Pontos de Atenção:**
- A inovação reside principalmente na *aplicação* e *validação* em um novo cenário crítico, e não necessariamente na criação de uma nova arquitetura de rede neural, visto que o modelo base é adaptado de trabalhos anteriores.

---

## 3. ESTRUTURA E CONTEÚDO

**Avaliação:** ⚠️ **BOM (PROPOSTA)**

**Estrutura Atual:**
- ✅ **Introdução:** Bem contextualizada e direta.
- ✅ **Trabalhos Relacionados:** Excelente uso de Tabela comparativa para destacar a lacuna.
- ✅ **Metodologia:** Detalhada tecnicamente (parâmetros da DNN).
- ⚠️ **Resultados:** Preliminares (apenas Fase 1 concluída).
- ⚠️ **Conclusão:** Coerente com o estágio atual do trabalho.
- ⚠️ **Referências:** Atuais e pertinentes.

**Pontos Fortes:**
- A Tabela 1 resume de forma excelente a limitação dos trabalhos atuais e o diferencial da proposta.
- A descrição da arquitetura da DNN é técnica e precisa (camadas, ativação, otimizador), favorecendo a reprodutibilidade.

**Pontos Fracos:**
- O artigo apresenta-se como uma "Proposta Metodológica" e "Análise de Lacuna". Embora cumpra o que o título promete, carece dos resultados finais das Fases 2 e 3 (cenários V2X), o que o caracteriza mais como um *Short Paper* ou *Work in Progress*.

---

## 4. CONTEXTUALIZAÇÃO (Problema e Objetivos)

**Avaliação:** ✅ **BEM DEFINIDO**

**Problema de Pesquisa:**
O problema é claramente definido: a ausência de validação experimental de modelos PLS baseados em Deep Learning em cenários de alta mobilidade e canais não-estacionários.

**Justificativa:**
A justificativa é sólida: redes veiculares (V2X) operam em ambientes de desvanecimento rápido, e modelos testados apenas em canais estáticos (como o COST 259) podem falhar nessas condições críticas.

**Objetivos:**
- **Objetivo Geral:** Propor um framework avançado de PLS para 6G com DNN.
- **Objetivos Específicos:**
  1. Replicar o baseline (Fase 1).
  2. Modificar o ambiente para V2X/Rayleigh (Fase 2).
  3. Realizar análise comparativa de BER vs SNR (Fase 3).

---

## 5. ANÁLISE DA METODOLOGIA

**Avaliação:** ✅ **EXCELENTE**

**Pontos Fortes:**
- Uso de ferramentas padrão de mercado/academia (Python, PyTorch).
- Detalhamento técnico da DNN permite reprodutibilidade:
    - *Entrada:* Vetor da mensagem.
    - *Camadas Ocultas:* 512 e 256 neurônios (ReLU).
    - *Otimizador:* Adam (lr=0.002).
- Divisão metodológica em 3 fases lógicas e incrementais.

**Reprodutibilidade:**
Alta. Os parâmetros da rede neural e do treinamento (500 épocas, SNR 10dB) são explicitados no texto.

**Desenho do Estudo:**
Adequado. A estratégia de treinar a rede "on-the-fly" com dataset sintético para corrigir distorções não-lineares é uma abordagem moderna e viável.

---

## 6. ANÁLISE DOS RESULTADOS

**Avaliação:** ⚠️ **PRELIMINAR**

**Status:** Resultados parciais apresentados.

**Análise do que foi apresentado:**
- Foi apresentada a validação da Fase 1 (Prova de Conceito/Baseline).
- A Figura 1 demonstra a convergência da função de perda (*Loss*) ao longo de 500 épocas, indicando que a rede aprendeu a mitigar o ruído AWGN.
- O sistema atingiu BER próxima de zero no cenário controlado.

**Limitação:**
- Não foram apresentados os resultados das Fases 2 e 3 (Cenário V2X e Comparativo), que constituem o cerne da lacuna identificada. O artigo comprova que o modelo funciona no cenário simples, mas ainda não provou a eficácia no cenário complexo proposto.

**Recomendação:**
Para um artigo completo (*Full Paper*), é indispensável a inclusão dos gráficos de BER vs SNR do cenário V2X. Como *Short Paper* ou proposta metodológica, o conteúdo atual é aceitável.

---

## 7. ANÁLISE DA DISCUSSÃO

**Avaliação:** ⚠️ **PARCIAL**

**Status:** Limitada aos resultados preliminares.

**Análise:**
- A discussão foca na capacidade da rede neural de aprender progressivamente a reduzir a entropia binária no canal estático.
- A conclusão reconhece corretamente que os próximos passos são críticos para validar a robustez em V2X.

**Expectativas não atendidas:**
- Faltou discutir o impacto computacional (tempo de treinamento) da geração de datasets "on-the-fly" proposta na metodologia, o que pode ser um desafio em aplicações de tempo real.

---

## 8. RESUMO E PALAVRAS-CHAVE

**Avaliação:** ✅ **ADEQUADO**

**Resumo:**
Reflete fielmente o conteúdo. Apresenta o contexto (6G/PLS), a inovação (DNN), a lacuna (falta de validação V2X) e a metodologia proposta (3 fases).

**Abstract:**
A tradução apresenta-se coerente com o resumo em português.

---

## 9. REFERÊNCIAS

**Avaliação:** ✅ **ADEQUADAS E ATUAIS**

**Pontos Fortes:**
- Utiliza referências extremamente recentes (2022, 2024), demonstrando que o trabalho está na fronteira do conhecimento.
- **Citações chave:**
    1. Ara e Kelley (2024) - Base para a arquitetura DNN e identificação da lacuna.
    2. Abdel Hakeem et al. (2022) - Requisitos de segurança 6G.

**Padronização:**
As referências seguem um padrão consistente, contendo autores, ano, título e veículo de publicação.

---

## 10. PARECER FINAL E VEREDITO

### VEREDITO: ✅ **ACEITAR COMO PROPOSTA (WORK IN PROGRESS)**

### JUSTIFICATIVA:

O artigo é tecnicamente sólido e metodologicamente rigoroso. A identificação da lacuna de pesquisa (falta de validação de PLS-DNN em alta mobilidade) é excelente e bem fundamentada. A metodologia proposta para resolver o problema é clara e reprodutível. No entanto, o trabalho classifica-se melhor como uma "Proposta Metodológica" ou "Short Paper", pois apresenta apenas resultados preliminares da fase de baseline (Fase 1), deixando a validação principal (Fase 2 e 3) para trabalhos futuros.

### PONTOS FORTES:
1. ✅ Clareza na definição da lacuna de pesquisa (Tabela 1 é um destaque).
2. ✅ Relevância do tema para redes 6G e cenários V2X.
3. ✅ Metodologia de Deep Learning descrita com precisão técnica.
4. ✅ Resultados preliminares validam a funcionalidade básica do modelo.

### PONTOS FRACOS:
1. ⚠️ Ausência de resultados experimentais do cenário V2X (o principal diferencial proposto).
2. ⚠️ Validação limitada ao canal AWGN na versão atual.

### SUGESTÕES CONCRETAS PARA MELHORIA:

#### Curto Prazo (Para esta versão):
1. **Reforçar na Introdução** que este artigo foca na definição metodológica e validação inicial (*baseline*), gerenciando a expectativa do leitor quanto aos resultados V2X.
2. **Expandir a Metodologia** detalhando qual modelo específico de canal V2X será utilizado na Fase 2 (ex: quais parâmetros Doppler serão simulados?).

#### Médio Prazo (Para trabalhos futuros/versão final):
3. **Executar as Fases 2 e 3:** Gerar e incluir os gráficos de BER vs SNR comparando o modelo DNN com métodos tradicionais em cenário de alta mobilidade.
4. **Análise de Custo Computacional:** Avaliar se o retreino "on-the-fly" é viável para aplicações de baixa latência como V2X.

### CONCLUSÃO

O trabalho é uma excelente contribuição inicial, estabelecendo uma base sólida para a investigação de segurança na camada física em 6G.

---

**Assinatura dos Avaliadores**
Mariana Costa (UPE) / Sérgio Mendonça (UFAPE) 