# Decodificação de LFP Hipocampal em Tarefa de Discriminação de Objetos por Similaridade — TCC

## 1. Objetivo do projeto

Este projeto de TCC investiga a atividade eletrofisiológica (LFP — *Local Field
Potential*) registrada no hipocampo de ratos Wistar durante uma tarefa de
**discriminação de objetos com diferentes níveis de similaridade** (25%, 50% e
75% de semelhança entre o objeto familiar e o objeto novo).

A ideia central é usar as sessões de **teste de memória** (exposição ao objeto
novo, trials `T1`–`T4`) para extrair *features* espectrais do sinal de LFP
(potência por banda, centroide espectral, *Phase-Amplitude Coupling*
Theta-Gama, tendência temporal do z-score, etc.) e, a partir delas, treinar
modelos de Machine Learning capazes de **classificar automaticamente em qual
condição de similaridade (25% / 50% / 75%) o animal estava**, testando se o
padrão eletrofisiológico do hipocampo carrega informação suficiente para
distinguir o grau de dificuldade da discriminação.

O trabalho se apoia na dissertação de mestrado da autora (Ana Paula de Castro
Araújo), que padronizou a tarefa comportamental de reconhecimento/discriminação
de objetos com múltiplas tentativas e investigou a expressão de c-Fos em
regiões do hipocampo e córtex parahipocampal. O TCC dá continuidade a essa
linha de pesquisa, trocando a leitura por imuno-histoquímica (c-Fos) pela
leitura direta da atividade elétrica (LFP) durante a tarefa, com apoio de
ferramentas de análise de sinais e de aprendizado de máquina.

Os trials `A1`–`A4` (aquisição/habituação) e as sessões `NOR` (reconhecimento
de objeto simples, sem gradiente de similaridade) não entram na análise
principal — o foco é exclusivamente o momento de teste da tarefa de
discriminação por similaridade.

## 2. Estrutura do que foi feito até agora

O projeto está dividido em duas grandes etapas, cada uma correspondendo a um
notebook:

```
eda_lfp.ipynb              → Pipeline de pré-processamento + EDA + extração de features do LFP
Pipeline_RF_LogReg_XGB.ipynb   → Engenharia de atributos + modelos de ML (RF, LogReg, XGBoost)
```

---

### 2.1 `eda_lfp.ipynb` — Pré-processamento, EDA e extração de features

Notebook que parte dos registros brutos de LFP (arquivos `.int`, um por rato/
condição/trial) e produz o dataset final de features (`features_all.parquet` /
`.csv`) usado depois no pipeline de ML.

Principais decisões e etapas do pipeline (fases numeradas dentro do notebook):

| Fase | O que faz |
|------|-----------|
| 0 | Imports, configurações globais e parâmetros |
| 1 | Leitura dos arquivos `.int` + anotações do CSV de acompanhamento |
| 2 | Exclusão do rato R8 (critério de exclusão do estudo) |
| 2.5 | Filtro: mantém apenas trials de **Teste (T1–T4)**; descarta aquisição (A1–A4) |
| 3 | Identificação de canais ruins (critério automático: RMS/kurtosis/std + anotações manuais no CSV) — canais são marcados, não removidos, nesta fase |
| 4 | Inspeção visual do sinal bruto por canal |
| 5 | Filtro notch (remoção de ruído 55–65 Hz) |
| 6 | Remoção de DC offset |
| 7 | Downsample do sinal (→ 1 kHz) |
| 8 | Conversão do sinal para blocos por segundo |
| 9 | Extração de features estatísticas por canal |
| 10A | PSD por Welch (Hanning, janela de 1s, 50% overlap, NFFT=1024) + potência por banda |
| 10B | Centroide espectral (frequência contínua por canal/segundo) |
| 10D | Agregação do sinal por **área anatômica** (lida dinamicamente de `planilha_ratos_R1_R7.xlsx`, permitindo adicionar novos ratos sem alterar código) — a partir daqui as análises (Welch, harmônicas, espectrograma, centroide, PAC) passam a ser feitas por área, não por canal isolado |
| 10C | Detecção do pico Theta real e de suas harmônicas — controle metodológico para o PAC (flag de alerta quando uma harmônica cai na faixa de Slow Gamma, seguindo Scheffer-Teixeira & Tort, 2016; Neves et al., 2022) |
| 11 | Espectrograma via Wavelet de Morlet |
| 12 | *Phase-Amplitude Coupling* (PAC) Theta–Gama lento e Theta–Gama rápido — inspirado em Neves et al. (2022), que associaram o PAC Theta–Fast Gamma à discriminação bem-sucedida de objeto deslocado |
| 13 | Z-score de potência por banda e por canal |
| 14 | Espectrograma multi-painel por banda |
| 15 | Figura 1 do TCC — histologia + registros eletrofisiológicos representativos |
| **Pipeline** | Execução completa e automatizada por arquivo, com seletor `RATO_PIPELINE` (roda um rato específico ou `'ALL'`) |
| **Pós-pipeline** | Cálculo da **tendência** (slope do z-score ao longo do trial, substituindo os valores por segundo por 1 valor/trial), z-score comparativo entre seções (25/50/75%), recomputação de PAC e sumário dos outputs gerados |

**Saída principal:** `features_all.parquet` / `features_all.csv` — no estado
atual, **19.285 linhas × 105 colunas**, contendo, por segundo (ou por trial,
após a etapa de tendência), as features espectrais de cada área anatômica,
identificadas por rato, condição (25%/50%/75%/NOR) e trial (T1–T4).

#### Features criadas (por área anatômica / canal, salvas em `08_features/features.parquet`)

Todas as colunas abaixo são calculadas **por área anatômica** (média dos
canais bons daquela área) e prefixadas pelo nome da área (ex.:
`CA1_pot_theta`, `DG_centroide_hz`).

| Categoria | Feature (nome da coluna) | Descrição |
|---|---|---|
| **Identificação** | `rato`, `condicao`, `trial`, `stem`, `segundo` | Metadados de cada linha (1 linha = 1 segundo de gravação de 1 área) |
| **Estatísticas temporais** (Fase 9) | `mean` | Média do sinal no segundo |
| | `std` | Desvio-padrão do sinal |
| | `rms` | Valor RMS (raiz da média quadrática) |
| | `kurtosis` | Curtose (achatamento da distribuição) |
| | `skewness` | Assimetria da distribuição |
| | `pico_a_pico` | Amplitude pico-a-pico (max − min) |
| | `pct_outlier_3s` | % de amostras além de 3 desvios-padrão da média |
| **Potência espectral por banda** (Fase 10A, Welch) | `pot_delta` | Potência na banda Delta (1–4 Hz) |
| | `pot_theta` | Potência na banda Theta (6–12 Hz) |
| | `pot_beta` | Potência na banda Beta (13–30 Hz) |
| | `pot_gamma_lento` | Potência na banda Gama Lento (25–55 Hz) |
| | `pot_gamma_rapido` | Potência na banda Gama Rápido (65–110 Hz) |
| **Razões entre bandas** (Fase 10A) | `theta_gamma_lento_ratio` | Razão Theta / Gama Lento |
| | `theta_gamma_rapido_ratio` | Razão Theta / Gama Rápido |
| | `delta_theta_ratio` | Razão Delta / Theta |
| **Complexidade espectral** (Fase 10A) | `entropia_espectral` | Entropia espectral normalizada do PSD (0–1; mede o quão "espalhada" está a energia no espectro) |
| **Frequência dominante** (Fase 10B) | `centroide_hz` | Centroide espectral — frequência média contínua (Hz), ponderada pela potência, na faixa 1–110 Hz |
| **Controle metodológico do PAC** (Fase 10C) | `theta_peak_hz` | Frequência do pico espectral dominante dentro da banda Theta |
| | `pac_sl_alerta_harmonicas` | Flag booleano: indica se uma harmônica (h3/h4) do pico Theta cai dentro da faixa de Gama Lento (±3 Hz) — usado para decidir qual PAC é mais confiável para reportar |
| **Acoplamento Fase-Amplitude (PAC)** (Fase 12) | `pac_mi_theta_gamma_lento` | Índice de Modulação (MI, Tort et al. 2010) do acoplamento Theta (fase) → Gama Lento (amplitude) |
| | `pac_mi_theta_gamma_rapido` | Índice de Modulação (MI) do acoplamento Theta (fase) → Gama Rápido (amplitude) — hipótese principal do TCC |
| **Normalização dentro do rato** (Fase 13, calculada sob demanda) | `<feature>_zscore` | Z-score de cada coluna de potência, usando a sessão/rato inteiro como baseline |
| **Dinâmica temporal por trial** (Pós-pipeline) | `<feature>_tendencia` | Slope (inclinação) da regressão linear do z-score da feature ao longo do tempo dentro do trial — resume se a potência sobe ou desce durante a sessão; substitui as colunas `_zscore` por segundo no `features_all` final; não calculada para sessões `NOR` |

> As colunas de potência (`pot_*`), razões e centroide são calculadas tanto no
> nível "PSD médio da área" quanto, na Fase 10D, agregadas a partir do sinal
> médio dos canais bons de cada área — por isso os nomes finais em
> `features_all` seguem o padrão `<área>_<feature>` (ex.: `CA3_pot_theta`,
> `LEC_theta_gamma_rapido_ratio`, `PRH_pac_mi_theta_gamma_rapido`).

#### Features derivadas na etapa de modelagem (`Pipeline_RF_LogReg_XGB.ipynb`)

Antes de treinar os modelos, o notebook de ML faz uma **engenharia de
atributos adicional**, transformando o formato longo (1 linha por segundo)
em formato largo (1 linha por `arquivo × área`):

| Feature derivada | Descrição |
|---|---|
| `<feature>_mean` | Média da feature dentro de cada janela percentual do trial |
| `<feature>_std` | Desvio-padrão da feature dentro da janela |
| `<feature>_min` | Valor mínimo da feature dentro da janela |
| `<feature>_max` | Valor máximo da feature dentro da janela |
| `<feature>_median` | Mediana da feature dentro da janela |

Essas estatísticas são calculadas para **cada uma das features espectrais
listadas acima**, dentro de janelas percentuais da gravação (parametrizadas
por `passo_pct`, testado entre 5% e 50% da duração do trial — ex.: 10 janelas
de 10% cada). Em seguida, atributos redundantes (correlação absoluta > 0,8)
são removidos, e o `passo_pct` ótimo é escolhido por busca de hiperparâmetros
específica para cada modelo (RF, LogReg, XGBoost).

---

### 2.2 `Pipeline_RF_LogReg_XGB.ipynb` — Modelagem (Machine Learning)

Notebook que parte do `features_all.csv` gerado na etapa anterior e treina
modelos **multiclasse nativos** para prever a condição de similaridade (`25%`,
`50%`, `75%`) a partir das features de LFP — as sessões `NOR` e gravações
muito curtas são removidas nesta etapa.

**Fluxo do notebook:**

1. Configuração e carregamento dos dados brutos (`features_all.csv`)
2. Limpeza (remove sessões `NOR` e gravações curtas)
3. Engenharia de atributos, parametrizada por `n_bins` / `passo_pct` (testando
   diferentes granularidades de agregação temporal — candidatos de
   `passo_pct`: 5 a 50%, em passos de 5)
4. Funções genéricas reutilizáveis: pipeline multiclasse, busca de
   hiperparâmetros, treino/avaliação via **Leave-One-Rat-Out** (validação
   cruzada em que cada rato é deixado de fora em um fold — evita vazamento de
   dados entre treino e teste do mesmo animal) e cálculo de importância SHAP
5. **RandomForest** — pipeline completo
6. **Regressão Logística** — pipeline completo
7. **XGBoost** — pipeline completo
8. Teste de permutação (para o RF, validando que o desempenho é
   significativamente melhor que o acaso)
9. Assinatura eletrofisiológica por **área anatômica** (RF treinado
   separadamente por área, para identificar quais regiões carregam mais
   informação discriminativa)
10. Comparação final entre os três modelos e exportação dos resultados

**Metodologia comum aos três modelos (RF, LogReg, XGBoost):**

1. Escolhe o melhor `passo_pct` via busca de hiperparâmetros (grid mediano),
   sem seleção de features;
2. Com o `passo_pct` vencedor, treina o modelo **com** e **sem** seleção de
   features (mesmo grid) e mantém a variante com melhor `balanced_accuracy`
   em validação cruzada;
3. Treina/avalia via **Leave-One-Rat-Out**, reportando as *top features*;
4. Calcula e plota a importância **SHAP** do modelo final.

> A seleção de features, quando usada, é feita **dentro** do `Pipeline` do
> scikit-learn (refeita a cada fold de treino), evitando vazamento de dados
> entre treino e teste.

**Métricas usadas:** `balanced_accuracy`, F1, precisão, recall, matriz de
confusão e `classification_report`, comparadas ao nível de chance (1/3, por
serem 3 classes).

---

## 3. Arquivos de apoio do projeto

- `planilha_ratos_R1_R7.xlsx` — mapeamento de rato → canal → área anatômica,
  usado dinamicamente pelo pipeline de EDA (fase 10D) para permitir adicionar
  novos ratos sem alterar código.
- `Dicionario_de_dados_1.pdf` — dicionário de dados das variáveis do projeto.
- `AnaPaulaDeCastroAraujo_Tese.pdf` — dissertação de mestrado que originou a
  padronização da tarefa comportamental e fundamenta a hipótese do TCC.
- `fncel171144260.pdf`, `fnbeh16970083.pdf` — artigos de referência sobre PAC
  Theta-Gama e oscilações hipocampais, usados para embasar as escolhas
  metodológicas (detecção de harmônicas, interpretação de slow vs. fast
  gamma, etc.).

## 4. Estado atual / próximos passos

- [x] Pipeline de pré-processamento do LFP (filtragem, downsample, remoção de
      ruído) — concluído (`eda_lfp`)
- [x] Extração de features espectrais por área anatômica (Welch, centroide,
      PAC, tendência) — concluído
- [x] Consolidação do dataset `features_all` (19.285 × 105) — concluído
- [x] Pipeline de modelagem multiclasse (RF, LogReg, XGBoost) com
      Leave-One-Rat-Out, seleção de features e SHAP — implementado
- [ ] Consolidar e interpretar os resultados finais por modelo e por área
      anatômica para a redação do TCC
- [ ] Redigir a discussão relacionando os achados de ML com a literatura de
      PAC Theta-Gama e discriminação de objetos por similaridade

---