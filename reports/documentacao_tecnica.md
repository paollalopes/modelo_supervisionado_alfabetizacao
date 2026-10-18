# Documentação técnica

Complementa o README com o fluxo de dados, as regras de qualidade, o dicionário de variáveis, o registro
das decisões analíticas e as condições de reprodutibilidade.

## 1. Fluxo de dados

```text
data/raw/*.csv (Bronze – 6 CSVs originais do INEP)
   │  src/preprocessing/gold.py :: construir_silver()   tipos, duplicidades, UF/região, rótulo de rede
   │  src/preprocessing/gold.py :: validar_silver()     9 regras de qualidade (PASS/FAIL)
   ▼
data/processed/gold_*.parquet (Gold)
   │  src/preprocessing/external.py                      IBGE + Atlas → tabela com 1 linha por município
   │  src/preprocessing/features.py                      bases analíticas com variáveis do ano anterior
   ▼
data/processed/base_alunos_2024 | base_municipios_2023 | base_municipios_2024 (.parquet)
   │  src/modeling/pipeline.py, train.py                 modelo de alunos (LightGBM)
   │  src/evaluation/metrics.py                          métricas e gráficos de avaliação
   │  src/modeling/municipios.py                         projeção do risco de meta + K-Means
   │  src/visualization/style.py, rotulos.py             paleta e nomes legíveis dos gráficos
   ▼
models/*.joblib · reports/*.csv/json · images/*.png
```

| Notebook | Lê | Gera |
|---|---|---|
| 01 | `data/raw/` | tabelas Gold |
| 02 | Gold e `data/external/` | bases analíticas |
| 03 | Gold, fontes externas e bases analíticas | `images/eda_*` |
| 04 | `base_alunos_2024` | modelo de alunos, `images/modelo_*`, métricas |
| 05 | `base_alunos_2024` e o modelo salvo pelo 04 | `images/interp_*`, importância SHAP |
| 06 | bases municipais | modelo de projeção, ranking, perfis, `images/estrategia_*` |

## 2. Regras de qualidade (Silver)

| Regra | Resultado |
|---|---|
| `id_aluno` + `ano` único | PASS |
| binários (`presenca`, `preenchimento_caderno`, `alfabetizado`) em {0, 1} | PASS |
| caderno não preenchido ⇒ sem proficiência | PASS |
| `alfabetizado` == (`proficiencia` ≥ 743) entre as avaliações válidas | PASS |
| caderno não preenchido ⇒ `alfabetizado` = 0 | PASS |
| taxas municipais em [0, 100] | PASS |
| metas municipais em [0, 100] | PASS |
| ano + município + rede único (avaliação) | PASS |
| ano + município único (metas) | PASS |

As regras 4 e 5 não apontam problemas de qualidade, mas **comportamentos que definem o tratamento de
vazamento**: a proficiência determina o rótulo, e os alunos ausentes (13.749, 13,2% da base) recebem
rótulo 0 sem terem sido avaliados. Por isso, a modelagem usa só as avaliações válidas.

## 3. Códigos de rede

| Código | Rótulo | Evidência |
|---|---|---|
| 2 | Estadual | — |
| 3 | Municipal | — |
| 5 | **Pública** (estadual + municipal) | Em 100% dos municípios com as três redes, a taxa da rede 5 fica entre a 2 e a 3; quando só existe a rede municipal, a rede 5 é idêntica a ela. Na análise anterior estava rotulada como "Privada". |
| 0 | Outros (não documentado) | 398 registros, todos de 2024; não utilizado. |

## 4. Dicionário de variáveis

### Modelo de alunos (`base_alunos_2024`)

| Grupo | Variável | Descrição | Fonte / ano |
|---|---|---|---|
| Alvo | `alfabetizado` | 1 se proficiência ≥ 743 | INEP 2024 |
| Aluno | `rede_ensino` | estadual ou municipal | INEP |
| Território | `sigla_uf`, `regiao` | UF e região (a partir do código IBGE) | IBGE |
| Território | `porte_municipio`, `log_populacao`, `crescimento_pop_2010_2022` | porte e dinâmica populacional | IBGE 2010/2022 |
| Território | `capital`, `latitude`, `longitude` | localização da sede | IBGE |
| Socioeconômicas | `idhm`, `idhm_educacao`, `idhm_longevidade`, `idhm_renda` | IDH Municipal e dimensões | Atlas 2010 |
| Socioeconômicas | `renda_per_capita`, `gini`, `pct_pobres`, `pct_extrema_pobreza`, `pct_criancas_extrema_pobreza` | renda, desigualdade e pobreza | Atlas 2010 |
| Socioeconômicas | `taxa_analfabetismo_18mais`, `anos_estudo_esperados`, `pct_criancas_lar_sem_fund_completo` | escolaridade do entorno | Atlas 2010 |
| Socioeconômicas | `taxa_freq_escola_4a5`, `taxa_freq_escola_5a6`, `taxa_atraso_2anos_fund` | acesso e fluxo escolar | Atlas 2010 |
| Socioeconômicas | `pct_agua_encanada`, `pct_sem_agua_esgoto_adequado`, `pct_energia_eletrica` | infraestrutura domiciliar | Atlas 2010 |
| Socioeconômicas | `mortalidade_ate_5anos`, `esperanca_vida`, `razao_dependencia`, `grau_formalizacao_trabalho`, `taxa_desocupacao`, `taxa_urbanizacao_2010` | saúde, demografia e trabalho | Atlas 2010 |
| Histórico 2023 | `taxa_alfabetizacao_{estadual,municipal,publica}_ant`, `media_portugues_{...}_ant` | resultado de cada rede no município em 2023 | INEP 2023 |
| Histórico 2023 | `taxa_rede_aluno_ant`, `media_rede_aluno_ant` | resultado de 2023 da **mesma rede do aluno** | INEP 2023 |
| Histórico 2023 | `participacao_ant`, `nivel_alfabetizacao_ant`, `taxa_uf_publica_ant` | participação, nível e resultado da UF | INEP 2023 |
| Metas | `meta_alfabetizacao_2024` a `2029`, `esforco_meta_2024` | metas pactuadas e esforço (meta de 2024 − taxa de 2023) | INEP |
| Derivadas (pipeline) | `tem_rede_estadual`, `dif_estadual_municipal_ant`, `gap_rede_aluno_vs_uf_ant`, `log_renda_per_capita` | engenharia de atributos | — |

**Excluídas por vazamento:** `proficiencia`, `peso_aluno`, `presenca`, `preenchimento_caderno`, taxas e
níveis **do mesmo ano**, `id_aluno` e `id_escola`. O motivo de cada exclusão está em
`VARIAVEIS_PROIBIDAS` (`src/preprocessing/features.py`). **Excluída pela EDA:** `caderno` (sem sinal).

**Faltantes relevantes:**
- variáveis da rede estadual (~59%): a maioria dos municípios não tem rede estadual nos anos iniciais;
- variáveis da rede pública (~11%): a base do INEP de 2023 não traz esse resultado para 564 dos 643
  municípios de São Paulo;
- variáveis do Atlas (~0,1%): os 5 municípios instalados em 2013.

A imputação usa a mediana e cria **indicadores de ausência**, porque a falta do dado é informativa.

### Modelo municipal (`base_municipios_2023` e `2024`)

| Grupo | Variáveis |
|---|---|
| Desempenho atual | `taxa_atual`, `media_atual`, `participacao_atual`, `nivel_atual`, `taxa_estadual_atual`, `media_estadual_atual`, `taxa_uf_publica_atual` |
| Meta | `meta_alvo`, `esforco_meta`, `distancia_meta_2030`, `gap_para_uf` (no classificador; a projeção não usa `meta_alvo` nem `esforco_meta`) |
| Território e socioeconomia | as mesmas do modelo de alunos, sem `sigla_uf` e `regiao` |
| Alvo | `nao_atingiu_meta` (classificação) ou `taxa_proximo_ano` (projeção) |

**Excluídas por vazamento:** `taxa_proximo_ano`, `meta_ano`, `status_meta` e `nao_atingiu_meta` como
variáveis explicativas (`VAZAMENTO_MUNICIPIO`, em `src/modeling/municipios.py`).

## 5. Decisões analíticas (registro)

| # | Decisão | Alternativas consideradas | Motivo |
|---|---|---|---|
| 1 | Reimplementar a Gold em pandas | ler o Parquet do S3 | reprodutibilidade sem credenciais de AWS ou Databricks |
| 2 | Excluir alunos sem caderno preenchido | tratá-los como não alfabetizados | o resultado deles não foi observado |
| 3 | Prever alunos de 2024 com indicadores de 2023 | usar 2023 e 2024 juntos | não há indicadores do ano anterior para 2023, e os agregados do mesmo ano vazam o resultado |
| 4 | Validação por município | validação aleatória | medir o desempenho em municípios novos (a aleatória superestima: 0,671 contra 0,664) |
| 5 | LightGBM para alunos | Logística, Random Forest, HistGB, XGBoost | empate técnico; capta interações e permite SHAP exato |
| 6 | Priorizar pela ordenação de risco | limiar de 0,70 (melhor F1 da classe "não alfabetizado") | o limiar de 0,70 alcança 90% dos não alfabetizados, mas sinaliza 82% dos alunos, com precisão de 44% (pouco acima dos 40% ao acaso) |
| 7 | Não usar o histórico por escola | taxa da escola em 2023 (suavizada) | só 47% dos alunos de 2024 têm a escola observada em 2023, com mediana de 1 aluno por escola; ROC-AUC isolada de 0,54 |
| 8 | Sem indicadores de UF no modelo municipal | com indicadores (ROC-AUC de 0,79) | eles captam o choque do RS em 2024 e o levariam para a projeção de 2025 |
| 9 | RS fora do treino da projeção | mantê-lo | a transição de 2023 para 2024 provavelmente reflete as enchentes, e não um ano letivo normal |
| 10 | Projeção por regressão + P(taxa < meta) | classificador direto | 35% dos municípios teriam esforço para 2025 fora da faixa vista no treino |
| 11 | Incerteza (σ dos erros) por porte | σ único | municípios pequenos variam muito mais (no teste, σ de 14,9 pontos nos menores contra 5,7 nos de 100 a 500 mil) |
| 12 | K-Means com k = 5 | k = 2 (maior silhueta) | k = 2 separa apenas "bom × ruim"; k = 5 gera perfis interpretáveis e acionáveis |
| 13 | Métricas identificadas pela classe | precisão, recall e F1 sempre da classe 1 | no modelo de alunos, a classe de interesse é "não alfabetizado" (classe 0); as colunas trazem o nome da classe para evitar leitura errada |

## 6. Premissas e limites dos modelos

- **Modelo de alunos:** as variáveis são do município e da rede, então alunos da mesma rede no mesmo
  município recebem a mesma probabilidade. O modelo prioriza **redes e municípios**, não crianças.
- **Modelo de projeção:** o risco é calculado como Φ((meta − taxa prevista) / σ do porte), o que supõe
  erros com distribuição aproximadamente normal. Riscos próximos de 100% indicam taxa prevista muito abaixo
  da meta, não um resultado garantido.
- **Matriz de prioridade:** os cortes (taxa de 50% e risco de 50%) são critérios de gestão e podem ser
  ajustados.
- **Interpretabilidade e perfis:** descrevem associações, não causas.

## 7. Reprodutibilidade

- Semente única `RANDOM_STATE = 42` (`src/config.py`) em divisões, validação, buscas e modelos.
- Dados de entrada versionados (`data/raw`, `data/external`); bases intermediárias regeneradas pelos
  notebooks.
- `executar_pipeline.py` executa os seis notebooks em ordem; `requirements.txt` lista as dependências.
- Os metadados de cada modelo (`reports/*_metadados.json`) registram as versões das bibliotecas usadas no
  treino.

**Ambiente de desenvolvimento:** Python 3.12.5, scikit-learn 1.9.0, LightGBM 4.7.0, XGBoost 3.4.1, pandas
2.3.3 e numpy 2.3.5.

## 8. Artefatos gerados

| Arquivo | Conteúdo |
|---|---|
| `models/modelo_alunos_lgbm.joblib` | pipeline completa do modelo de alunos |
| `models/modelo_projecao_meta_municipio.joblib` | projeção do risco de meta (regressor + σ por porte) |
| `reports/comparacao_modelos_cv.csv` | comparação dos algoritmos (alunos) |
| `reports/comparacao_modelos_municipio_cv.csv` | comparação dos classificadores (municípios) |
| `reports/metricas_teste_modelo_alunos.csv` | métricas no teste, com precisão, recall e F1 da classe "não alfabetizado" |
| `reports/indices_teste_alunos.csv` | índices do conjunto de teste, usados pelo notebook 05 |
| `reports/importancia_shap_alunos.csv` | importância SHAP por variável e grupo |
| `reports/ranking_prioridade_municipios_2025.csv` | ranking de prioridade com risco e taxa prevista (5.352 municípios) |
| `reports/municipios_risco_perfil_2025.csv` | base para painéis: risco, prioridade, perfil e coordenadas |
| `reports/*_metadados.json` | hiperparâmetros, métricas, configuração e versões de cada modelo |

> Os perfis são calculados sobre 5.396 municípios (com taxa e variação de 2024). A base para painéis tem
> 5.352, os que também têm projeção para 2025. Por isso, a contagem de cada perfil no CSV é um pouco menor
> que a do notebook 06 (por exemplo, 891 contra 899 em "Supera o contexto").
