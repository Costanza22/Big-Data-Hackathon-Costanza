# 🚀 Modelo de Previsão de Vendas - One-Click Order

## 📋 Visão Geral

Este projeto desenvolve um modelo de previsão de vendas (forecast) para apoiar o varejo na reposição de produtos. O objetivo é prever a quantidade semanal de vendas por PDV (Ponto de Venda) e SKU (Stock Keeping Unit) para as cinco semanas de janeiro/2023, utilizando como base o histórico de vendas de 2022.

## 🎯 Objetivo

Desenvolver uma solução de machine learning que:
- Preveja vendas semanais por PDV/SKU
- Apoie decisões de reposição de estoque
- Supere o baseline interno da Big Data
- Seja escalável e robusta

## 📊 Dados Utilizados

### Dados de Treino (2022)
- **Transações**: 6.5M registros com data, PDV, produto, quantidade e faturamento
- **Cadastro de Produtos**: 7.092 SKUs com categoria, descrição e atributos
- **Cadastro de PDVs**: 14.419 pontos de venda com localização e categoria

### Estrutura dos Dados
- **Período**: 52 semanas de 2022
- **Agregação**: Vendas semanais por PDV/Produto
- **Volume**: 6.2M registros de vendas semanais

## 🤖 Metodologia

### 1. **Preprocessamento**
- Conversão de datas e tipos de dados
- Agregação semanal das transações
- Tratamento de valores nulos e outliers
- Normalização de IDs para consistência

### 2. **Feature Engineering**
- **Features Temporais**: Sazonalidade (seno/cosseno)
- **Features Estatísticas**: Médias, desvios, min/max por PDV e produto
- **Features Categóricas**: Codificação de premise, categoria PDV, categoria produto
- **Features de Interação**: Produto entre médias de PDV e produto

### 3. **Modelagem**
- **Algoritmo**: Random Forest Regressor
- **Validação**: Split temporal (40 semanas treino, 12 validação)
- **Otimização**: Hiperparâmetros ajustados para o domínio
- **Métricas**: MAE e RMSE para avaliação

### 4. **Previsão**
- Geração de previsões para semanas 1-5 de janeiro/2023
- Aplicação de constraints (valores não-negativos)
- Arredondamento para números inteiros

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install -r requirements.txt
```

### Execução Rápida
```bash
python src/quick_forecast.py
```

### Execução Completa
```bash
python src/simple_forecast_model.py
```

### Saída
- `output/forecast_january_2023.csv`: Arquivo de previsões no formato solicitado

## 📁 Estrutura do Projeto

```
├── data/                           # Dados de entrada
│   ├── part-00000-tid-*.parquet   # Arquivos de dados originais
├── src/                           # Código fonte
│   ├── quick_forecast.py         # Execução rápida
│   ├── simple_forecast_model.py  # Modelo principal
│   ├── sales_forecast_model.py   # Versão avançada
│   └── analyze_data.py           # Análise exploratória
├── output/                        # Resultados
│   └── forecast_january_2023.csv # Previsões geradas
├── docs/                          # Documentação
│   └── README.md                 # Documentação técnica
├── requirements.txt              # Dependências Python
└── README.md                    # Este arquivo
```

## 📈 Resultados

### Performance do Modelo
- **Modelo**: Random Forest Regressor
- **Amostras de Treino**: 4.3M
- **Amostras de Validação**: 1.3M
- **Features**: 17 features criadas

### Previsões Geradas
- **Período**: Semanas 1-5 de janeiro/2023
- **Registros**: 5.2M previsões
- **Quantidade Total**: 133.5M unidades previstas
- **Formato**: CSV com separador ';' e encoding UTF-8
- **Colunas**: semana, pdv, produto, quantidade

## 🔧 Características Técnicas

### Robustez
- Tratamento de dados faltantes
- Validação temporal para evitar data leakage
- Constraints de negócio (valores não-negativos)

### Escalabilidade
- Processamento eficiente de grandes volumes
- Uso de paralelização (n_jobs=-1)
- Otimização de memória

### Interpretabilidade
- Feature importance ranking
- Análise de padrões temporais
- Validação com métricas de negócio

## 📊 Análise Exploratória

### Insights Principais
- Distribuição de vendas por categoria de PDV
- Padrões sazonais identificados
- Correlações entre features
- Análise de outliers e tendências

### Visualizações
- Séries temporais de vendas
- Distribuições por categoria
- Heatmaps de correlação
- Análise de feature importance

## 🎯 Próximos Passos

### Melhorias Futuras
1. **Modelos Avançados**: LSTM, Prophet, XGBoost
2. **Features Adicionais**: Promoções, feriados, clima
3. **Validação Cruzada**: Time series cross-validation
4. **Ensemble Methods**: Combinação de múltiplos modelos

### Monitoramento
- Acompanhamento da performance em produção
- Retreinamento periódico do modelo
- Ajuste de hiperparâmetros baseado em feedback


**Nota**: Este modelo foi desenvolvido especificamente para o contexto do One-Click Order e otimizado para superar o baseline interno da Big Data.
