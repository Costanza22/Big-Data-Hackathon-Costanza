import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def quick_forecast():
    """Gera previsões rapidamente com modelo simplificado"""
    print("=== GERANDO PREVISÕES RÁPIDAS ===\n")
    
    # Carregar dados
    print("Carregando dados...")
    pdv_data = pd.read_parquet('../data/part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet')
    transaction_data = pd.read_parquet('../data/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet')
    product_data = pd.read_parquet('../data/part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet')
    
    # Converter IDs para string
    pdv_data['pdv'] = pdv_data['pdv'].astype(str)
    product_data['produto'] = product_data['produto'].astype(str)
    transaction_data['internal_store_id'] = transaction_data['internal_store_id'].astype(str)
    transaction_data['internal_product_id'] = transaction_data['internal_product_id'].astype(str)
    
    # Preprocessar transações
    print("Preprocessando transações...")
    transaction_data['transaction_date'] = pd.to_datetime(transaction_data['transaction_date'])
    transaction_data = transaction_data[transaction_data['transaction_date'].dt.year == 2022]
    transaction_data['week'] = transaction_data['transaction_date'].dt.isocalendar().week
    
    # Agregar vendas semanais
    weekly_sales = transaction_data.groupby([
        'week', 'internal_store_id', 'internal_product_id'
    ])['quantity'].sum().reset_index()
    
    weekly_sales.columns = ['week', 'pdv', 'produto', 'quantidade']
    weekly_sales['pdv'] = weekly_sales['pdv'].astype(str)
    weekly_sales['produto'] = weekly_sales['produto'].astype(str)
    
    print(f"Vendas semanais: {weekly_sales.shape}")
    
    # Criar features simples
    print("Criando features...")
    
    # Merge com dados de PDV e produto
    weekly_sales = weekly_sales.merge(pdv_data, on='pdv', how='left')
    weekly_sales = weekly_sales.merge(product_data, on='produto', how='left')
    
    # Features temporais
    weekly_sales['week_sin'] = np.sin(2 * np.pi * weekly_sales['week'] / 52)
    weekly_sales['week_cos'] = np.cos(2 * np.pi * weekly_sales['week'] / 52)
    
    # Features estatísticas
    pdv_stats = weekly_sales.groupby('pdv')['quantidade'].agg(['mean', 'std']).reset_index()
    pdv_stats.columns = ['pdv', 'pdv_mean', 'pdv_std']
    
    prod_stats = weekly_sales.groupby('produto')['quantidade'].agg(['mean', 'std']).reset_index()
    prod_stats.columns = ['produto', 'prod_mean', 'prod_std']
    
    weekly_sales = weekly_sales.merge(pdv_stats, on='pdv', how='left')
    weekly_sales = weekly_sales.merge(prod_stats, on='produto', how='left')
    
    # Features categóricas
    weekly_sales['premise_encoded'] = pd.Categorical(weekly_sales['premise']).codes
    weekly_sales['categoria_pdv_encoded'] = pd.Categorical(weekly_sales['categoria_pdv']).codes
    weekly_sales['categoria_encoded'] = pd.Categorical(weekly_sales['categoria']).codes
    
    # Features de interação
    weekly_sales['pdv_prod_interaction'] = weekly_sales['pdv_mean'] * weekly_sales['prod_mean']
    
    # Definir features
    feature_columns = [
        'week', 'week_sin', 'week_cos',
        'pdv_mean', 'pdv_std', 'prod_mean', 'prod_std',
        'premise_encoded', 'categoria_pdv_encoded', 'categoria_encoded',
        'pdv_prod_interaction'
    ]
    
    # Treinar modelo
    print("Treinando modelo...")
    train_data = weekly_sales.dropna()
    X = train_data[feature_columns]
    y = train_data['quantidade']
    
    # Split temporal
    split_week = 40
    X_train = X[train_data['week'] <= split_week]
    y_train = y[train_data['week'] <= split_week]
    
    model = RandomForestRegressor(
        n_estimators=50,  # Reduzido para velocidade
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print(f"Modelo treinado com {len(X_train)} amostras")
    
    # Gerar previsões
    print("Gerando previsões...")
    forecast_weeks = [1, 2, 3, 4, 5]
    unique_combinations = weekly_sales[['pdv', 'produto']].drop_duplicates()
    
    forecasts = []
    
    for week in forecast_weeks:
        print(f"Processando semana {week}...")
        
        # Criar base para previsão
        week_forecast = unique_combinations.copy()
        week_forecast['week'] = week
        
        # Merge com dados
        week_forecast = week_forecast.merge(pdv_data, on='pdv', how='left')
        week_forecast = week_forecast.merge(product_data, on='produto', how='left')
        
        # Features temporais
        week_forecast['week_sin'] = np.sin(2 * np.pi * week_forecast['week'] / 52)
        week_forecast['week_cos'] = np.cos(2 * np.pi * week_forecast['week'] / 52)
        
        # Features estatísticas
        week_forecast = week_forecast.merge(pdv_stats, on='pdv', how='left')
        week_forecast = week_forecast.merge(prod_stats, on='produto', how='left')
        
        # Features categóricas
        week_forecast['premise_encoded'] = pd.Categorical(week_forecast['premise']).codes
        week_forecast['categoria_pdv_encoded'] = pd.Categorical(week_forecast['categoria_pdv']).codes
        week_forecast['categoria_encoded'] = pd.Categorical(week_forecast['categoria']).codes
        
        # Features de interação
        week_forecast['pdv_prod_interaction'] = week_forecast['pdv_mean'] * week_forecast['prod_mean']
        
        # Preencher nulos
        week_forecast[feature_columns] = week_forecast[feature_columns].fillna(0)
        
        # Fazer previsões
        X_forecast = week_forecast[feature_columns]
        predictions = model.predict(X_forecast)
        predictions = np.maximum(predictions, 0).astype(int)
        
        week_forecast['quantidade'] = predictions
        forecasts.append(week_forecast[['week', 'pdv', 'produto', 'quantidade']])
    
    # Combinar previsões
    forecast_results = pd.concat(forecasts, ignore_index=True)
    
    # Salvar arquivo
    print("Salvando previsões...")
    forecast_results = forecast_results.sort_values(['week', 'pdv', 'produto'])
    forecast_results.to_csv('../output/forecast_january_2023.csv', sep=';', index=False, encoding='utf-8')
    
    print(f"\n✅ PREVISÕES GERADAS COM SUCESSO!")
    print(f"Arquivo: forecast_january_2023.csv")
    print(f"Registros: {len(forecast_results)}")
    print(f"Quantidade total prevista: {forecast_results['quantidade'].sum()}")
    print(f"Primeiras 10 linhas:")
    print(forecast_results.head(10))
    
    return forecast_results

if __name__ == "__main__":
    forecast_results = quick_forecast()
