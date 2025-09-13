import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import joblib
import os

warnings.filterwarnings('ignore')

class SalesForecastModel:
    def __init__(self):
        self.pdv_data = None
        self.transaction_data = None
        self.product_data = None
        self.weekly_sales = None
        self.model = None
        self.feature_columns = []
        
    def load_data(self):
        """Carrega os dados dos arquivos Parquet"""
        print("Carregando dados...")
        
        # Carregar dados de PDVs
        self.pdv_data = pd.read_parquet('part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet')
        print(f"PDVs carregados: {self.pdv_data.shape}")
        
        # Carregar dados de transações
        self.transaction_data = pd.read_parquet('part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet')
        print(f"Transações carregadas: {self.transaction_data.shape}")
        
        # Carregar dados de produtos
        self.product_data = pd.read_parquet('part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet')
        print(f"Produtos carregados: {self.product_data.shape}")
        
        # Converter IDs para string para consistência
        self.pdv_data['pdv'] = self.pdv_data['pdv'].astype(str)
        self.product_data['produto'] = self.product_data['produto'].astype(str)
        
    def preprocess_data(self):
        """Preprocessa os dados para análise"""
        print("Preprocessando dados...")
        
        # Converter datas
        self.transaction_data['transaction_date'] = pd.to_datetime(self.transaction_data['transaction_date'])
        self.transaction_data['reference_date'] = pd.to_datetime(self.transaction_data['reference_date'])
        
        # Filtrar apenas dados de 2022
        self.transaction_data = self.transaction_data[
            self.transaction_data['transaction_date'].dt.year == 2022
        ]
        
        # Criar coluna de semana
        self.transaction_data['week'] = self.transaction_data['transaction_date'].dt.isocalendar().week
        self.transaction_data['year'] = self.transaction_data['transaction_date'].dt.year
        
        # Converter IDs para string para evitar problemas de tipo
        self.transaction_data['internal_store_id'] = self.transaction_data['internal_store_id'].astype(str)
        self.transaction_data['internal_product_id'] = self.transaction_data['internal_product_id'].astype(str)
        
        # Agregar vendas por semana, PDV e produto
        self.weekly_sales = self.transaction_data.groupby([
            'year', 'week', 'internal_store_id', 'internal_product_id'
        ])['quantity'].sum().reset_index()
        
        # Renomear colunas para facilitar
        self.weekly_sales.columns = ['year', 'week', 'pdv', 'produto', 'quantidade']
        
        # Converter IDs para string
        self.weekly_sales['pdv'] = self.weekly_sales['pdv'].astype(str)
        self.weekly_sales['produto'] = self.weekly_sales['produto'].astype(str)
        
        print(f"Vendas semanais agregadas: {self.weekly_sales.shape}")
        print(f"Período: {self.weekly_sales['week'].min()} a {self.weekly_sales['week'].max()} de 2022")
        
    def create_features(self):
        """Cria features para o modelo"""
        print("Criando features...")
        
        # Merge com dados de PDV
        self.weekly_sales = self.weekly_sales.merge(
            self.pdv_data, 
            left_on='pdv', 
            right_on='pdv', 
            how='left'
        )
        
        # Merge com dados de produto
        self.weekly_sales = self.weekly_sales.merge(
            self.product_data, 
            left_on='produto', 
            right_on='produto', 
            how='left'
        )
        
        # Features temporais
        self.weekly_sales['week_sin'] = np.sin(2 * np.pi * self.weekly_sales['week'] / 52)
        self.weekly_sales['week_cos'] = np.cos(2 * np.pi * self.weekly_sales['week'] / 52)
        
        # Features de lag (vendas das semanas anteriores)
        for lag in [1, 2, 3, 4]:
            lag_col = f'quantidade_lag_{lag}'
            self.weekly_sales[lag_col] = self.weekly_sales.groupby(['pdv', 'produto'])['quantidade'].shift(lag)
        
        # Médias móveis
        for window in [2, 4, 8]:
            ma_col = f'quantidade_ma_{window}'
            self.weekly_sales[ma_col] = self.weekly_sales.groupby(['pdv', 'produto'])['quantidade'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
        
        # Features estatísticas por PDV
        pdv_stats = self.weekly_sales.groupby('pdv')['quantidade'].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        pdv_stats.columns = ['pdv', 'pdv_mean_qty', 'pdv_std_qty', 'pdv_min_qty', 'pdv_max_qty']
        
        self.weekly_sales = self.weekly_sales.merge(pdv_stats, on='pdv', how='left')
        
        # Features estatísticas por produto
        prod_stats = self.weekly_sales.groupby('produto')['quantidade'].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        prod_stats.columns = ['produto', 'prod_mean_qty', 'prod_std_qty', 'prod_min_qty', 'prod_max_qty']
        
        self.weekly_sales = self.weekly_sales.merge(prod_stats, on='produto', how='left')
        
        # Features categóricas
        self.weekly_sales['premise_encoded'] = pd.Categorical(self.weekly_sales['premise']).codes
        self.weekly_sales['categoria_pdv_encoded'] = pd.Categorical(self.weekly_sales['categoria_pdv']).codes
        self.weekly_sales['categoria_encoded'] = pd.Categorical(self.weekly_sales['categoria']).codes
        
        # Definir colunas de features
        self.feature_columns = [
            'week', 'week_sin', 'week_cos',
            'quantidade_lag_1', 'quantidade_lag_2', 'quantidade_lag_3', 'quantidade_lag_4',
            'quantidade_ma_2', 'quantidade_ma_4', 'quantidade_ma_8',
            'pdv_mean_qty', 'pdv_std_qty', 'pdv_min_qty', 'pdv_max_qty',
            'prod_mean_qty', 'prod_std_qty', 'prod_min_qty', 'prod_max_qty',
            'premise_encoded', 'categoria_pdv_encoded', 'categoria_encoded'
        ]
        
        print(f"Features criadas: {len(self.feature_columns)}")
        
    def train_model(self):
        """Treina o modelo de previsão"""
        print("Treinando modelo...")
        
        # Preparar dados para treino
        train_data = self.weekly_sales.dropna()
        
        X = train_data[self.feature_columns]
        y = train_data['quantidade']
        
        # Split temporal (usar primeiras 40 semanas para treino, últimas 12 para validação)
        split_week = 40
        X_train = X[train_data['week'] <= split_week]
        y_train = y[train_data['week'] <= split_week]
        X_val = X[train_data['week'] > split_week]
        y_val = y[train_data['week'] > split_week]
        
        print(f"Treino: {len(X_train)} amostras")
        print(f"Validação: {len(X_val)} amostras")
        
        # Treinar modelo (Random Forest com otimização)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Avaliar modelo
        y_pred = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 features mais importantes:")
        print(feature_importance.head(10))
        
    def generate_forecast(self):
        """Gera previsões para janeiro/2023"""
        print("Gerando previsões para janeiro/2023...")
        
        # Semanas de janeiro/2023 (1 a 5)
        forecast_weeks = [1, 2, 3, 4, 5]
        forecast_year = 2023
        
        # Obter todas as combinações únicas de PDV e produto
        unique_combinations = self.weekly_sales[['pdv', 'produto']].drop_duplicates()
        
        forecasts = []
        
        for week in forecast_weeks:
            print(f"Processando semana {week}...")
            
            # Criar base para previsão
            week_forecast = unique_combinations.copy()
            week_forecast['year'] = forecast_year
            week_forecast['week'] = week
            
            # Merge com dados de PDV e produto
            week_forecast = week_forecast.merge(
                self.pdv_data, 
                left_on='pdv', 
                right_on='pdv', 
                how='left'
            )
            week_forecast = week_forecast.merge(
                self.product_data, 
                left_on='produto', 
                right_on='produto', 
                how='left'
            )
            
            # Features temporais
            week_forecast['week_sin'] = np.sin(2 * np.pi * week_forecast['week'] / 52)
            week_forecast['week_cos'] = np.cos(2 * np.pi * week_forecast['week'] / 52)
            
            # Features de lag (usar dados históricos)
            for lag in [1, 2, 3, 4]:
                lag_col = f'quantidade_lag_{lag}'
                # Para previsão, usar médias históricas como proxy
                lag_values = self.weekly_sales.groupby(['pdv', 'produto'])['quantidade'].mean()
                week_forecast = week_forecast.merge(
                    lag_values.rename(lag_col), 
                    left_on=['pdv', 'produto'], 
                    right_index=True, 
                    how='left'
                )
            
            # Médias móveis (usar médias históricas)
            for window in [2, 4, 8]:
                ma_col = f'quantidade_ma_{window}'
                ma_values = self.weekly_sales.groupby(['pdv', 'produto'])['quantidade'].mean()
                week_forecast = week_forecast.merge(
                    ma_values.rename(ma_col), 
                    left_on=['pdv', 'produto'], 
                    right_index=True, 
                    how='left'
                )
            
            # Features estatísticas (já calculadas)
            pdv_stats = self.weekly_sales.groupby('pdv')['quantidade'].agg([
                'mean', 'std', 'min', 'max'
            ]).reset_index()
            pdv_stats.columns = ['pdv', 'pdv_mean_qty', 'pdv_std_qty', 'pdv_min_qty', 'pdv_max_qty']
            
            prod_stats = self.weekly_sales.groupby('produto')['quantidade'].agg([
                'mean', 'std', 'min', 'max'
            ]).reset_index()
            prod_stats.columns = ['produto', 'prod_mean_qty', 'prod_std_qty', 'prod_min_qty', 'prod_max_qty']
            
            week_forecast = week_forecast.merge(pdv_stats, on='pdv', how='left')
            week_forecast = week_forecast.merge(prod_stats, on='produto', how='left')
            
            # Features categóricas
            week_forecast['premise_encoded'] = pd.Categorical(week_forecast['premise']).codes
            week_forecast['categoria_pdv_encoded'] = pd.Categorical(week_forecast['categoria_pdv']).codes
            week_forecast['categoria_encoded'] = pd.Categorical(week_forecast['categoria']).codes
            
            # Preencher valores nulos
            week_forecast[self.feature_columns] = week_forecast[self.feature_columns].fillna(0)
            
            # Fazer previsões
            X_forecast = week_forecast[self.feature_columns]
            predictions = self.model.predict(X_forecast)
            
            # Garantir que as previsões sejam não-negativas
            predictions = np.maximum(predictions, 0)
            
            # Adicionar previsões ao resultado
            week_forecast['quantidade'] = predictions.astype(int)
            forecasts.append(week_forecast[['week', 'pdv', 'produto', 'quantidade']])
        
        # Combinar todas as previsões
        self.forecast_results = pd.concat(forecasts, ignore_index=True)
        
        print(f"Previsões geradas: {self.forecast_results.shape}")
        print(f"Total de previsões: {self.forecast_results['quantidade'].sum()}")
        
    def save_forecast(self, filename='forecast_january_2023.csv'):
        """Salva as previsões em formato CSV"""
        print(f"Salvando previsões em {filename}...")
        
        # Ordenar por semana, PDV e produto
        self.forecast_results = self.forecast_results.sort_values(['week', 'pdv', 'produto'])
        
        # Salvar em CSV com separador ';' e encoding UTF-8
        self.forecast_results.to_csv(
            filename, 
            sep=';', 
            index=False, 
            encoding='utf-8'
        )
        
        print(f"Arquivo salvo: {filename}")
        print(f"Primeiras 10 linhas:")
        print(self.forecast_results.head(10))
        
    def run_complete_pipeline(self):
        """Executa o pipeline completo"""
        print("=== INICIANDO PIPELINE DE PREVISÃO DE VENDAS ===\n")
        
        self.load_data()
        self.preprocess_data()
        self.create_features()
        self.train_model()
        self.generate_forecast()
        self.save_forecast()
        
        print("\n=== PIPELINE CONCLUÍDO ===")

if __name__ == "__main__":
    model = SalesForecastModel()
    model.run_complete_pipeline()
