import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_parquet_files():
    """Analisa os arquivos Parquet disponíveis"""
    
    files = [
        'part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet',
        'part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet',
        'part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet'
    ]
    
    print("=== ANÁLISE DOS DADOS ===\n")
    
    for i, file in enumerate(files, 1):
        print(f"--- ARQUIVO {i}: {file} ---")
        try:
            df = pd.read_parquet(file)
            print(f"Shape: {df.shape}")
            print(f"Colunas: {list(df.columns)}")
            print(f"Tipos de dados:")
            print(df.dtypes)
            print(f"\nPrimeiras 5 linhas:")
            print(df.head())
            print(f"\nEstatísticas descritivas:")
            print(df.describe())
            print(f"\nValores nulos:")
            print(df.isnull().sum())
            print("\n" + "="*80 + "\n")
            
        except Exception as e:
            print(f"Erro ao ler arquivo {file}: {e}")
            print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    analyze_parquet_files()
