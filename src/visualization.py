import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data_generator import DataGenerator
from src.causal_model import TLearner

def generate_plots():
    # Crear carpeta para imágenes si no existe
    if not os.path.exists('images'):
        os.makedirs('images')

    # 1. Preparar datos y modelo
    df = DataGenerator().generate_data()
    X = df[['ingresos', 'edad', 'compras_previas']]
    T = df['descuento_aplicado']
    y = df['gasto_total']

    learner = TLearner()
    learner.fit(X, T, y)
    cate_predictions = learner.predict_cate(X)

    # Configurar estilo visual profesional
    sns.set_theme(style="whitegrid")

    # --- Gráfico 1: Distribución del Efecto Causal (Uplift) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(cate_predictions, kde=True, color="#1f77b4", bins=40)
    plt.axvline(x=150, color='red', linestyle='--', linewidth=2, label='Efecto Real ($150)')
    plt.axvline(x=cate_predictions.mean(), color='green', linestyle='-', linewidth=2, label=f'Efecto Estimado (ATE: ${cate_predictions.mean():.2f})')
    
    plt.title('Distribución del Efecto Causal Individual (CATE)', fontsize=14, fontweight='bold')
    plt.xlabel('Efecto del Descuento en el Gasto ($)', fontsize=12)
    plt.ylabel('Frecuencia de Clientes', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/cate_distribution.png', dpi=300)
    plt.close()

    # --- Gráfico 2: Gasto Total (Con vs Sin Descuento) ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=T, y=y, palette=["#ff7f0e", "#2ca02c"])
    plt.title('Gasto Total: Control vs Tratamiento (Descuento)', fontsize=14, fontweight='bold')
    plt.xlabel('Grupo (0 = Sin Descuento, 1 = Con Descuento)', fontsize=12)
    plt.ylabel('Gasto Total ($)', fontsize=12)
    plt.tight_layout()
    plt.savefig('images/spending_comparison.png', dpi=300)
    plt.close()

    print("Imágenes generadas con éxito en la carpeta 'images/'")

if __name__ == "__main__":
    generate_plots()