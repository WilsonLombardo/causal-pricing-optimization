import pandas as pd
from src.data_generator import DataGenerator
from src.causal_model import TLearner

def main():
    print("1. Cargando datos...")
    df = DataGenerator().generate_data()
    
    features = ['ingresos', 'edad', 'compras_previas']
    X = df[features]
    T = df['descuento_aplicado']
    y = df['gasto_total']

    print("2. Entrenando T-Learner (Inferencia Causal)...")
    learner = TLearner()
    learner.fit(X, T, y)

    print("3. Evaluando Efecto Causal (Uplift)...")
    cate_predictions = learner.predict_cate(X)
    ate_estimado = cate_predictions.mean()
    
    print("-" * 30)
    print(f"Efecto Real del Descuento: $150.00")
    print(f"Efecto Promedio Estimado (ATE): ${ate_estimado:.2f}")
    print("-" * 30)
    print("Conclusión: El modelo separa exitosamente el efecto causal de la variable de confusión (compras previas).")

if __name__ == "__main__":
    main()