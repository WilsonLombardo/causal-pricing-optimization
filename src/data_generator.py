import os
import numpy as np
import pandas as pd

class DataGenerator:
    def __init__(self, n_samples: int = 10000, random_state: int = 42):
        self.n_samples = n_samples
        np.random.seed(random_state)

    def generate_data(self) -> pd.DataFrame:
        # Características del cliente
        ingresos = np.random.normal(50000, 15000, self.n_samples)
        edad = np.random.randint(18, 65, self.n_samples)
        compras_previas = np.random.poisson(3, self.n_samples)

        # Variable de confusión: Clientes con más compras previas reciben más descuentos
        prob_descuento = 1 / (1 + np.exp(-(compras_previas - 3))) 
        tratamiento = np.random.binomial(1, prob_descuento) # T = 1 (Descuento), T = 0 (Sin descuento)

        # Resultado (Ventas/Gasto): Depende de las características y del tratamiento
        # Efecto real del descuento: +$150 de gasto en promedio
        gasto_base = 50 + (ingresos * 0.001) + (edad * 0.5) + (compras_previas * 10)
        gasto_final = gasto_base + (tratamiento * 150) + np.random.normal(0, 20, self.n_samples)

        return pd.DataFrame({
            'ingresos': ingresos,
            'edad': edad,
            'compras_previas': compras_previas,
            'descuento_aplicado': tratamiento,
            'gasto_total': gasto_final
        })

if __name__ == "__main__":
    # 1. Obtener la ruta absoluta del directorio actual del script (src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Subir un nivel para llegar a la raíz del proyecto (causal-pricing-optimization/)
    project_root = os.path.dirname(script_dir)
    
    # 3. Definir la ruta de la carpeta de datos
    data_dir = os.path.join(project_root, 'data')
    
    # 4. Crear la carpeta 'data' si no existe
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # 5. Generar y guardar los datos
    gen = DataGenerator()
    df = gen.generate_data()
    
    file_path = os.path.join(data_dir, 'causal_dataset.csv')
    df.to_csv(file_path, index=False)
    
    print(f"Datos generados exitosamente en: {file_path}")