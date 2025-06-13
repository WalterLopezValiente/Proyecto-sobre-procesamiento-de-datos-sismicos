import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow.keras.utils import plot_model

def cargar_y_preparar_datos(archivo_csv):
    """Carga y prepara los datos para predicción"""
    df = pd.read_csv(archivo_csv)
    X = df.iloc[:, :-1].values
    
    # Normalizar datos
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshapear para CNN
    X = X.reshape(-1, 2, 4, 1)
    return X, df

def visualizar_arquitectura(modelo):
    """Visualiza la arquitectura del modelo"""
    try:
        # Mostrar resumen del modelo
        print("\nArquitectura del Modelo:")
        modelo.summary()
        
        try:
            # Intenta usar tensorflow.keras.utils.plot_model
            from tensorflow.keras.utils import plot_model
            plot_model(modelo, 
                      to_file='arquitectura_modelo.png',
                      show_shapes=True,
                      show_layer_names=True)
            print("\nArquitectura guardada en 'arquitectura_modelo.png'")
        except:
            # Si falla, guarda el resumen en un archivo de texto
            with open('arquitectura_modelo.txt', 'w') as f:
                modelo.summary(print_fn=lambda x: f.write(x + '\n'))
            print("\nArquitectura guardada en formato texto en 'arquitectura_modelo.txt'")
            
            # Sugerir instalación de dependencias
            print("\nPara visualizar la arquitectura gráficamente, intente:")
            print("1. pip uninstall pydot")
            print("2. pip uninstall pydotplus")
            print("3. pip install pydot")
            print("4. pip install graphviz")
            
    except Exception as e:
        print(f"Error al visualizar arquitectura: {e}")

def hacer_predicciones(modelo, X):
    """Realiza predicciones con el modelo"""
    predicciones = modelo.predict(X)
    return predicciones

def visualizar_resultados(predicciones, datos_originales):
    """Visualiza las predicciones"""
    # Convertir predicciones a clases
    clases_predichas = np.argmax(predicciones, axis=1)
    
    # Crear DataFrame con resultados
    resultados = pd.DataFrame({
        'Longitud': datos_originales['Longitud'],
        'Latitud': datos_originales['Latitud'],
        'Probabilidad_Clase_0': predicciones[:, 0],
        'Probabilidad_Clase_1': predicciones[:, 1],
        'Clase_Predicha': clases_predichas
    })
    
    # Visualizar mapa de predicciones
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(resultados['Longitud'], 
                         resultados['Latitud'],
                         c=resultados['Clase_Predicha'],
                         cmap='coolwarm',
                         alpha=0.6)
    plt.colorbar(scatter, label='Clase Predicha')
    plt.title('Mapa de Predicciones Sísmicas')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.grid(True)
    plt.savefig('mapa_predicciones.png')
    plt.close()
    
    # Histograma de probabilidades
    plt.figure(figsize=(10, 6))
    plt.hist(predicciones[:, 1], bins=30, alpha=0.7)
    plt.title('Distribución de Probabilidades de Predicción')
    plt.xlabel('Probabilidad')
    plt.ylabel('Frecuencia')
    plt.savefig('distribucion_probabilidades.png')
    plt.close()
    
    return resultados

def main():
    try:
        # 1. Cargar el modelo
        print("Cargando modelo...")
        modelo = load_model('modelo_sismos.h5')
        
        # 2. Visualizar arquitectura
        visualizar_arquitectura(modelo)
        
        # 3. Cargar y preparar datos
        print("\nCargando datos...")
        X, datos_originales = cargar_y_preparar_datos('datos_procesados_combinados.csv')
        
        # 4. Hacer predicciones
        print("Realizando predicciones...")
        predicciones = hacer_predicciones(modelo, X)
        
        # 5. Visualizar resultados
        print("Generando visualizaciones...")
        resultados = visualizar_resultados(predicciones, datos_originales)
        
        # 6. Guardar resultados
        resultados.to_csv('resultados_predicciones.csv', index=False)
        print("\nResultados guardados en 'resultados_predicciones.csv'")
        print("Visualizaciones guardadas como 'mapa_predicciones.png' y 'distribucion_probabilidades.png'")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()