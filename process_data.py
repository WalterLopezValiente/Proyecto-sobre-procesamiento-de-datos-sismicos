import pandas as pd
import numpy as np

# Definir los nombres de las columnas según el archivo help.txt
column_names = [
    'Longitud',
    'Latitud', 
    'Rate_anual_M4.5+',
    'Rate_anual_M4.5+_ultimo_año_antes_M7',
    'Rate_anual_entre_celdas_M6',
    'Rate_anual_entre_celdas_M7',
    'Rate_M6_x_tiempo_desde_ultimo_M6',
    'Rate_M7_x_tiempo_desde_ultimo_M7',
    'Ocurrencia'
]

try:
    # Leer el archivo allhitexp.txt
    df_hits = pd.read_csv('allhitexp.txt',
                         delim_whitespace=True,
                         names=column_names,
                         header=None)
    
    # Leer el archivo nothitexp.txt
    df_nohits = pd.read_csv('nohitexp.txt',
                           delim_whitespace=True,
                           names=column_names,
                           header=None)
    
    # Combinar ambos dataframes
    df_combined = pd.concat([df_hits, df_nohits], ignore_index=True)
    
    # Guardar el dataset combinado como CSV
    df_combined.to_csv('datos_procesados_combinados.csv', index=False)
    
    print("Archivos procesados exitosamente!")
    print(f"Dimensiones del dataset combinado: {df_combined.shape}")
    print("\nDistribución de datos por origen:")
    print(df_combined['Origen'].value_counts())
    
    # Mostrar las primeras 5 filas para verificar
    print("\nPrimeras 5 filas del dataset combinado:")
    print(df_combined.head())

except Exception as e:
    print(f"Error al procesar los archivos: {e}")