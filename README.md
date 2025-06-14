# Proyecto-sobre-procesamiento-de-datos-sismicos
Como utilizar el codigo

1-Compile el archivo python llamado process data.py (el dataset del cenais tiene que estar descomprimido y los txt llamados 'allhitexp' y 'nothitexp' deben estar en la misma carpeta del codigo

2-Se generara un archivo csv llamado 'datos_procesados_combinados.csv'

3-Ejecutar el archivo python llamado 'entrenando_modelo.py'este se encargara de entrenar el algoritmo basandose en los datos del csv anterior dicho y de balancear los mismos utilizando tecnicas de sobremuestreo (el csv debe estar en la misma carpeta del archivo pyhon)

4-Se generara un archivo h5 llamado 'modelo_sismos.h5'

5-Ejecutar el archivo python llamado 'visualizar_modelo.py' este se encargara de visualizar el modelo ya entrenado(el archivo 'modelo_sismos.h5' debe estar en la misma carpeta del archivo python)

6-Se generaran 2 imagenes.png, una muestra el mapa de predicciones y la otra la distribucion de probabilidades, ademas de que se generara un archivo csv llamado 'resultados_predicciones.csv' que mostrara la longitud, latitud, probabilidad de la clase 0 y de la clase 1 las cuales identifican la ocurrencia de un sismos y la calse predicha y por ultimo un txt llamado arquitectura_modelo.txt que muestra la arquitectura del modelo.
