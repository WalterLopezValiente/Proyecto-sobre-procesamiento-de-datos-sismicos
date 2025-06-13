import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 1. Cargar y verificar los datos
df = pd.read_csv('datos_procesados_combinados.csv')
print("Clases únicas en el conjunto de datos:", np.unique(df.iloc[:, -1]))
print("Distribución de clases:\n", df.iloc[:, -1].value_counts())

# 2. Separar características (X) y etiquetas (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Verificar si tenemos suficientes clases
n_classes = len(np.unique(y))
if n_classes < 2:
    raise ValueError(f"Se necesitan al menos 2 clases. Actualmente hay {n_classes} clase(s)")

# 3. Balancear los datos con sobremuestreo
ros = RandomOverSampler(random_state=42)
X_bal, y_bal = ros.fit_resample(X, y)

# 4. Normalizar características
scaler = StandardScaler()
X_bal = scaler.fit_transform(X_bal)

# 5. Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

# 6. Convertir las etiquetas a formato categórico
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# 7. Reestructurar los datos para la CNN
X_train = X_train.reshape(-1, 2, 4, 1)
X_val = X_val.reshape(-1, 2, 4, 1)

# 8. Crear modelo CNN 2D
model = Sequential([
    Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(2, 4, 1), padding='same'),
    MaxPooling2D(pool_size=(1, 2)),
    Conv2D(64, kernel_size=(1, 2), activation='relu', padding='same'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(n_classes, activation='softmax')  # Cambiado a n_classes y softmax
])

# 9. Compilar el modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',  # Cambiado a categorical_crossentropy
    metrics=['accuracy']
)

# 10. Entrenar el modelo
try:
    history = model.fit(
        X_train, 
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # 11. Evaluar el modelo
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f'Accuracy en validación: {accuracy:.2f}')

except Exception as e:
    print(f"Error durante el entrenamiento: {str(e)}")

# Guardar el modelo
model.save('modelo_sismos.h5')