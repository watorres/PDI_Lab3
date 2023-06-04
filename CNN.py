import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Ruta del banco de imágenes
ruta_banco = r'C:\BD'

# Obtener la lista de carpetas
carpetas = ['FRESA', 'MANGO', 'MANZANA', 'PERA']

# Tamaño deseado para redimensionar las imágenes
nuevo_tamaño = (512, 512)

# Parámetros para HOG
orientaciones = 9
pixeles_por_celda = (8, 8)

# Lista para almacenar las características HOG y las etiquetas
hog_features_list = []
etiquetas = []

# Procesar cada imagen en las carpetas
for carpeta in carpetas:
    # Ruta completa de la carpeta
    ruta_carpeta = os.path.join(ruta_banco, carpeta)
    
    # Obtener la lista de imágenes en la carpeta
    imagenes = os.listdir(ruta_carpeta)
    
    # Procesar cada imagen
    for imagen_nombre in imagenes:
        # Ruta completa de la imagen
        ruta_imagen = os.path.join(ruta_carpeta, imagen_nombre)
        
        # Leer la imagen
        imagen = cv2.imread(ruta_imagen)
        
        # Redimensionar la imagen
        imagen_redimensionada = cv2.resize(imagen, nuevo_tamaño)
        
        # Calcular las características HOG
        hog_features = hog(imagen_redimensionada, orientations=orientaciones, pixels_per_cell=pixeles_por_celda,
                           channel_axis=-1)
        
        # Agregar las características HOG y la etiqueta a las listas
        hog_features_list.append(hog_features)
        etiquetas.append(carpeta)

# Convertir las listas a matrices numpy
X = np.array(hog_features_list)
y = np.array(etiquetas)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape de los datos para ajustarse a la entrada de la CNN
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], 1)

# Crear y entrenar la CNN
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(1, X_train.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(len(carpetas), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict_classes(X_test)

# Calcular las métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
confusion = confusion_matrix(y_test, y_pred)
f_score = f1_score(y_test, y_pred, average='weighted')

# Imprimir los resultados
print("Convolutional Neural Network (CNN)")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Confusion Matrix:")
print(confusion)