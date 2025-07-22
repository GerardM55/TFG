import os
import unicodedata
import wave
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr  # Importar la librería para reducir el ruido
import librosa

from pydub import AudioSegment
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
import psutil

# Diccionario de carpetas y etiquetas
carpetas = {
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/agarre_cilindrico": "cilindrico",
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/agarre_gancho": "gancho",
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/agarre_obliquo": "obliquo",
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/agarre_pinza": "pinza",
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/agarre_pinza_lateral": "pinza lateral",
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/otro": "otro",
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/stop": "stop"
}

# 🏷 Etiquetas (excluyendo "otro" para el clasificador)
etiquetas = list(set(carpeta for carpeta in carpetas.values() if carpeta != "otro"))

# Normalización de audio y reducción de ruido
def normalizar_audio(ruta_audio):
    # Cargar el audio
    audio = AudioSegment.from_file(ruta_audio)
    
    # Convertir a un array numpy para usar noisereduce
    audio_data = np.array(audio.get_array_of_samples())

    # Aplicar la reducción de ruido
    audio_reducido = nr.reduce_noise(y=audio_data, sr=audio.frame_rate)

    # Convertir de nuevo a un objeto AudioSegment
    audio_filtrado = AudioSegment(
        audio_reducido.tobytes(), 
        frame_rate=audio.frame_rate, 
        sample_width=audio.sample_width, 
        channels=audio.channels
    )
    
    # Normalizar el audio
    return audio_filtrado.apply_gain(-audio_filtrado.max_dBFS)

# Limpieza de texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto

# Extraer MFCC + deltas (derivadas) con longitud fija
def extraer_mfcc(ruta_audio, max_frames=130):  # Tamaño máximo de frames a extraer
    # Cargar el archivo de audio con librosa
    y, sr = librosa.load(ruta_audio, sr=None)  # sr=None para conservar la frecuencia original

    # Extraer los MFCC (13 coeficientes)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Calcular las deltas (derivadas) y delta-deltas
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Concatenar los MFCC, deltas y delta-deltas
    mfcc_combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

    # Recortar o rellenar para obtener un número fijo de frames
    if mfcc_combined.shape[1] > max_frames:
        mfcc_combined = mfcc_combined[:, :max_frames]  # Recortar
    else:
        # Rellenar con ceros si la longitud es menor que max_frames
        padding = np.zeros((mfcc_combined.shape[0], max_frames - mfcc_combined.shape[1]))
        mfcc_combined = np.hstack([mfcc_combined, padding])

    # Normalizar las características
    mfcc_normalizado = (mfcc_combined - np.mean(mfcc_combined)) / np.std(mfcc_combined)
    
    return mfcc_normalizado.flatten()

# Procesar audios y extraer características MFCC
def procesar_audios():
    archivos, caracteristicas, etiquetas_reales = [], [], []

    for carpeta, etiqueta_real in carpetas.items():
        for archivo in os.listdir(carpeta):
            if archivo.endswith('.wav'):
                ruta_audio = os.path.join(carpeta, archivo)

                # Normalizar y filtrar ruido
                audio_normalizado = normalizar_audio(ruta_audio)
                ruta_temp = f"temp_{archivo}"
                audio_normalizado.export(ruta_temp, format="wav")

                # Extraer características MFCC
                mfcc = extraer_mfcc(ruta_temp)

                archivos.append(archivo)
                caracteristicas.append(mfcc)
                etiquetas_reales.append(etiqueta_real)

                os.remove(ruta_temp)

    return archivos, caracteristicas, etiquetas_reales

# Ejecutar el procesamiento
archivos, caracteristicas, etiquetas_reales = procesar_audios()

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(caracteristicas, etiquetas_reales, test_size=0.2, random_state=42)

# Realizar búsqueda de hiperparámetros con GridSearchCV para SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'degree': [3, 4, 5],  # Solo si usas el kernel polinómico
    'class_weight': ['balanced', None]  # Añadido para manejar clases desbalanceadas
}

grid_search = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_model = grid_search.best_estimator_

# Evaluar precisión con los mejores parámetros
y_pred_test = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Precisión del modelo con los mejores parámetros (SVM): {accuracy * 100:.2f}%")

# Mostrar predicciones
y_pred_full = best_model.predict(X_test)

# Matriz de confusión por palabra
cm = confusion_matrix(y_test, y_pred_test, labels=etiquetas + ["otro"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas + ["otro"])
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusión por Palabra (SVM)")
plt.tight_layout()
plt.show()

# Mostrar los mejores parámetros encontrados con GridSearchCV
print("Mejores parámetros encontrados (SVM):")
print(grid_search.best_params_)

# Guardar el modelo entrenado y las características MFCC
joblib.dump(best_model, 'mejor_modelo_svm+mcff.pkl')  # Guarda el modelo entrenado

# Ruta con los audios reales grabados (nuevos)
ruta_reales = r"C:\Users\gerar\Desktop\tfg\Control_de_veu\mi_env\datos_reales_para_entrenar"

# Procesar audios reales
archivos_reales, caracteristicas_reales, etiquetas_reales_reales = [], [], []

for carpeta_nombre in os.listdir(ruta_reales):
    carpeta_path = os.path.join(ruta_reales, carpeta_nombre)
    if not os.path.isdir(carpeta_path):
        continue
    etiqueta = limpiar_texto(carpeta_nombre)
    
    for archivo in os.listdir(carpeta_path):
        if archivo.endswith(".wav"):
            ruta_audio = os.path.join(carpeta_path, archivo)

            # Normalizar y reducir ruido
            audio_normalizado = normalizar_audio(ruta_audio)
            ruta_temp = f"temp_reales_{archivo}"
            audio_normalizado.export(ruta_temp, format="wav")

            # Extraer características
            mfcc = extraer_mfcc(ruta_temp)

            archivos_reales.append(archivo)
            caracteristicas_reales.append(mfcc)
            etiquetas_reales_reales.append(etiqueta)

            os.remove(ruta_temp)

# Hacer predicción con el modelo entrenado sobre los audios reales
predicciones_reales = best_model.predict(caracteristicas_reales)

# Mostrar precisión en estos audios reales
accuracy_reales = accuracy_score(etiquetas_reales_reales, predicciones_reales)
print(f"\nPrecisión en los audios REALES: {accuracy_reales * 100:.2f}%")

# Matriz de confusión de los audios reales
cm_reales = confusion_matrix(etiquetas_reales_reales, predicciones_reales, labels=etiquetas)
disp_reales = ConfusionMatrixDisplay(confusion_matrix=cm_reales, display_labels=etiquetas)
disp_reales.plot(cmap=plt.cm.Oranges, xticks_rotation=45)
plt.title("Matriz de Confusión en Audios Reales")
plt.tight_layout()
plt.show()
