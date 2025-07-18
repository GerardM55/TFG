import os
import unicodedata
import wave
import joblib
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr
import librosa

from pydub import AudioSegment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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

# Etiquetas (excluyendo "otro")
etiquetas = list(set(carpeta for carpeta in carpetas.values() if carpeta != "otro"))

# Normalización y reducción de ruido
def normalizar_audio(ruta_audio):
    audio = AudioSegment.from_file(ruta_audio)
    audio_data = np.array(audio.get_array_of_samples())
    audio_reducido = nr.reduce_noise(y=audio_data, sr=audio.frame_rate)
    audio_filtrado = AudioSegment(
        audio_reducido.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    return audio_filtrado.apply_gain(-audio_filtrado.max_dBFS)

# Limpieza de texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto

# Extraer MFCC + deltas con longitud fija
def extraer_mfcc(ruta_audio, max_frames=130):
    y, sr = librosa.load(ruta_audio, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

    if mfcc_combined.shape[1] > max_frames:
        mfcc_combined = mfcc_combined[:, :max_frames]
    else:
        padding = np.zeros((mfcc_combined.shape[0], max_frames - mfcc_combined.shape[1]))
        mfcc_combined = np.hstack([mfcc_combined, padding])

    mfcc_normalizado = (mfcc_combined - np.mean(mfcc_combined)) / np.std(mfcc_combined)
    return mfcc_normalizado.flatten()

# Procesar audios y extraer características MFCC
def procesar_audios():
    archivos, caracteristicas, etiquetas_reales = [], [], []

    for carpeta, etiqueta_real in carpetas.items():
        for archivo in os.listdir(carpeta):
            if archivo.endswith('.wav'):
                ruta_audio = os.path.join(carpeta, archivo)
                audio_normalizado = normalizar_audio(ruta_audio)
                ruta_temp = f"temp_{archivo}"
                audio_normalizado.export(ruta_temp, format="wav")

                mfcc = extraer_mfcc(ruta_temp)
                archivos.append(archivo)
                caracteristicas.append(mfcc)
                etiquetas_reales.append(etiqueta_real)

                os.remove(ruta_temp)

    return archivos, caracteristicas, etiquetas_reales

# Ejecutar procesamiento
archivos, caracteristicas, etiquetas_reales = procesar_audios()

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(caracteristicas, etiquetas_reales, test_size=0.2, random_state=42)

# Búsqueda de hiperparámetros con GridSearchCV para Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, n_jobs=-1, verbose=1)
grid_search_rf.fit(X_train, y_train)

# Mejor modelo Random Forest
best_rf = grid_search_rf.best_estimator_

# Evaluación del modelo
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Precisión del modelo Random Forest: {accuracy_rf * 100:.2f}%")

#  Matriz de confusión
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=etiquetas + ["otro"])
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=etiquetas + ["otro"])
disp_rf.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusión por Palabra (Random Forest)")
plt.tight_layout()
plt.show()

# Mostrar los mejores parámetros encontrados
print("Mejores parámetros encontrados (Random Forest):")
print(grid_search_rf.best_params_)

# Guardar el modelo entrenado
joblib.dump(best_rf, 'mejor_modelo_random_forest.pkl')
