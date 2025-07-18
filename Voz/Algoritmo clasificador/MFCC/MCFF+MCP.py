import os
import unicodedata
import wave
import joblib
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr
import librosa

from pydub import AudioSegment
from sklearn.neural_network import MLPClassifier
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

etiquetas = list(set(carpeta for carpeta in carpetas.values() if carpeta != "otro"))

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

def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto

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

#  Procesar audios
archivos, caracteristicas, etiquetas_reales = procesar_audios()

# División de datos
X_train, X_test, y_train, y_test = train_test_split(caracteristicas, etiquetas_reales, test_size=0.2, random_state=42)

# Búsqueda de hiperparámetros para MLP pequeña
param_grid_mlp = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [1e-4, 1e-5],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [300]
}

grid_search_mlp = GridSearchCV(MLPClassifier(random_state=42), param_grid_mlp, cv=3, verbose=1, n_jobs=-1)
grid_search_mlp.fit(X_train, y_train)

# Mejor modelo
best_mlp = grid_search_mlp.best_estimator_

# Evaluación
y_pred_mlp = best_mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"Precisión del modelo MLP: {accuracy_mlp * 100:.2f}%")

# Matriz de confusión
cm_mlp = confusion_matrix(y_test, y_pred_mlp, labels=etiquetas + ["otro"])
disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=etiquetas + ["otro"])
disp_mlp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusión por Palabra (MLP)")
plt.tight_layout()
plt.show()

# 🔧 Mostrar los mejores parámetros
print("Mejores parámetros encontrados (MLP):")
print(grid_search_mlp.best_params_)

# Guardar modelo
joblib.dump(best_mlp, 'mejor_modelo_mlp.pkl')
