import os
import unicodedata
import wave
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr

from pydub import AudioSegment
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from vosk import Model, KaldiRecognizer

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

etiquetas = list(set(carpeta for carpeta in carpetas.values() if carpeta != "otro")) # Excluir "otro" de las etiquetas

vosk_model_path = "C:/Users/gerar/Desktop/tfg/Control_de_veu/vosk-model-es-0.42/vosk-model-es-0.42"
vosk_model = Model(vosk_model_path)

# Funciones para normalizar audio
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

# Función para limpiar texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto

# Función para transcribir audio usando Vosk
def transcribir_audio(ruta_audio):
    wf = wave.open(ruta_audio, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(True)
    texto = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data): #
            result = json.loads(rec.Result())
            texto += result.get("text", "") + " " # Agregar espacio entre resultados
    final_result = json.loads(rec.FinalResult()) # Obtener el resultado final
    texto += final_result.get("text", "") 
    return texto.strip() # Eliminar espacios al inicio y al final

# Procesar audios 
def procesar_audios():
    archivos, transcripciones, etiquetas_reales = [], [], []
    for carpeta, etiqueta_real in carpetas.items():
        for archivo in os.listdir(carpeta):
            if archivo.endswith('.wav'):
                ruta_audio = os.path.join(carpeta, archivo)
                audio_normalizado = normalizar_audio(ruta_audio) # Normalizar el audio
                ruta_temp = f"temp_{archivo}"
                audio_normalizado.export(ruta_temp, format="wav") # Exportar el audio normalizado a un archivo temporal formato WAV
                transcripcion = transcribir_audio(ruta_temp) # Transcribir el audio normalizado
                transcripcion_limpia = limpiar_texto(transcripcion) # Limpiar la transcripción
                archivos.append(archivo)
                transcripciones.append(transcripcion_limpia)
                etiquetas_reales.append(etiqueta_real)
                os.remove(ruta_temp)
    return archivos, transcripciones, etiquetas_reales

archivos, transcripciones, etiquetas_reales = procesar_audios()

vectorizador = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)) # Vectorizador TF-IDF con n-gramas de 1 a 2
X = vectorizador.fit_transform(transcripciones) # Transformar las transcripciones en una matriz TF-IDF

X_train, X_test, y_train, y_test = train_test_split(X, etiquetas_reales, test_size=0.2, random_state=42)

# GridSearchCV para MLPClassifier
param_grid = {
    'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'max_iter': [300],
}

grid_search = GridSearchCV(MLPClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1) # Crear el objeto GridSearchCV
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred_test = best_model.predict(X_test) # Predecir las etiquetas del conjunto de prueba
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Precisión del modelo con los mejores parámetros (MLPClassifier): {accuracy * 100:.2f}%")

y_pred_full = best_model.predict(X)

# Calcular la matriz de confusión 
cm = confusion_matrix(y_test, y_pred_test, labels=etiquetas + ["otro"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas + ["otro"])
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusión por Palabra (MLPClassifier)")
plt.tight_layout()
plt.show()

print("Mejores parámetros encontrados (MLPClassifier):")
print(grid_search.best_params_)

#joblib.dump(best_model, "modelo_transcriptor_mlp_best.pkl")
#joblib.dump(vectorizador, "vectorizador_tfidf.pkl")
