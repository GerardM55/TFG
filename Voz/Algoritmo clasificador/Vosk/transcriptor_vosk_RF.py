import os
import unicodedata
import wave
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr  # Importar la librería para reducir el ruido

from pydub import AudioSegment
from sklearn.ensemble import RandomForestClassifier
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

# Etiquetas (excluyendo "otro" para el clasificador)
etiquetas = list(set(carpeta for carpeta in carpetas.values() if carpeta != "otro"))

# Cargar modelo Vosk en español
vosk_model_path = "C:/Users/gerar/Desktop/tfg/Control_de_veu/vosk-model-es-0.42/vosk-model-es-0.42"
vosk_model = Model(vosk_model_path)

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

# Transcripción con Vosk
def transcribir_audio(ruta_audio):
    wf = wave.open(ruta_audio, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(True)

    texto = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            texto += result.get("text", "") + " "
    final_result = json.loads(rec.FinalResult())
    texto += final_result.get("text", "")
    return texto.strip()

# Procesar audios
def procesar_audios():
    archivos, transcripciones, etiquetas_reales = [], [], []

    for carpeta, etiqueta_real in carpetas.items():
        for archivo in os.listdir(carpeta):
            if archivo.endswith('.wav'):
                ruta_audio = os.path.join(carpeta, archivo)

                # Normalizar y filtrar ruido
                audio_normalizado = normalizar_audio(ruta_audio)
                ruta_temp = f"temp_{archivo}"
                audio_normalizado.export(ruta_temp, format="wav")

                # Transcribir
                transcripcion = transcribir_audio(ruta_temp)
                transcripcion_limpia = limpiar_texto(transcripcion)

                archivos.append(archivo)
                transcripciones.append(transcripcion_limpia)
                etiquetas_reales.append(etiqueta_real)

                os.remove(ruta_temp)

    return archivos, transcripciones, etiquetas_reales

# Ejecutar el procesamiento
archivos, transcripciones, etiquetas_reales = procesar_audios()

# Vectorizar texto con TF-IDF (ajustes)
vectorizador = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizador.fit_transform(transcripciones)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, etiquetas_reales, test_size=0.2, random_state=42)

# Realizar búsqueda de hiperparámetros con GridSearchCV para Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']  # 👈 Importante para clases desbalanceadas
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_model = grid_search.best_estimator_

# Evaluar precisión con los mejores parámetros
y_pred_test = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Precisión del modelo con los mejores parámetros (Random Forest): {accuracy * 100:.2f}%")

# Mostrar predicciones
y_pred_full = best_model.predict(X)

# Matriz de confusión por palabra
cm = confusion_matrix(y_test, y_pred_test, labels=etiquetas + ["otro"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas + ["otro"])
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusión por Palabra (Random Forest)")
plt.tight_layout()
plt.show()

# Mostrar los mejores parámetros encontrados con GridSearchCV
print("Mejores parámetros encontrados (Random Forest):")
print(grid_search.best_params_)

#Guardar modelo y vectorizador
#joblib.dump(best_model, "modelo_transcriptor_rf_best.pkl")
#joblib.dump(vectorizador, "vectorizador_tfidf.pkl")
# print("Modelo y vectorizador guardados exitosamente.")
