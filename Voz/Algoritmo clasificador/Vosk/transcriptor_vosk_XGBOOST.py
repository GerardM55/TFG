import os
import unicodedata
import wave
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr

from pydub import AudioSegment
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
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

etiquetas = list(set(carpeta for carpeta in carpetas.values() if carpeta != "otro"))

# Cargar modelo Vosk
vosk_model_path = "C:/Users/gerar/Desktop/tfg/Control_de_veu/vosk-model-es-0.42/vosk-model-es-0.42"
vosk_model = Model(vosk_model_path)

# Normalizar audio y aplicar reducción de ruido
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

# Limpiar texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto

# Transcripción Vosk
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
                audio_normalizado = normalizar_audio(ruta_audio)
                ruta_temp = f"temp_{archivo}"
                audio_normalizado.export(ruta_temp, format="wav")
                transcripcion = transcribir_audio(ruta_temp)
                transcripcion_limpia = limpiar_texto(transcripcion)
                archivos.append(archivo)
                transcripciones.append(transcripcion_limpia)
                etiquetas_reales.append(etiqueta_real)
                os.remove(ruta_temp)
    return archivos, transcripciones, etiquetas_reales

# Ejecutar procesamiento de audios
archivos, transcripciones, etiquetas_reales = procesar_audios()

# Vectorizar texto
vectorizador = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizador.fit_transform(transcripciones)

# Codificar etiquetas a números
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(etiquetas_reales)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# GridSearchCV para XGBoost
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.8, 1.0],
}

grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Mejor modelo
best_model = grid_search.best_estimator_

# Evaluar modelo
y_pred_test = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Precisión del modelo con los mejores parámetros (XGBoost): {accuracy * 100:.2f}%")

# Matriz de confusión
y_pred_labels = label_encoder.inverse_transform(y_pred_test)
y_test_labels = label_encoder.inverse_transform(y_test)

cm = confusion_matrix(y_test_labels, y_pred_labels, labels=etiquetas + ["otro"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas + ["otro"])
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusión por Palabra (XGBoost)")
plt.tight_layout()
plt.show()

# Mostrar los mejores parámetros
print("Mejores parámetros encontrados (XGBoost):")
print(grid_search.best_params_)

# Guardar modelo, vectorizador y codificador si se desea
# joblib.dump(best_model, "modelo_transcriptor_xgboost_best.pkl")
# joblib.dump(vectorizador, "vectorizador_tfidf.pkl")
# joblib.dump(label_encoder, "codificador_etiquetas.pkl")
