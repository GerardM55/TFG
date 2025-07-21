import os
import unicodedata
import wave
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr  # Librería para reducción de ruido

from pydub import AudioSegment
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC  # Clasificador SVM
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
    
    # Convertir a array numpy
    audio_data = np.array(audio.get_array_of_samples())

    # Aplicar reducción de ruido
    audio_reducido = nr.reduce_noise(y=audio_data, sr=audio.frame_rate)

    # Convertir de nuevo a AudioSegment
    audio_filtrado = AudioSegment(
        audio_reducido.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    
    # Normalizar volumen
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
        if rec.AcceptWaveform(data):  # Verifica si se ha reconocido un segmento
            result = json.loads(rec.Result())
            texto += result.get("text", "") + " "  # Agregar espacio entre resultados
    final_result = json.loads(rec.FinalResult())  # Obtener el resultado final
    texto += final_result.get("text", "")
    return texto.strip()  # Quitar espacios al inicio y al final

# Procesamiento de audios
def procesar_audios():
    archivos, transcripciones, etiquetas_reales = [], [], []
    for carpeta, etiqueta_real in carpetas.items():
        for archivo in os.listdir(carpeta):
            if archivo.endswith('.wav'):
                ruta_audio = os.path.join(carpeta, archivo)

                # Normalizar y filtrar ruido
                audio_normalizado = normalizar_audio(ruta_audio)
                ruta_temp = f"temp_{archivo}"
                audio_normalizado.export(ruta_temp, format="wav")  # Exportar como WAV temporal

                # Transcribir y limpiar
                transcripcion = transcribir_audio(ruta_temp)
                transcripcion_limpia = limpiar_texto(transcripcion)

                # Guardar datos
                archivos.append(archivo)
                transcripciones.append(transcripcion_limpia)
                etiquetas_reales.append(etiqueta_real)

                os.remove(ruta_temp)  # Eliminar archivo temporal

    return archivos, transcripciones, etiquetas_reales

# Ejecutar procesamiento
archivos, transcripciones, etiquetas_reales = procesar_audios()

# Vectorizar texto con TF-IDF
vectorizador = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # n-gramas de 1 a 2
X = vectorizador.fit_transform(transcripciones)

# Codificar etiquetas como números
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(etiquetas_reales)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Búsqueda de hiperparámetros con GridSearchCV para SVM
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Crear el objeto GridSearchCV
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_model = grid_search.best_estimator_

# Evaluar el modelo en el conjunto de prueba
y_pred_test = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Precisión del modelo con los mejores parámetros (SVM): {accuracy * 100:.2f}%")

# Convertir predicciones codificadas a etiquetas reales
y_pred_labels = label_encoder.inverse_transform(y_pred_test)
y_test_labels = label_encoder.inverse_transform(y_test)

# Matriz de confusión por palabra
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=etiquetas + ["otro"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas + ["otro"])
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusión por Palabra (SVM)")
plt.tight_layout()
plt.show()

# Mostrar los mejores parámetros encontrados con GridSearchCV
print("Mejores parámetros encontrados (SVM):")
print(grid_search.best_params_)

# Guardar modelo, vectorizador y codificador
# joblib.dump(best_model, "modelo_transcriptor_svm_best.pkl")
# joblib.dump(vectorizador, "vectorizador_tfidf.pkl")
# joblib.dump(label_encoder, "codificador_etiquetas.pkl")
# print("Modelo, vectorizador y codificador guardados exitosamente.")
