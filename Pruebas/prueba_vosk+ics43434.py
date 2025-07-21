#!/usr/bin/env python3

import wave
import os
import joblib
import unicodedata
from vosk import Model, KaldiRecognizer
import json
from pydub import AudioSegment
import io

# 📦 Cargar modelo de clasificación y vectorizador TF-IDF
modelo_path = "/home/raspberry/vosk_models/modelo_vosk.pkl"
vectorizador_path = "/home/raspberry/vosk_models/vectorizador_tfidf.pkl"
modelo = joblib.load(modelo_path)
vectorizador = joblib.load(vectorizador_path)

# 🧠 Cargar modelo de Vosk en español
vosk_model_path = "/home/raspberry/vosk_models/vosk-model-small-es-0.42"
vosk_model = Model(vosk_model_path)

# Función para convertir audio a 16 kHz, 16 bits, mono en memoria
def convertir_audio(ruta_entrada):
    audio = AudioSegment.from_file(ruta_entrada)
    audio = audio.set_frame_rate(16000)        # Re-muestrear a 16 kHz
    audio = audio.set_sample_width(2)          # 16 bits = 2 bytes
    audio = audio.set_channels(1)              # Mono (recomendado para Vosk)
    
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    return buffer

# Limpiar texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto

# Transcripción con Vosk usando el audio convertido
def transcribir_audio(ruta_audio):
    wav_buffer = convertir_audio(ruta_audio)
    wf = wave.open(wav_buffer, "rb")
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

# Clasificación
def clasificar_palabra(texto):
    texto_limpio = limpiar_texto(texto)
    X = vectorizador.transform([texto_limpio])
    prediccion = modelo.predict(X)
    return prediccion[0]

# Procesar audio
def procesar_audio(ruta_audio):
    print(f"Procesando {ruta_audio}...")

    transcripcion = transcribir_audio(ruta_audio)
    etiqueta = clasificar_palabra(transcripcion)
    return transcripcion, etiqueta

# Ruta donde se encuentra el audio
ruta_audio = "/home/raspberry/Grabacion.wav"

# Ejecutar procesamiento
transcripcion, etiqueta = procesar_audio(ruta_audio)

# Mostrar resultados
print(f"\nArchivo: Grabacion.wav")
print(f"Transcripción: {transcripcion}")
print(f"Etiqueta: {etiqueta}")
os.remove(ruta_audio)
print(f"Archivo {ruta_audio} borrado.")
