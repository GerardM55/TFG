#!/usr/bin/env python3
import alsaaudio
import time
import wave
import os
import threading
import lgpio
import joblib
import unicodedata
from vosk import Model, KaldiRecognizer
import json
from pydub import AudioSegment
import io

# Parámetros de grabación
sampling_rate = 48000
num_channels = 2
sample_width_bytes = 4  # 32 bits
MICRO_PIN = 26
ruta_audio_base = "/home/raspberry/Grabacion"
chip = None

recording = False
recording_thread = None
recorded_data = b''
i2s_device = None

# Evento para sincronización threads
datos_enviados_event = threading.Event()

modelo = None
vectorizador = None
vosk_model = None
modelos_cargados = False
modelos_cargando = False

# Lista para almacenar rutas de audios pendientes de procesar
cola_audios_pendientes = []
cola_lock = threading.Lock()

modelo_path = "/home/raspberry/vosk_models/modelo_vosk.pkl"
vectorizador_path = "/home/raspberry/vosk_models/vectorizador_tfidf.pkl"
vosk_model_path = "/home/raspberry/vosk_models/vosk-model-small-es-0.42"

def init_pin():
    global chip
    chip = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_input(chip, MICRO_PIN)

def init_i2s():
    device = alsaaudio.PCM(
        type=alsaaudio.PCM_CAPTURE,
        mode=alsaaudio.PCM_NORMAL,
        device='hw:1,0'
    )
    device.setchannels(num_channels)
    device.setrate(sampling_rate)
    device.setformat(alsaaudio.PCM_FORMAT_S32_LE)
    device.setperiodsize(1024)
    return device

def record_loop(device):
    global recording, recorded_data
    frames = []
    print("Grabando...")
    while recording:
        length, data = device.read()
        if length > 0:
            frames.append(data)
        else:
            time.sleep(0.01)
    recorded_data = b''.join(frames)
    print("Grabación finalizada")

def save_wav(filename, audio_data):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width_bytes)
        wf.setframerate(sampling_rate)
        wf.writeframes(audio_data)

def cargar_modelos():
    global modelo, vectorizador, vosk_model, modelos_cargados, modelos_cargando
    modelos_cargando = True
    print("Cargando modelos...")
    modelo = joblib.load(modelo_path)
    vectorizador = joblib.load(vectorizador_path)
    vosk_model = Model(vosk_model_path)
    modelos_cargados = True
    modelos_cargando = False
    print("Modelos cargados.")
    threading.Thread(target=procesar_cola, daemon=True).start()

def convertir_audio(ruta_entrada):
    audio = AudioSegment.from_file(ruta_entrada)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)
    audio = audio.set_channels(1)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    return buffer

def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto

def transcribir_audio(ruta_audio):
    global vosk_model
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

def clasificar_palabra(texto):
    global modelo, vectorizador
    texto_limpio = limpiar_texto(texto)
    X = vectorizador.transform([texto_limpio])
    prediccion = modelo.predict(X)
    return prediccion[0]

def procesar_audio(ruta_audio):
    print(f"Procesando audio: {ruta_audio}")
    transcripcion = transcribir_audio(ruta_audio)
    etiqueta = clasificar_palabra(transcripcion)
    print(f"Etiqueta: {etiqueta}")
    datos_enviados_event.set()

def procesar_cola():
    while True:
        cola_lock.acquire()
        if len(cola_audios_pendientes) == 0:
            cola_lock.release()
            time.sleep(1)
            continue
        audio_a_procesar = cola_audios_pendientes.pop(0)
        cola_lock.release()
        procesar_audio(audio_a_procesar)

def gpio_callback(chip, gpio, level, tick):
    global recording, recording_thread, recorded_data, i2s_device

    if level == 1 and not recording:
        recording = True
        i2s_device = init_i2s()
        recording_thread = threading.Thread(target=record_loop, args=(i2s_device,))
        recording_thread.start()

    elif level == 0 and recording:
        recording = False
        if recording_thread:
            recording_thread.join()
        if i2s_device:
            i2s_device.close()
            i2s_device = None

        timestamp = int(time.time() * 1000)
        ruta_audio_actual = f"{ruta_audio_base}_{timestamp}.wav"
        save_wav(ruta_audio_actual, recorded_data)
        print(f"Archivo guardado en: {ruta_audio_actual}")

        cola_lock.acquire()
        cola_audios_pendientes.append(ruta_audio_actual)
        cola_lock.release()

if __name__ == "__main__":
    init_pin()
    lgpio.gpio_claim_alert(chip, MICRO_PIN, lgpio.BOTH_EDGES, lgpio.SET_PULL_DOWN)
    lgpio.callback(chip, MICRO_PIN, lgpio.BOTH_EDGES, gpio_callback)

    # Iniciar temporizador
    start_time = time.time()

    # Cargar modelos en hilo
    threading.Thread(target=cargar_modelos, daemon=True).start()

    print("Esperando señal en el pin...")

    try:
        datos_enviados_event.wait()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Audio procesado, finalizando programa.")
        print(f"Tiempo transcurrido desde el inicio: {elapsed_time:.2f} segundos")
    except KeyboardInterrupt:
        print("Programa interrumpido por el usuario.")
    finally:
        lgpio.gpiochip_close(chip)
