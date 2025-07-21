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
sampling_rate = 48000 # Frecuencia de muestreo
num_channels = 2 # Número de canales
sample_width_bytes = 4  # 32 bits
MICRO_PIN = 26 # Pin GPIO para el micrófono
ruta_audio_base = "/home/raspberry/Grabacion" # Ruta base para guardar audios
chip = None

recording = False # Estado de grabación
recording_thread = None # Hilo de grabación
recorded_data = b'' # Datos grabados
i2s_device = None # Dispositivo I2S

# Evento para sincronización threads
datos_enviados_event = threading.Event()

modelo = None # Modelo de clasificación
vectorizador = None # Vectorizador TF-IDF
vosk_model = None # Modelo de Vosk
modelos_cargados = False # Variable que indica si se han cargado los modelos
modelos_cargando = False # Variable que indica si se están cargando los modelos

# Lista para almacenar rutas de audios pendientes de procesar
cola_audios_pendientes = []
cola_lock = threading.Lock()

modelo_path = "/home/raspberry/vosk_models/modelo_vosk.pkl" # Ruta del modelo de clasificación
vectorizador_path = "/home/raspberry/vosk_models/vectorizador_tfidf.pkl" # Ruta del vectorizador TF-IDF
vosk_model_path = "/home/raspberry/vosk_models/vosk-model-small-es-0.42" # Ruta del modelo de Vosk en español

# Función para inicializar el pin GPIO
def init_pin(): 
    global chip
    chip = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_input(chip, MICRO_PIN)

# Función para inicializar el dispositivo I2S
def init_i2s():
    device = alsaaudio.PCM( # Configuración del dispositivo I2S con ALSA
        type=alsaaudio.PCM_CAPTURE,
        mode=alsaaudio.PCM_NORMAL,
        device='hw:1,0'
    )
    # Configuración de canales, frecuencia de muestreo y formato
    device.setchannels(num_channels) 
    device.setrate(sampling_rate)
    device.setformat(alsaaudio.PCM_FORMAT_S32_LE)
    device.setperiodsize(1024)
    return device

# Función para grabar audio en un bucle
def record_loop(device):
    global recording, recorded_data
    frames = []
    print("Grabando...")
    while recording: # Mientras se esté grabando
        length, data = device.read() # Leer datos del dispositivo I2S
        if length > 0:
            frames.append(data) # Almacenar los datos grabados
        else:
            time.sleep(0.01)
    recorded_data = b''.join(frames)
    print("Grabación finalizada")

# Función para guardar los datos grabados en un archivo WAV
def save_wav(filename, audio_data):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width_bytes)
        wf.setframerate(sampling_rate)
        wf.writeframes(audio_data)

# Función para cargar el modelo de clasificación y Vosk
def cargar_modelos():
    global modelo, vectorizador, vosk_model, modelos_cargados, modelos_cargando
    modelos_cargando = True
    print("Cargando modelos...")
    modelo = joblib.load(modelo_path) # Cargar modelo de clasificación
    vectorizador = joblib.load(vectorizador_path) # Cargar vectorizador TF-IDF
    vosk_model = Model(vosk_model_path) # Cargar modelo de Vosk
    modelos_cargados = True
    modelos_cargando = False
    print("Modelos cargados.")
    threading.Thread(target=procesar_cola, daemon=True).start() # Iniciar hilo para procesar la cola de audios

# Función para convertir audio a formato WAV en memoria
def convertir_audio(ruta_entrada):
    # Convertir el archivo de audio a formato WAV
    audio = AudioSegment.from_file(ruta_entrada)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)
    audio = audio.set_channels(1)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    return buffer

# Limpiar texto para normalizarlo
def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto) # Normalizar caracteres Unicode
        if unicodedata.category(c) != 'Mn'
    )
    return texto

# Transcribir audio usando Vosk
def transcribir_audio(ruta_audio):
    global vosk_model
    wav_buffer = convertir_audio(ruta_audio)
    wf = wave.open(wav_buffer, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate()) # Inicializar el reconocedor de Vosk
    rec.SetWords(True)

    texto = ""
    while True: # Leer datos del archivo WAV
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            texto += result.get("text", "") + " "
    final_result = json.loads(rec.FinalResult()) # Obtener el resultado final
    texto += final_result.get("text", "")
    return texto.strip()

# Clasificar la palabra transcrita usando el modelo de clasificación
def clasificar_palabra(texto):
    global modelo, vectorizador
    texto_limpio = limpiar_texto(texto)
    X = vectorizador.transform([texto_limpio])
    prediccion = modelo.predict(X)
    return prediccion[0]

# Procesar el audio grabado
def procesar_audio(ruta_audio):
    print(f"Procesando audio: {ruta_audio}")
    transcripcion = transcribir_audio(ruta_audio)
    etiqueta = clasificar_palabra(transcripcion)
    print(f"Etiqueta: {etiqueta}")
    datos_enviados_event.set()

# Iniciar un hilo para procesar la cola de audios pendientes
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

# Callback para manejar eventos del GPIO
def gpio_callback(chip, gpio, level, tick):
    global recording, recording_thread, recorded_data, i2s_device

    if level == 1 and not recording: # Si se detecta un flanco ascendente y no se está grabando se inicia la grabación
        recording = True
        i2s_device = init_i2s()
        recording_thread = threading.Thread(target=record_loop, args=(i2s_device,))
        recording_thread.start()

    elif level == 0 and recording: # Si se detecta un flanco descendente y se está grabando se finaliza la grabación
        recording = False
        if recording_thread:
            recording_thread.join()
        if i2s_device:
            i2s_device.close()
            i2s_device = None

        timestamp = int(time.time() * 1000)
        ruta_audio_actual = f"{ruta_audio_base}_{timestamp}.wav"
        save_wav(ruta_audio_actual, recorded_data) # Guardar el audio grabado en un archivo WAV
        print(f"Archivo guardado en: {ruta_audio_actual}")

        cola_lock.acquire() # Añadir la ruta del audio grabado a la cola de pendientes
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
        datos_enviados_event.wait() # Esperar a que se envíen los datos
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Audio procesado, finalizando programa.")
        print(f"Tiempo transcurrido desde el inicio: {elapsed_time:.2f} segundos")
    except KeyboardInterrupt:
        print("Programa interrumpido por el usuario.")
    finally:
        lgpio.gpiochip_close(chip)
