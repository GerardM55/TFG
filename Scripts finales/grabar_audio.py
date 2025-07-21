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
from pydub import AudioSegment
import io
import json

# --- VARIABLES CONFIGURACIÓN I2S---
sampling_rate = 48000 # Frecuencia de muestreo
num_channels = 2 # Número de canales (2 para estéreo)
sample_width_bytes = 4  # 32 bits
MICRO_PIN = 26 # Pin GPIO para el micrófono

# --- VARIABLES ---
ruta_audio_base = "/home/raspberry/Grabacion" # Ruta base para guardar los audios grabados
#chip = None
etiqueta = " " # Etiqueta para almacenar la clasificación del audio
recording = False # Variable para indicar si se está grabando
recording_thread = None # Hilo para grabar audio
recorded_data = b'' # Datos grabados del audio
i2s_device = None # Dispositivo I2S
audio_grabado = False # Variable para indicar si ya se grabó un audio
datos_enviados_event = threading.Event() # Evento para indicar que ya se procesó un audio
modelo = None # Modelo de clasificación
vectorizador = None # Vectorizador TF-IDF
vosk_model = None # Modelo VOSK
modelos_cargados = False # Variable para indicar si los modelos están cargados
modelos_cargando = False # Variable para indicar si los modelos se están cargando
cola_audios_pendientes = [] # Lista para almacenar rutas de audios pendientes de procesar
cola_lock = threading.Lock() # Lock para manejar la cola de audios pendientes

# --- ALGORITMO CLASIFICADOR Y TRANSCRIPTOR VOSK ---
modelo_path = "/home/raspberry/vosk_models/modelo_vosk.pkl" 
vectorizador_path = "/home/raspberry/vosk_models/vectorizador_tfidf.pkl"
vosk_model_path = "/home/raspberry/vosk_models/vosk-model-small-es-0.42"


# --- FUNCIONES ---

def init_pin():
    # --- Función que configura el pin de grabación del audio  ---
    global chip
    lgpio.gpio_claim_input(chip, MICRO_PIN)

def compartir_chip(valor_chip):
    # --- Función que configura la variable chip con el valor Chip  ---
    global chip
    chip=valor_chip

def init_i2s():
    # --- Configura el I2S ---
    device = alsaaudio.PCM( # Dispositivo I2S utilizando ALSA
        type=alsaaudio.PCM_CAPTURE,
        mode=alsaaudio.PCM_NORMAL,
        device='hw:1,0'
    )
    # Configura el numero de canales, frecuencia de muestreo y formato
    device.setchannels(num_channels)
    device.setrate(sampling_rate)
    device.setformat(alsaaudio.PCM_FORMAT_S32_LE)
    device.setperiodsize(1024)
    return device

def loop_grabacion(device):
    # --- Función que graba el audio ---
    global recording, recorded_data
    frames = []
    print("Grabando...")
    while recording: # Mientras se esté grabando
        length, data = device.read()
        if length > 0:
            frames.append(data) # Añade los datos grabados a los frames
        else:
            time.sleep(0.01)
    recorded_data = b''.join(frames) # Une todos los frames grabados
    print("️ Grabación finalizada")

def guardar_wav(filename, audio_data):
    # --- Función que guarda el audio ---
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width_bytes)
        wf.setframerate(sampling_rate)
        wf.writeframes(audio_data)

def cargar_modelos():
    # --- Función para cargar los modelos importados ---
    global modelo, vectorizador, vosk_model, modelos_cargados, modelos_cargando
    modelos_cargando = True
    print("Cargando modelos...")
    modelo = joblib.load(modelo_path) # Carga el modelo de clasificación
    vectorizador = joblib.load(vectorizador_path) # Carga el vectorizador TF-IDF
    vosk_model = Model(vosk_model_path) # Carga el modelo VOSK
    modelos_cargados = True
    modelos_cargando = False
    print("Modelos cargados.")
    threading.Thread(target=procesar_cola, daemon=True).start() # Cuando cargan los modelos, empezar a procesar cola

def convertir_audio(ruta_entrada):
    # --- Función que convierte el archivo WAV a mono de 16 KHz y 16 bits ---
    audio = AudioSegment.from_file(ruta_entrada)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)
    audio = audio.set_channels(1)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    return buffer

def limpiar_texto(texto):
     # --- Función que limpia el texto convirtiendolo en minúsculas y eliminando acentos y caracteres especiales ---
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto) # Normaliza el texto
        if unicodedata.category(c) != 'Mn' # Elimina acentos
    )
    return texto

def transcribir_audio(ruta_audio):
     # --- Función que transcribe el audio con el transcriptor VOSK---
    global vosk_model
    wav_buffer = convertir_audio(ruta_audio)
    wf = wave.open(wav_buffer, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate()) # Crea el reconocedor VOSK
    rec.SetWords(True) # Configura el reconocedor para devolver palabras

    texto = ""
    while True:
        data = wf.readframes(4000) # Lee los datos del audio en bloques de 4000 frames
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data): # Si el reconocedor acepta el bloque de audio
            result = json.loads(rec.Result()) # Obtiene el resultado
            texto += result.get("text", "") + " "
    final_result = json.loads(rec.FinalResult()) # Obtiene el resultado final
    texto += final_result.get("text", "") # Añade el texto final
    return texto.strip()

def clasificar_palabra(texto):
     # --- Función que vectoriza la transcripción y predice el agarre---
    global modelo, vectorizador
    texto_limpio = limpiar_texto(texto) # Limpia el texto
    X = vectorizador.transform([texto_limpio]) # Vectoriza el texto
    prediccion = modelo.predict(X) # Predice la clase del texto
    return prediccion[0]

def procesar_audio(ruta_audio):
    # --- Función principal encargada de la transcripción y clasificación del audio---
    global etiqueta
    print(f"Procesando audio: {ruta_audio}")
    transcripcion = transcribir_audio(ruta_audio) # Transcribe el audio
    etiqueta = clasificar_palabra(transcripcion) # Clasifica la transcripción
    print(f"Etiqueta: {etiqueta}")
    datos_enviados_event.set()  # Señalamos que ya procesamos un audio
    # No borramos el archivo para conservarlo
    try:
          os.remove(ruta_audio)
    except Exception as e:
           print(f"Error borrando archivo {ruta_audio}: {e}")

def procesar_cola():
    # --- Función encargada de guardar el audio en una cola y enviarlo a procesarlo---
    while True:
        cola_lock.acquire()
        if len(cola_audios_pendientes) == 0:
            cola_lock.release()
            time.sleep(1)  # Espera para no consumir CPU si está vacía
            continue
        audio_a_procesar = cola_audios_pendientes.pop(0) # Toma el primer audio de la cola
        cola_lock.release() # Libera el lock de la cola
        procesar_audio(audio_a_procesar) # Procesa el audio

def gpio_callback(chip, gpio, level, tick):
    # --- Función activada cuando la variable MICRO_PIN cambia de valor. Cuando esta vale 1 se graba el audio y se procesa ---
    global recording, recording_thread, recorded_data, i2s_device, audio_grabado
    # Evita más grabaciones si ya se grabó un audio
    if audio_grabado:
        return

    if level == 1 and not recording: # Si el pin cambia a alto y no se está grabando se inicia la grabación
        # Empieza la grabación
        recording = True
        i2s_device = init_i2s()
        recording_thread = threading.Thread(target=loop_grabacion, args=(i2s_device,))
        recording_thread.start()

    elif level == 0 and recording: # Si el pin cambia a bajo y se está grabando se detiene la grabación
        # Termina la grabación
        recording = False
        if recording_thread:
            recording_thread.join()
        if i2s_device:
            i2s_device.close()
            i2s_device = None

        # Guarda el archivo con nombre único
        timestamp = int(time.time() * 1000)
        ruta_audio_actual = f"{ruta_audio_base}_{timestamp}.wav"
        guardar_wav(ruta_audio_actual, recorded_data)
        print(f"Archivo guardado en: {ruta_audio_actual}")

        # Añade el audio a la cola para procesar
        cola_lock.acquire()
        cola_audios_pendientes.append(ruta_audio_actual)
        cola_lock.release()

        # El audio ya se ha grabado
        audio_grabado = True
    
def main():
    global i2s_device, audio_grabado

    # Reiniciar estado antes de nueva grabación
    audio_grabado = False
    datos_enviados_event.clear() # Limpiar evento de datos enviados
    init_pin()
    lgpio.gpio_claim_alert(chip, MICRO_PIN, lgpio.BOTH_EDGES, lgpio.SET_PULL_DOWN) # Configurar el pin para alertas
    lgpio.callback(chip, MICRO_PIN, lgpio.BOTH_EDGES, gpio_callback) # Registrar la función de callback para el pin

    # Cargar modelos solo si no están cargados aún
    if not modelos_cargados and not modelos_cargando:
        threading.Thread(target=cargar_modelos, daemon=True).start() # Cargar modelos en un hilo separado

    print("Esperando señal en el pin...")
    datos_enviados_event.wait()  # Esperar hasta que haya audio procesado
    print("Audio procesado, finalizando programa.")

    # Liberar GPIO
    lgpio.gpio_free(chip, MICRO_PIN)

    # Cerrar comunicación i2s si sigue abierta
    if i2s_device:
        i2s_device.close()
        i2s_device = None

    return etiqueta

if __name__ == "__main__":
    main()