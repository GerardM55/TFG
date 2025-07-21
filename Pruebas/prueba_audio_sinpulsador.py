#!/usr/bin/env python3
import alsaaudio
import time
import wave
import os
import subprocess
import lgpio
import threading

# Parámetros de grabación
sampling_rate = 48000 # Frecuencia de muestreo
num_channels = 2 # Estéreo
sample_width_bytes = 4  # 32 bits
MICRO_PIN = 26
ruta_audio = "/home/raspberry/Grabacion.wav"
chip = None

recording = False # Estado de grabación
recording_thread = None # Hilo de grabación
recorded_data = b''
i2s_device = None
datos_enviados = False # Indica si los datos han sido enviados

start_time = time.time()  # Inicia temporizador

# Inicializa el pin GPIO para el micrófono
def init_pin(): 
    global chip
    chip = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_input(chip, MICRO_PIN)

# Inicializa el dispositivo I2S para grabación
def init_i2s():
    device = alsaaudio.PCM( # Configuración del dispositivo I2S mediante ALSA
        type=alsaaudio.PCM_CAPTURE,
        mode=alsaaudio.PCM_NORMAL,
        device='hw:1,0'
    )
    # Configuración de parámetros del micrófono
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
        length, data = device.read()
        if length > 0:
            frames.append(data) # Acumula los datos grabados
        else:
            time.sleep(0.01)
    recorded_data = b''.join(frames)
    print("Grabación finalizada")

# Guarda los datos grabados en un archivo WAV
def save_wav(filename, audio_data):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width_bytes)
        wf.setframerate(sampling_rate)
        wf.writeframes(audio_data)

# Callback para manejar la GPIO 26
def gpio_callback(chip, gpio, level, tick):
    global recording, recording_thread, recorded_data, i2s_device, datos_enviados, start_time

    if level == 1 and not recording: # Si se detecta un flanco ascendente y no se está grabando se inicia la grabación
        recording = True
        i2s_device = init_i2s()
        recording_thread = threading.Thread(target=record_loop, args=(i2s_device,))
        recording_thread.start()

    elif level == 0 and recording:# Si se detecta un flanco descendente y se está grabando, se detiene la grabación
        recording = False
        if recording_thread:
            recording_thread.join()
        if i2s_device:
            i2s_device.close()
            i2s_device = None
        save_wav(ruta_audio, recorded_data)
        print(f"Archivo guardado en: {ruta_audio}")
        print("Ejecutando clasificador de voz...")

        result = subprocess.run( # Ejecuta el script de clasificación de voz a través de subprocess
            ["python3", "/home/raspberry/prueba_vosk+ics43434.py"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.splitlines(): # Procesa el print del script prueba_vosk+ics43434.py a través del stdout
            if line.startswith("Etiqueta:"):
                print(line.strip())
                datos_enviados = True
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"Tiempo total desde inicio hasta transcripción: {elapsed:.2f} segundos")

if __name__ == "__main__":
    init_pin() # Inicializa el pin GPIO
    lgpio.gpio_claim_alert(chip, MICRO_PIN, lgpio.BOTH_EDGES, lgpio.SET_PULL_DOWN) # Configura el pin para alertas de flanco ascendente y descendente
    lgpio.callback(chip, MICRO_PIN, lgpio.BOTH_EDGES, gpio_callback) # Registra el callback para manejar los eventos del pin GPIO
    print("Esperando señal en el pin...")
    try:
        while not datos_enviados:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Programa interrumpido por el usuario.")
        lgpio.gpiochip_close(chip)
