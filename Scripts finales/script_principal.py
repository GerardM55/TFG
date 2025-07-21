#!/usr/bin/env python3
import subprocess
import lgpio
import time
import threading
import signal
import sys
from flask import Flask, request, jsonify
import joblib
from vosk import Model, KaldiRecognizer
import wave, json, unicodedata, io
from pydub import AudioSegment
import traceback
import grabar_audio
import PCA9685
import ADS1292

# --- Variables ---
EMG_MICRO_PIN = 14 # Pin para el micro EMG
chip = None # Variable para el chip de GPIO
datos_leidos = False # Indica si se han leído datos EMG
escucha_activada = False # Indica si la escucha está activada
mano_abierta = True # Indica si la mano está abierta
modo = " " # Modo actual, puede ser "voz" o "emg"
etiqueta = None # Etiqueta de la grabación
modo_change = False  # Asegúrate de declarar esto si lo usas en read_modo

# --- Funciones ---

def config_pin():
    # --- Función que configura las GPIOs ---
    global chip, EMG_MICRO_PIN
    chip = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_input(chip, EMG_MICRO_PIN)
    ADS1292.compartir_chip(chip) # Compartir chip con el script ADS1292
    PCA9685.compartir_chip(chip) # Compartir chip con el script PCA9685
    grabar_audio.compartir_chip(chip) # Compartir chip con el script grabar_audio

def read_modo():
    # --- Función que lee el modo actual de la GPIO EMG_MICRO_PIN ---
    global modo, chip, modo_change
    level = lgpio.gpio_read(chip, EMG_MICRO_PIN)
    if level == 1:
        if modo == "voz":
            modo_change = True
        modo = "emg"
    else:
        if modo == "emg":
            modo_change = True
        modo = "voz"

def main():
    # --- Función principal que ejecuta el script ---
    global etiqueta, datos_leidos, chip, modo_change, mano_abierta

    config_pin()
    try:
        read_modo()
        while True:
            if modo == "emg":
                ADS1292.activar_spi() # Activar el SPI para leer datos EMG en el script ADS1292
                ADS1292.lectura_emg() # Leer datos EMG en el script ADS1292
                if mano_abierta:
                    print("Agarre general")
                    PCA9685.ejecutar_agarre("general") # Ejecutar agarre general en el script PCA9685
                    mano_abierta = False
                else:
                    print("Agarre stop")
                    PCA9685.ejecutar_agarre("stop") # Ejecutar agarre stop en el script PCA9685
                    mano_abierta = True
            elif modo == "voz":
                etiqueta = grabar_audio.main() # Llamar a la función main del script grabar_audio para grabar el audio y obtener el agarre
                if not mano_abierta or etiqueta == "stop":
                    print("Agarre stop")
                    PCA9685.ejecutar_agarre("stop") # Ejecutar agarre stop en el script PCA9685
                    mano_abierta = True
                else:
                    print(f"Etiqueta recibida: {etiqueta}")
                    PCA9685.ejecutar_agarre(etiqueta) # Ejecutar agarre según la etiqueta en el script PCA9685
                    mano_abierta = False
                etiqueta = None

            print("Tiempo para cambiar de modo")
            time.sleep(3)
            read_modo()

    except Exception as e:
        print("Ocurrió un error:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
