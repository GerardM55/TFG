import spidev
import lgpio
import time
import math
import csv
from collections import deque

# --- Pines ---
EMG_VOZ_PIN = 14

# --- Variables globales ---
data_preparada = False
CONFIG_SPI_MASTER_DUMMY = 0x00
emgData = 0
servoData = 0
modo="emg"
try: 
    lgpio.gpiochip_close(0) # Cierra el chip GPIO si está abierto
except lgpio.error:
    pass

chip = lgpio.gpiochip_open(0) # Abre el chip GPIO

# --- Config GPIO ---
lgpio.gpio_claim_input(chip, EMG_VOZ_PIN)

def gpio_callback(chip, gpio, level, tick): # Callback para cambios en EMG_VOZ_PIN
    global modo, EMG_VOZ_PIN
    if modo == "voz":
        modo = "emg"
        print("emg")
    else:
        modo = "voz"
        print("voz")
def main():
    lgpio.gpio_claim_alert(chip, EMG_VOZ_PIN, lgpio.BOTH_EDGES) # Configura alerta para cambios en EMG_VOZ_PIN
    lgpio.callback(chip, EMG_VOZ_PIN, lgpio.BOTH_EDGES, gpio_callback) # Registra la función de callback
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
