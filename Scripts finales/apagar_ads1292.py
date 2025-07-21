#!/usr/bin/env python3

import time
import os
import signal
import subprocess
from datetime import datetime
from pijuice import PiJuice

LOGFILE = "/home/raspberry/logs/apagar_ads1292.log" # Ruta del archivo de log

def log_message(msg):  
    # --- Función encargada de abrir el mensaje log---
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOGFILE, "a") as f:
        f.write(f"[{now}] {msg}\n")

def main():
    pijuice = PiJuice(1, 0x14)  # Bus I2C 1, dirección 0x14 (default)

    try:
        log_message("Iniciando apagado del sistema...")
        time.sleep(2)  # Dar tiempo a que se apaguen conexiones limpias
        pijuice.power.SetPowerOff(15)  # Espera 15 segundos antes de cortar energía

        log_message("SetPowerOff llamado correctamente. Esperando apagado del sistema.")
    except Exception as e:
        log_message(f"Error en apagar_ads1292.py: {e}")

if __name__ == "__main__":
    main()
