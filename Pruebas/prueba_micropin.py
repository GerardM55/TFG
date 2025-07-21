import spidev
import lgpio
import time
import math
import csv
from collections import deque

# --- Constantes ---
Fmod = 128 * 10**3
Fclk = 2048 * 10**6
Tclk = 1 / Fmod
Vref = 4.036
Pga = 12
Resistencia_shunt = 0.5
Threshold_corriente = 0.9

# --- Comandos del ADS1292 ---
CMD_WAKEUP = 0x02
CMD_STANDBY = 0x04
CMD_RESET = 0x06
CMD_START = 0x08
CMD_STOP = 0x0A
CMD_RDATAC = 0x10
CMD_SDATAC = 0x11
CMD_RDATA = 0x12

# --- Pines ---
ADS1292_DRDY_PIN = 5
ADS1292_START_PIN = 27
ADS1292_PWDN_PIN = 17
ADS1292_CLKSEL_PIN = 4
EMG_VOZ_PIN = 14
MICRO_PIN = 26
MUX_A1_PIN = 23
MUX_A0_PIN = 22
MUX_EN_PIN = 24
SCK_PIN = 18
WS_PIN = 19
SD_PIN = 20

modo="emg"
# --- Variables globales ---
data_preparada = False
CONFIG_SPI_MASTER_DUMMY = 0x00
emgData = 0
servoData = 0
try: 
    lgpio.gpiochip_close(0) # Cierra el chip GPIO si está abierto
except lgpio.error:
    pass

chip = lgpio.gpiochip_open(0) # Abre el chip GPIO

# --- Registros ---
CONFIG1 = 0x01
CONFIG2 = 0x02
CH1SET = 0x04
CH2SET = 0x05
RLD_SENS = 0x06
RESP1 = 0x09
RESP2 = 0x0A

# --- Config GPIO ---
lgpio.gpio_claim_output(chip, ADS1292_START_PIN)
lgpio.gpio_claim_output(chip, ADS1292_CLKSEL_PIN)
lgpio.gpio_claim_output(chip, MUX_A0_PIN)
lgpio.gpio_claim_output(chip, MUX_A1_PIN)
lgpio.gpio_claim_output(chip, MUX_EN_PIN)
lgpio.gpio_claim_output(chip, SCK_PIN)
lgpio.gpio_claim_output(chip, WS_PIN)
lgpio.gpio_claim_output(chip, ADS1292_PWDN_PIN)

lgpio.gpio_claim_input(chip, EMG_VOZ_PIN)
lgpio.gpio_claim_input(chip, MICRO_PIN)
lgpio.gpio_claim_input(chip, SD_PIN)
lgpio.gpio_claim_input(chip, ADS1292_DRDY_PIN)

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
