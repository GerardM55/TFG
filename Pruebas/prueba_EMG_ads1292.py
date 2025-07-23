import spidev
import lgpio
import time
import math
from collections import deque
import numpy as np
import threading
import struct
import os
import select
import scipy.signal
import script_principal

# --- CONSTANTES GLOBAL ---
dataSize = 9   # 3 status + 2 channels * 3 bytes
spiDummyByte = 0x00  # Byte de relleno para SPI
config1Value= 0x02 #Registro de configuración 1 del ADS1292
fc = 245     # Frecuencia de corte (Hz)
fs = 500     # Frecuencia de muestreo (Hz)
order = 4    # Orden del filtro

# --- CONSTANTES EMG ---
vref = 2.42 # Voltaje de referencia del ADS1292
emgPga = 12.0 # Ganancia del amplificador de instrumentación en el modo EMG
sampleSize = 100 # Tamaño del buffer de picos
peakWindow = 5 # Tamaño de la ventana de picos
peakSampleSize = 200  # Tamaño del buffer de picos iniciales
config2EMG=0xA0 # Valor del registro de configuración 2 del ADS1292 en modo EMG      
ch1setEMG=0x60 # Valor del registro CH1SET del ADS1292 en modo EMG       
ch2setEMG=0x81 # Valor del registro CH2SET del ADS1292 en modo EMG
rldsenseEMG=0xE3 # Valor del registro RLD_SENS del ADS1292 en modo EMG      
resp1EMG=0x02 # Valor del registro RESP1 del ADS1292 en modo EMG
resp2EMG=0x07 # Valor del registro RESP2 del ADS1292 en modo EMG

# --- BUFFERS ---
emgBuffer = np.zeros(peakWindow, dtype=np.int32) # Buffer para almacenar los últimos 3 valores EMG
peakValues = np.zeros(sampleSize, dtype=np.int32) # Buffer para almacenar los picos detectados
initialPeaks = np.zeros(peakSampleSize, dtype=np.int32) # Buffer para almacenar los picos iniciales

# --- VARIABLES EMG---
dataReady = False # Variable que indica si hay datos listos para leer
calibrationStarted = False # Variable que indica si se ha iniciado la calibración
calibrationDone = False # Variable que indica si la calibración ha finalizado
detected_peak = False # Para parar la lectura
peakCount = 0 # Contador de picos detectados
peakMedian = np.int32(0) # Mediana de los picos detectados
peakStdDev = np.int32(0) # Desviación estándar de los picos detectados
threshold = np.int32(0) # Umbral para la detección de picos
lastPeakValue = np.int32(0) # Último valor de pico detectado
bufferFillCount = 0 # Contador de llenado del buffer EMG
lowPeakCount = 0 # Contador de picos bajos detectados
emgData = np.int32(0) # Variable para almacenar el valor EMG actual
initialPeakCount = 0 # Contador de picos iniciales detectados
peakRangeComputed = False # Variable que indica si el rango de picos ha sido calculado
peakMinRange = np.int32(0) # Rango mínimo de picos detectados
peakMaxRange = np.int32(0) # Rango máximo de picos detectados

# --- CONSTANTES SERVOMOTOR ---
shuntResistance = ((0.5*0.17)/(0.5+0.17)) # Valor de la resistencia de shunt que es el valor equivalente al paralelo de la resistencia de 0.5 ohmios y 0.17 ohmios
currentThreshold = 1 # Umbral de corriente para la detección de contracción del servomotor
servoPga = 1.0 # Ganancia del amplificador de instrumentación en el modo servomotor
config2Servo=0xB0 # Valor del registro de configuración 2 del ADS1292 para calcular la realimentación del servomotor
ch1setServo=0x81 # Valor del registro CH1SET del ADS1292 para calcular la realimentación del servomotor
ch2setServo=0x10 # Valor del registro CH2SET del ADS1292 para calcular la realimentación del servomotor
rldsenseServo=0xC0 # Valor del registro RLD_SENS del ADS1292 para calcular la realimentación del servomotor
resp1Servo=0x02 # Valor del registro RESP1 del ADS1292 para calcular la realimentación del servomotor
resp2Servo=0x07 # Valor del registro RESP2 del ADS1292 para calcular la realimentación del servomotor

# --- VARIABLES SERVOMOTOR  ---
servoData = np.int32(0) # Variable para almacenar el valor del servomotor actual
# --- COMANDOS ADS1292 ---
CMD_WAKEUP = 0x02 
CMD_STANDBY = 0x04
CMD_RESET = 0x06
CMD_START = 0x08
CMD_STOP = 0x0A
CMD_RDATAC = 0x10
CMD_SDATAC = 0x11
CMD_RDATA = 0x12

# --- PINES ---
ADS1292_DRDY_PIN = 5
ADS1292_START_PIN = 27
ADS1292_PWDN_PIN = 17

# --- VARIABLES GLOBALES ---
data_preparada = False # Variable que indica si los datos están preparados para ser leídos
CONFIG_SPI_MASTER_DUMMY = 0x00 # Byte de relleno para la comunicación SPI
chip=None # Variable para almacenar el chip GPIO
mano_abierta=True # Variable que indica si la mano está abierta
modo="voz" # Variable que indica el modo de operación
modo_change=False # Variable que indica si el modo ha cambiado
spi=None # Variable para almacenar la instancia de comunicación SPI

# --- Registros ---
CONFIG1 = 0x01
CONFIG2 = 0x02
CH1SET = 0x04
CH2SET = 0x05
RLD_SENS = 0x06
RESP1 = 0x09
RESP2 = 0x0A

# --- FUNCIONES ADS1292 ---

def send_command(command):
    # --- Función que envía un comando al ADS1292 ---
    global spi
    time.sleep(2e-6)
    spi.xfer2([command])
    time.sleep(0.001)

def read_register(reg):
    # --- Función que lee un registro del ADS1292 ---
    global spi
    spi.xfer2([0x20 | reg, 0x00])
    time.sleep(1e-3)
    data = spi.xfer2([0x00])
    return data[0]

def write_register(reg, val):
    # --- Función que escribe un valor en un registro del ADS1292 ---
    global spi
    spi.xfer2([0x40 | reg, 0x00, val])
    time.sleep(0.01)

def enable_continuous_reading():
    # --- Función que activa la lectura continua de datos del ADS1292 ---
    send_command(CMD_RDATAC)

def disable_continuous_reading():
    # --- Función que desactiva la lectura continua de datos del ADS1292 ---
    send_command(CMD_SDATAC)

def start_conversion():
    # --- Función que inicia la conversión de datos del ADS1292 ---
    send_command(CMD_START)

def stop_conversion():
    # --- Función que detiene la conversión de datos del ADS1292 ---
    send_command(CMD_STOP)

def read_data():
    # --- Función que lee los datos del ADS1292 ---
    data = spi.xfer2([CMD_RDATA] + [0x00] * 9)
    return data[1:]

def reset_ads1292():
    # --- Función que resetea el ADS1292 ---
    global chip
    lgpio.gpio_write(chip, ADS1292_PWDN_PIN, 0)
    time.sleep(0.002)
    lgpio.gpio_write(chip, ADS1292_PWDN_PIN, 1)
    time.sleep(0.1)
    send_command(CMD_RESET)
    time.sleep(0.01)
    send_command(CMD_SDATAC)
    time.sleep(0.01)
    lgpio.gpio_write(chip, ADS1292_START_PIN, 1)

def config_ini():
    # --- Config GPIO ---
    global chip
    #chip = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(chip, ADS1292_START_PIN)
    lgpio.gpio_claim_output(chip, ADS1292_PWDN_PIN)
    lgpio.gpio_claim_input(chip, ADS1292_DRDY_PIN)

def configuracion_ini():
    # --- Config GPIO ---
    global chip
    chip = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(chip, ADS1292_START_PIN)
    lgpio.gpio_claim_output(chip, ADS1292_PWDN_PIN)
    lgpio.gpio_claim_input(chip, ADS1292_DRDY_PIN)
    #prueba_retroalimentacion.compartir_chip(chip)
    return chip
 
# --- FUNCIONES GLOBALES ---

def spi_ini():
    # --- Función que configura la comunicación SPI e inicializa el ADS1292 ---
    global spi, chip
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 500000
    spi.mode = 0b01
    
    # inicialización del ADS1292
    lgpio.gpio_write(chip, ADS1292_PWDN_PIN, 1)
    time.sleep(0.1)
    lgpio.gpio_write(chip, ADS1292_PWDN_PIN, 0)
    time.sleep(0.1)
    lgpio.gpio_write(chip, ADS1292_PWDN_PIN, 1)
    time.sleep(0.1)
    lgpio.gpio_write(chip, ADS1292_START_PIN, 0)
    time.sleep(0.02)

def comprobar_spi():
    # --- Función que comprueba la correcta comunicación SPI entre el ADS1292 y la Z2W ---
    for i in range(3):
        id_val = read_register(0x00)
        print(f"Intent {i+1}: Registre ID = 0x{id_val:02X}")
        if id_val == 0x53:
            print("✅ ID vàlid detectat.")
            return True
        reset_ads1292()
        time.sleep(0.1)
    print("⚠️ Error: No ID vàlid després de 3 intents.")
    return False

def leer_valores_ADS1292():
    # --- Función de lectura de datos del ADS1292 ---
    global emgData, servoData
    spiBuffer = spi.xfer2([CONFIG_SPI_MASTER_DUMMY] * 9)

    # Lectura del canal EMG
    # Leer como uint32 sin signo
    rawEmg = (np.uint32(spiBuffer[3]) << 16) | (np.uint32(spiBuffer[4]) << 8) | np.uint32(spiBuffer[5])
    if rawEmg & 0x800000:  # Si el bit de signo está activado
        rawEmg |= 0xFF000000  # Extiende el signo a 32 bits
    emgData = np.int32(rawEmg)

    # Lectura del canal de servomotor
    rawServo = (np.uint32(spiBuffer[6]) << 16) | (np.uint32(spiBuffer[7]) << 8) | np.uint32(spiBuffer[8])
    if rawServo & 0x800000:
        rawServo |= 0xFF000000
    servoData = np.int32(rawServo)

def comprobar_registros(state):
    # --- Función de comprueba que la correcta configuración de los registros del ADS1292 ---
    cf2 = read_register(CONFIG2) # Lectura del registro CONFIG2
    ch1 = read_register(CH1SET) # Lectura del registro CH1SET
    ch2 = read_register(CH2SET) # Lectura del registro CH2SET
    rld = read_register(RLD_SENS) # Lectura del registro RLD_SENS
    resp = read_register(RESP2) # Lectura del registro RESP2

    print(f"CH1SET: 0x{ch1:02X}")
    print(f"CH2SET: 0x{ch2:02X}")
    print(f"RLD_SENS: 0x{rld:02X}")
    print(f"RESP2: 0x{resp:02X}")
    print("--------------------")
    if state == "emg": # Si los valores adquiridos del modo EMG no son iguales a los valores esperados, devuelve False
        if cf2 != config2EMG or ch1 != ch1setEMG or ch2 != ch2setEMG or rld != rldsenseEMG or resp != resp2EMG:
            return False
        
    else: # Si los valores adquiridos del modo servomotor no son iguales a los valores esperados, devuelve False
        if cf2 != config2Servo or ch1 != ch1setServo or ch2 != ch2setServo or rld != rldsenseServo or resp != resp2Servo:
            return False
    
    return True

def activar_spi():
    # --- Función para activar la comunicación serie SPI ---
    global chip, spi
    spi_ini()

def desactivar_spi():
    # --- Función para desactivar la comunicación serie SPI ---
    global spi
    spi.close()

# --- FUNCIONES EMG ---   
#  
def emg_configuracion():
    # --- Función que configura los registros del ADS1292 para la lectura del EMG ---
    disable_continuous_reading()
    write_register(CONFIG1, config1Value)
    time.sleep(0.1)
    write_register(CONFIG2, config2EMG)
    time.sleep(0.1)
    write_register(0x03,0x10)
    time.sleep(0.1)
    write_register(CH1SET, ch1setEMG)
    time.sleep(0.1)
    write_register(CH2SET, ch2setEMG)
    time.sleep(0.1)
    write_register(RLD_SENS, rldsenseEMG)
    time.sleep(0.1)
    write_register(RESP1, resp1EMG)
    time.sleep(0.1)
    write_register(RESP2, resp2EMG)
    
def esperar_DRDY(chip, gpio, level, tick):
    # --- Función llamada cuando DRDY cambia a nivel bajo y que pone la variable dataReady a True ---
    global dataReady
    dataReady=True
    
def emg_inicializacion():
    # --- Función encargada de la inicialización y correcta configuración del ADS1292 previa a la adquisición de datos del EMG ---
    global chip
    configuration = False
    while not configuration: # Mientras no se haya configurado correctamente el ADS1292
        reset_ads1292() # Resetea el ADS1292
        emg_configuracion() # Configura el ADS1292
        configuration=comprobar_registros("emg")   # Verifica registros
        time.sleep(0.008)     # delay(8) ms en Arduino
    enable_continuous_reading() # Activa la lectura continua de datos
    time.sleep(0.01)
    start_conversion()
    time.sleep(0.01)
    
def stop_lectura_emg():
    # --- Función encargada de parar la adquisición de datos por parte del ADS1292 y el tratado posterior a las variables y pines involucrados en la adquisición ---
    global detected_peak, calibrationStarted, peakRangeComputed

    print("🛑 Deteniendo lectura EMG...")
    
    # Enviar comando RDATAC para desactivar lectura continua
    disable_continuous_reading()

    # Detener conversiones
    stop_conversion()
    
    #Liberar DRDY pin
    lgpio.gpio_free(chip, ADS1292_DRDY_PIN)
    calibrationStarted=False
    peakRangeComputed=False

    print("✅ EMG detenido completamente.")

def lectura_emg():
    # --- Función principal encargada de la configuración, adquisición y lectura del threshold en el modo emg ---
    global chip, dataReady, detected_peak, peakRangeComputed, calibrationDone 

    # Inicializa EMG (configuración, buffers, etc.)
    emg_inicializacion()
    
    # Reincia los valores de las variables del emg para que se haga la calibracion
    reiniciar_valores_emg()

    # Configura callback DRDY
    lgpio.gpio_claim_alert(chip, ADS1292_DRDY_PIN, lgpio.FALLING_EDGE, lgpio.SET_PULL_UP)
    lgpio.callback(chip, ADS1292_DRDY_PIN, lgpio.FALLING_EDGE, esperar_DRDY)

    # Inicia hilo de polling
    detected_peak = False
    poll_thread = threading.Thread(target=polling_drdy, daemon=True)
    poll_thread.start()

    # Espera hasta que se detecte un pico válido
    while not detected_peak:
        time.sleep(0.01)
    stop_lectura_emg()

def reiniciar_valores_emg():
     # --- Función encargada de reiniciar las variables asociadas al EMG para la próxima adquisición ---
    global peakRangeComputed, calibrationDone, calibrationStarted
    global peakCount, initialPeakCount
    global threshold, peakStdDev, lastPeakValue, lowPeakCount, bufferFillCount
    global peakValues, initialPeaks, emgBuffer
    
    # Reset de banderas de estado
    peakRangeComputed = False
    calibrationDone = False
    calibrationStarted = False

    # Reset de contadores y umbrales
    peakCount = 0
    initialPeakCount = 0
    threshold = 0
    peakStdDev = 0
    lastPeakValue = 0
    lowPeakCount = 0
    bufferFillCount = 0

    # Limpieza de buffers de picos y señal EMG
    peakValues.fill(0)
    initialPeaks.fill(0)
    emgBuffer.fill(0)


# --- FUNCIONES DETECCIÓN DE LA CONTRACCIÓN ---

def millis():
    # --- Función encargada de devolver el tiempo actual en milisegundos---
    return int(round(time.time() * 1000))


def adquirir_picos_iniciales(emg_value):
    # --- Función encargada de adquirir los picos iniciales---
    global initialPeakCount, peakRangeComputed, PEAK_MIN_HEIGHT, PEAK_MAX_HEIGHT

    if es_pico() and initialPeakCount < peakSampleSize : # Si es un pico y no se ha alcanzado el tamaño del buffer de picos iniciales se añade al buffer
        initialPeaks[initialPeakCount] = emgBuffer[1]
        initialPeakCount += 1

    if initialPeakCount >= peakSampleSize:
        sorted_peaks = np.sort(initialPeaks)
        # Median
        median = np.median(sorted_peaks)

        # Median Absolute Deviation (MAD)
        abs_diffs = np.abs(initialPeaks - median)
        mad = np.median(abs_diffs)

        print(f"📉 Median Absolute Deviation (MAD): {mad}")
        print(f"📊 Median: {median}")

        PEAK_MIN_HEIGHT = int(median - 5 * mad) # Ajuste del rango mínimo de picos
        PEAK_MAX_HEIGHT = int(median + 5 * mad) # Ajuste del rango máximo de picos

        peakRangeComputed = True

        print("✅ Peak range adjusted with tertiles:")
        print(f"PEAK_MIN_HEIGHT = {PEAK_MIN_HEIGHT}")
        print(f"PEAK_MAX_HEIGHT = {PEAK_MAX_HEIGHT}")


def calibrar_threshold():
    # --- Función encargada de calcular la mediana y la desviación standard para determinar un threshold---
    global emgBuffer, calibrationDone, threshold, peakStdDev, peakCount, lastPeakValue, lowPeakCount

    if es_pico_de_calibracion(emgBuffer[1]) and peakCount < sampleSize: # Si es un pico de calibración y no se ha alcanzado el tamaño del buffer de picos se añade al buffer
        peakValues[peakCount] = emgBuffer[1]
        peakCount += 1

    if peakCount >= sampleSize: # Si se ha alcanzado el tamaño del buffer de picos, se calcula la media y la desviación estándar
        mean_peak = calcular_media(peakValues, peakCount)
        peakStdDev = calcular_std_dev(peakValues, peakCount, mean_peak)
        threshold = mean_peak + 9* peakStdDev #Se calcula el threshold como la media más 9 veces la desviación estándar

        calibrationDone = True

        print("Calibration complete")
        print(f" Mean peak: {mean_peak}")
        print(f" Std deviation: {peakStdDev}")
        print(f" Threshold (mean + 9σ): {threshold}")

        lastPeakValue = 0
        lowPeakCount = 0

def detectar_contraccion():
    # --- Función encargada de detectar si un pico es superior al threshold---
    global detected_peak

    if es_pico():
        current_peak = emgBuffer[1]

        if threshold <current_peak: # Si el pico actual es superior al threshold se determina que se ha detectado una contracción
            print(f"Pico bajo detectado: {current_peak}")
            print("🟢 Muscle ON detected")
            detected_peak = True

def es_pico():
    # --- Función encargada de detectar si es un pico durante la comprovación de la detección muscular---
    return emgBuffer[1] > emgBuffer[0] and emgBuffer[1] >emgBuffer[2] 

def es_pico_de_calibracion(value):
    # --- Función encargada de detectar si es un pico durante la calibración ---
    is_local_peak = emgBuffer[1]>emgBuffer[0] and emgBuffer[1] > emgBuffer[2]
    height_in_range = PEAK_MIN_HEIGHT <= emgBuffer[1] <= PEAK_MAX_HEIGHT
    return is_local_peak and height_in_range 

def calcular_media(values, size):
    # --- Función encargada de calcular la media ---
    return int(np.sum(values[:size]) / size)


def calcular_std_dev(values, size, mean):
    # --- Función encargada de calcular la desviación standard ---
    return int(np.sqrt(np.sum((values[:size] - mean) ** 2) / size))

def emg_deteccion(valor):
    # --- Función encargada de gestionar el proceso de calibración y detección de la contracción ---
    global emgData, calibrationStarted, calibrationStartTime, bufferFillCount
    if not calibrationStarted:
        calibrationStartTime = millis()
        calibrationStarted = True

    emgBuffer[0] = emgBuffer[1]
    emgBuffer[1] = emgBuffer[2]
    emgBuffer[2] = valor

    if bufferFillCount < 3:
        bufferFillCount += 1
        return

    if millis() - calibrationStartTime < 6000: # Si han pasado menos de 6 segundos desde el inicio de la función, se hace return sin hacer nada
        return

    if not peakRangeComputed:
        adquirir_picos_iniciales(emgData) # Si no se ha calculado el rango de picos, se adquieren los picos iniciales
        return

    if not calibrationDone:
        calibrar_threshold() # Si no se ha hecho la calibración, se calibra el threshold
        return

    detectar_contraccion() # Se detecta si hay contracción muscular

def polling_drdy():
    # --- Función principal de gestión de la lectura por EMG ---
    global chip, emgData, dataReady, live_filter, detected_peak
    while not detected_peak:
        if dataReady:
            dataReady=False
            # Flanco detectado
            leer_valores_ADS1292()
            emg_valor=emgData
            valor_filtrado =round(live_filter(emg_valor), 4)
            #print(valor_filtrado)
            emg_deteccion(valor_filtrado)
        time.sleep(0.00002)  # 200 us para no saturar CPU demasiado

class LiveSosFilter:
    # --- Clase que hace el filtro pasa bajos aplicado ---
    def __init__(self, sos):
        self.sos = np.atleast_2d(sos)
        self.n_sections = self.sos.shape[0]
        self.state = np.zeros((self.n_sections, 2))

    def __call__(self, x):
        """Apply SOS filter to a single sample."""
        for section in range(self.n_sections):
            b = self.sos[section, :3]
            a = self.sos[section, 3:]
            y = b[0] * x + self.state[section][0]
            self.state[section][0] = b[1] * x - a[1] * y + self.state[section][1]
            self.state[section][1] = b[2] * x - a[2] * y
            x = y
        return y
sos = scipy.signal.iirfilter(
    N=order,
    Wn=fc,
    fs=fs,
    btype='low',
    ftype='butter',
    output='sos'
)
live_filter = LiveSosFilter(sos)

# --- FUNCIONES REALIMENTACIÓN SERVOMOTORES ---

def servo_configuracion():
    # --- Función que configura los registros del ADS1292 para la lectura de la realimentación de los servomotores ---
    disable_continuous_reading()
    write_register(CONFIG1, config1Value)
    time.sleep(0.1)
    write_register(CONFIG2, config2Servo)
    time.sleep(0.1)
    write_register(CH1SET, ch1setServo)
    time.sleep(0.1)
    write_register(CH2SET, ch2setServo)
    time.sleep(0.1)
    write_register(RLD_SENS, rldsenseServo)
    time.sleep(0.1)
    write_register(RESP1, resp1Servo)
    time.sleep(0.1)
    write_register(RESP2, resp2Servo)

def servo_inicializacion():
    # --- Función encargada de la inicialización y correcta configuración del ADS1292 previa a la adquisición de datos de la retroalimentación de los servomotores ---
    global chip
    configuracion = False
    while not configuracion:
        reset_ads1292()
        servo_configuracion()
        configuracion=comprobar_registros("servo")   # Verifica registros
        time.sleep(0.008)     # delay(8) ms en Arduino
    start_conversion()
    time.sleep(0.01)

def servo_config():
    # --- Función principal encargada de gestionar las funciones para inicializar la configuración de la adquisición de datos de retroalimentación de los servomotores ---
    global chip, spi, dataReady
    activar_spi()
    servo_inicializacion()
    
def realimentacion(dedo):
    # --- Función que adquiere el valor del ADS1292, hace la transformación a intensidad y comprueba si esta supera el threshold ---
    global chip, spi, dataReady, servoData, CMD_RDATA, vref, currentThreshold
    lgpio.gpio_claim_alert(chip, ADS1292_DRDY_PIN, lgpio.FALLING_EDGE, lgpio.SET_PULL_UP)
    lgpio.callback(chip, ADS1292_DRDY_PIN, lgpio.FALLING_EDGE, esperar_DRDY)
    while not dataReady:
        time.sleep(0.001)  # Espera 1 ms para no saturar CPU
    if dataReady:
        dataReady=False
        send_command(CMD_RDATA) # Envía el comando de lectura de datos
        leer_valores_ADS1292()
        servo_valor=round(servoData,4) # Lee el valor del servomotor
        print("valor ads ")
        print(servo_valor)
        voltaje = servo_valor * vref / 8388607.0 # Convierte el valor a voltaje
        intensidad= voltaje/shuntResistance     # Calcula la intensidad de corriente
        print("intensidad ") 
        print(intensidad)
        return intensidad>currentThreshold # Devuelve si la intensidad supera el threshold
    
def servo_stop():
    # --- Función que detiene la adquisición del ADS1292, libera la GPIO ADS1292_DRDY_PIN y termina la comunicación SPI  ---
    global chip, CMD_STOP
    send_command(CMD_STOP)
    lgpio.gpio_free(chip, ADS1292_DRDY_PIN)
    desactivar_spi()
    
    
def main():
    print("🔧 Inicializando GPIO y SPI...")
    config_ini()
    spi_ini()

    if not comprobar_spi():
        print("❌ No se pudo establecer conexión con ADS1292.")
        return

    print("⚙️ Iniciando lectura de EMG con calibración...")
    lectura_emg()

    print("🎯 Pico de activación muscular detectado. Proceso finalizado.")

if __name__ == "__main__":
    main()
        
    

