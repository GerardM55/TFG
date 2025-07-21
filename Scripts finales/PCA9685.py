import time
import lgpio
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
import ADS1292

# --- VARIABLES ---
pca=None # Objeto PCA9685
i2c_bus=None # Bus I2C
chip=None # Variable para el chip de GPIO
# --- GPIOS MUX---
MUX_A1_PIN = 23
MUX_A0_PIN = 22
MUX_EN_PIN = 24

# --- DICCIONARIOS ---

# Diccionario que asocia cada dedo al canal del módulo PCA9685W corresponediente
dedos = {
    "pulgar": 4,
    "indice": 0,
    "medio_anular": 6,
    "menique": 2,
    "rot": 8
}

# Diccionario de los agarres disponibles. Cada agarre contiene los dedos involucrados, sus rango máximo de movimiento y el tipo de movimiento ("S") o ("R")
agarre_config = {
    "obliquo": [
        ["indice", 0.5, "S"],
        ["medio_anular", 2.5, "S"],
        ["menique", 0.5, "S"],
        ["pulgar", 1.2, "S"],
    ],
    "general": [
        ["pulgar", 0.8, "R"],
        ["rot", 0.8, "R"],
        ["indice", 0.5, "R"],
        ["medio_anular", 2.5, "R"],
        ["menique", 0.5, "R"]
        
    ],
    "pinza": [
        ["rot", 0.5, "S"],
        ["pulgar", 1, "S"],
        ["indice", 0.9, "S"]
    ],
    "cilindrico": [
        ["indice", 0.5, "S"],
        ["medio_anular", 2.5, "S"],
        ["menique", 0.5, "S"],
        ["rot", 0.8, "S"],
        ["pulgar", 1.3, "S"]
    ],
    "pinza lateral": [
        ["indice", 0.5, "R"],
        ["rot", 1.3, "S"],
        ["pulgar", 0.7, "S"]
    ],
    "gancho": [
        ["indice", 0.5, "S"],
        ["medio_anular", 2.5, "S"],
        ["menique", 0.5, "S"],
        ["pulgar", 0.5, "S"]
    ],
    "stop": [
        ["pulgar", 2.5, "R"],
        ["indice", 2.5, "R"],
        ["medio_anular", 0.5, "R"],
        ["menique", 2.5, "R"],
        ["rot", 1.5, "R"]
    ]
}
# Diccionario de los valores de las variables del MUX en función de los dedos
valoresMuxDedo = {
    "pulgar":   {"A0": 0, "A1": 0},
    "indice":   {"A0": 0, "A1": 1},
    "medio_anular": {"A0": 1, "A1": 0},
    "menique":  {"A0": 1, "A1": 1},
}


# --- Funciones ---

def abrir_i2c():
    # --- Crear bus I2C ---
    global pca,i2c_bus 
    i2c_bus = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c_bus)
    pca.frequency = 50  # 50Hz 
    
def mover_servo(canal, pulse_ms):
    # --- Función para mover el servomor de canal al pulso pulse_ms---
    global pca
    duty_cycle = int(pulse_ms * 65535 / 20.0) # Convertir el pulso a duty cycle
    pca.channels[canal].duty_cycle = duty_cycle # Mover el servo al duty cycle correspondiente

def mover_directo(dedo, posicion):
    # --- Función para seleccionar el canal correspondiente a dedo y ejecutar la Función mover_servo del dedo al destino posicion---
    canal = dedos[dedo]
    mover_servo(canal, posicion)
    
def obtener_posicion_stop(dedo):
    # --- Función para obtener el pulso en ms del dedo dedo en la posición stop ---
    for d, valor, _ in agarre_config["stop"]:
        if d == dedo:
            return valor
    return 2.5 if dedo == "rot" else 1.5

def mover_suavemente(dedo, objetivo, delay=0.03, paso=0.01):
    # --- Función para hacer el movimiento de un dedo con control por realimentación ---
    canal = dedos[dedo]
    if dedo!="rot":
        configurarMUX(dedo) # Configura el MUX para el dedo correspondiente
    posicion_actual = obtener_posicion_stop(dedo)  # La posición inicial es la posición de extensión, que corresponde a la posición final del movimiento stop.
    if objetivo < posicion_actual: # Si el objetivo es menor que la posición actual, el paso debe ser negativo
        paso = -abs(paso)
    else:
        paso = abs(paso)

    while abs(posicion_actual - objetivo) > 0.01: # Mientras la posición actual no sea igual al objetivo
        mover_servo(canal, posicion_actual)
        time.sleep(delay)
        
        if dedo!="rot":
            if leer_señal_control(dedo):
                print(f"¡Consumo alto! Deteniendo movimiento suave.")
                break
        
        posicion_actual += paso
        if paso > 0 and posicion_actual > objetivo:
            posicion_actual = objetivo
        elif paso < 0 and posicion_actual < objetivo:
            posicion_actual = objetivo

    mover_servo(canal, objetivo) #

def gestion_servos(agarre):
    # --- Función que en Función de la variable tipo_mov determina si el movimiento se realiza con la función mover_suavemente o mover_directo ---
    if agarre not in agarre_config:
        print(f"Agarre '{agarre}' no definido.")
        return
    for dedo, valor, tipo_mov in agarre_config[agarre]:
        if tipo_mov == "S":
            mover_suavemente(dedo, valor)
        elif tipo_mov == "R":
            mover_directo(dedo, valor)
        else:
            print(f"Tipo de movimiento desconocido '{tipo_mov}' para {dedo}")

def ejecutar_agarre(nombre_agarre):
    # --- Función encargada de llamar a la inicialización del bus I2C y la comunicación principal con el script ADS1292 para activar los registros para la realimentación de los servomotores---
    global  i2c_bus
    abrir_i2c()
    if nombre_agarre not in ("stop", "general"):
        ADS1292.servo_config() # Configura el ADS1292 para la realimentación de los servos
        activar_gpiosMUX() # Activa los GPIOs del MUX
    gestion_servos(nombre_agarre)
    if nombre_agarre not in ("stop", "general"):
        ADS1292.servo_stop() # Detiene la adquisición del ADS1292 y libera los GPIOs del ADS1292
        servo_stop() 
    i2c_bus.deinit() # Libera el bus I2C

def leer_señal_control(dedo):
    # --- Función que devuelve true or false en función de si el valor de la retroalimentación es superior al threshold---
    return ADS1292.realimentacion(dedo)

def configurarMUX(dedo):
    # --- Función que configura la salida del MUX en función de dedo ---
    global chip
    print("configurar dedo ")
    print (dedo)
    lgpio.gpio_write(chip, MUX_A0_PIN, valoresMuxDedo[dedo]["A0"])
    lgpio.gpio_write(chip, MUX_A1_PIN, valoresMuxDedo[dedo]["A1"])
    
def activar_gpiosMUX():
    # --- Función que activa las GPIOS del MUX y el pin ENABLE lo pone en HIGH ---
    global chip
    lgpio.gpio_claim_output(chip, MUX_A0_PIN)
    lgpio.gpio_claim_output(chip, MUX_A1_PIN)
    lgpio.gpio_claim_output(chip, MUX_EN_PIN)
    lgpio.gpio_write(chip, MUX_EN_PIN, 1)
    
def servo_stop():
    # --- Función que pone el pin ENABLE en LOW y desactiva las GPIOS del MUX ---
    global chip
    lgpio.gpio_write(chip, MUX_EN_PIN, 0)
    lgpio.gpio_free(chip, MUX_EN_PIN)
    lgpio.gpio_free(chip, MUX_A0_PIN)
    lgpio.gpio_free(chip, MUX_A1_PIN)        
    
def compartir_chip(Chip):
    # --- Función que configura la variable chip con el valor Chip  ---
    global chip
    chip=Chip    

