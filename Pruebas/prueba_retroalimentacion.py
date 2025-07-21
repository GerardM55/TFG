import time
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
import ADS1292
import lgpio

# --- VARIABLES GLOBALES ---
MUX_A1_PIN = 23  # Pin A1 del MUX
MUX_A0_PIN = 22  # Pin A0 del MUX
MUX_EN_PIN = 24  # Pin EN del MUX
pca = None 
i2c_bus = None

# --- Función para inicializar el bus I2C ---
def abrir_i2c():
    global pca, i2c_bus 
    # Crear el bus I2C
    i2c_bus = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c_bus)
    pca.frequency = 50  # 50Hz para servos MG996R

# --- Canal asociado a cada dedo ---
dedos = {
    "pulgar": 4,
    "indice": 0,
    "medio_anular": 6,
    "menique": 2,
    "rot": 8
}

# --- Diccionario de agarres con valor y tipo de movimiento (S o R) ---
agarre_config = {
    "obliquo": [
        ["indice", 0.5, "S"],
        ["medio_anular", 2.5, "S"],
        ["menique", 0.5, "S"],
        ["pulgar", 0.9, "S"],
    ],
    "general": [
        ["pulgar", 1, "R"],
        ["rot", 0.7, "R"],
        ["indice", 0.5, "R"],
        ["medio_anular", 2.5, "R"],
        ["menique", 0.5, "R"]
    ],
    "pinza": [
        ["rot", 0.6, "S"],
        ["pulgar", 1, "S"],
        ["indice", 0.9, "S"]
    ],
    "cilindrico": [
        ["indice", 0.5, "S"],
        ["medio_anular", 2.5, "S"],
        ["menique", 0.5, "S"],
        ["rot", 0.8, "S"],
        ["pulgar", 0.9, "S"]
    ],
    "pinza lateral": [
        ["indice", 0.5, "R"],
        ["rot", 0.8, "S"],
        ["pulgar", 1.2, "S"]
    ],
    "gancho": [
        ["indice", 0.5, "S"],
        ["medio_anular", 2.5, "S"],
        ["menique", 0.5, "S"]
    ],
    "stop": [
        ["pulgar", 2.5, "R"],
        ["indice", 2.5, "R"],
        ["medio_anular", 0.5, "R"],
        ["menique", 2.5, "R"],
        ["rot", 1.5, "R"]
    ]
}

# --- Función para mover el servomotor ---
def mover_servo(canal, pulse_ms):
    global pca
    duty_cycle = int(pulse_ms * 65535 / 20.0)  # Convertir a duty cycle
    pca.channels[canal].duty_cycle = duty_cycle

# --- Función para seleccionar el canal de movimiento del servomotor en la PCA9685W en función del dedo ---
def mover_directo(dedo, posicion):
    canal = dedos[dedo]
    mover_servo(canal, posicion)

# --- Función para obtener la posición del dedo ---
def obtener_posicion_stop(dedo):
    for d, valor, _ in agarre_config["stop"]:
        if d == dedo:
            return valor
    return 2.5  # Valor por defecto si no está en 'stop'

# --- Función para mover por pasos el dedo. ---
def mover_suavemente(dedo, objetivo, delay=0.03, paso=0.01):
    canal = dedos[dedo]
    if dedo != "rot":
        configurarMUX(dedo)
    
    # Obtener la posición de 'stop' del dedo
    posicion_actual = obtener_posicion_stop(dedo)
    print(posicion_actual)

    if objetivo < posicion_actual:
        paso = -abs(paso)
    else:
        paso = abs(paso)

    while abs(posicion_actual - objetivo) > 0.01:
        mover_servo(canal, posicion_actual)
        time.sleep(0.01)

        if dedo != "rot":
            if leer_señal_control(dedo) and False:
                print(f"¡Consumo alto! Deteniendo movimiento suave.")
                break

        posicion_actual += paso

        # Clamp si se pasa
        if paso > 0 and posicion_actual > objetivo:
            posicion_actual = objetivo
        elif paso < 0 and posicion_actual < objetivo:
            posicion_actual = objetivo

    mover_servo(canal, objetivo)

# --- Función que gestiona el tipo del movimiento de cada dedo en función del agarre deseado ---
def gestion_servos(agarre):
    if agarre not in agarre_config:
        print(f"Agarre '{agarre}' no definido.")
        return
    for dedo, valor, tipo_mov in agarre_config[agarre]:
        if tipo_mov == "S":
            print("dentro movimiento suave")
            mover_suavemente(dedo, valor)
        elif tipo_mov == "R":
            mover_directo(dedo, valor)
        else:
            print(f"Tipo de movimiento desconocido '{tipo_mov}' para {dedo}")

# --- Función principal que ejecuta un agarre completo (con o sin realimentación) ---
def ejecutar_agarre(nombre_agarre):
    global i2c_bus
    abrir_i2c()
    if nombre_agarre not in ("stop", "general"):
        ADS1292.servo_config()
    gestion_servos(nombre_agarre)
    if nombre_agarre not in ("stop", "general"):
        ADS1292.servo_stop()
    i2c_bus.deinit()

# --- Función que adquiere el valor del ADS1292, hace la transformación a intensidad y comprueba si esta supera el threshold ---
def leer_señal_control(dedo):
    return ADS1292.realimentacion(dedo)

# --- Función que configura el MUX para redirigir la señal del dedo correspondiente ---
def configurarMUX(dedo):
    global chip
    lgpio.gpio_write(chip, MUX_EN_PIN, 1)
    print("configurar dedo ")
    print(dedo)
    lgpio.gpio_write(chip, MUX_A0_PIN, valoresMuxDedo[dedo]["A0"])
    lgpio.gpio_write(chip, MUX_A1_PIN, valoresMuxDedo[dedo]["A1"])

# --- Función para declarar como salida los pines del MUX ---
def activar_gpiosMUX():
    global chip
    lgpio.gpio_claim_output(chip, MUX_A0_PIN)
    lgpio.gpio_claim_output(chip, MUX_A1_PIN)
    lgpio.gpio_claim_output(chip, MUX_EN_PIN)

# --- Función que detiene la adquisición del ADS1292, libera la GPIO ADS1292_DRDY_PIN y termina la comunicación SPI ---
def servo_stop():
    global chip
    lgpio.gpio_write(chip, MUX_EN_PIN, 0)
    lgpio.gpio_free(chip, MUX_EN_PIN)
    lgpio.gpio_free(chip, MUX_A0_PIN)
    lgpio.gpio_free(chip, MUX_A1_PIN)

# --- Ejecución principal para pruebas ---
if __name__ == "__main__":
    # Paso 1: Movimiento "stop"
    print("Ejecutando movimiento: stop")
    ejecutar_agarre("stop")
    time.sleep(1)
    ADS1292.config_ini()

    # Paso 2: Movimiento de un dedo (por ejemplo, índice)
    print("Ejecutando movimiento de dedo individual: menique")
    abrir_i2c()
    ADS1292.servo_config()
    activar_gpiosMUX()
    mover_suavemente("indice", 0.9)  # Cambia el valor si deseas otra posición
    servo_stop()
    i2c_bus.deinit()
    time.sleep(3)

    # Paso 3: Volver a "stop"
    print("Volviendo a movimiento: stop")
    ejecutar_agarre("stop")
