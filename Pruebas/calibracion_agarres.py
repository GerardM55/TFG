import time
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
import ADS1292

# Variables
pca=None
i2c_bus=None

# Función para inicializar el bus I2C
def abrir_i2c():
    global pca,i2c_bus 
    # Crear el bus I2C
    i2c_bus = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c_bus)
    pca.frequency = 50  # 50Hz para servos MG996R
    
# Canal asociado a cada dedo
dedos = {
    "pulgar": 4,
    "indice": 0,
    "medio_anular": 6,
    "menique": 2,
    "rot": 8
}

# Diccionario de agarres con valor y tipo de movimiento (S o R)
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
# Función para mover el servomotor
def mover_servo(canal, pulse_ms):
    global pca
    duty_cycle = int(pulse_ms * 65535 / 20.0) # Convertir a duty cycle
    pca.channels[canal].duty_cycle = duty_cycle 

# Función para seleccionar el canal de movimiento del servomotor en la PCA9685W en función del dedo
def mover_directo(dedo, posicion):
    canal = dedos[dedo]
    mover_servo(canal, posicion)

# Función para obtener la posición del dedo
def obtener_posicion_stop(dedo):
    for d, valor, _ in agarre_config["stop"]:
        if d == dedo:
            return valor
    return 2.5  # Valor por defecto si no está en 'stop'

# Función para mover por pasos el dedo. 
def mover_suavemente(dedo, objetivo, delay=0.03, paso=0.05):
    canal = dedos[dedo]
    posicion_actual = obtener_posicion_stop(dedo) # Obtener la posición de 'stop' del dedo
    if objetivo < posicion_actual:
        paso = -abs(paso)
    else:
        paso = abs(paso)

    while abs(posicion_actual - objetivo) > 0.05: # Si la posición actual menos el objetivo es mayor que 0.05
        mover_servo(canal, posicion_actual)
        time.sleep(delay) 
        posicion_actual += paso

        if paso > 0 and posicion_actual > objetivo:
            posicion_actual = objetivo
        elif paso < 0 and posicion_actual < objetivo:
            posicion_actual = objetivo

    mover_servo(canal, objetivo)
    
# Función que gestiona el tipo del movimiento de cada dedo en función del agarre deseado
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

def main():
    global  i2c_bus
    try:  
        abrir_i2c() # Inicializar el bus I2C
        print("Ejecutando posición de reposo (stop)...")
        gestion_servos("stop") # ejecutar el agarre "stop"
        time.sleep(2)

        agarre = "general"  # Cambia este valor por el agarre que quieras usar
        print(f"Ejecutando agarre: {agarre}...")
        gestion_servos(agarre) # Ejecutar el agarre deseado
        time.sleep(6)

        print("Volviendo a posición de reposo (stop)...")
        gestion_servos("stop") # Ejecutar el agarre "stop" al finalizar
        print("Proceso finalizado.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()




