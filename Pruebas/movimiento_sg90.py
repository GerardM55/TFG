import time
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685

# Crear el bus I2C
i2c_bus = busio.I2C(SCL, SDA)

# Crear el objeto del controlador PCA9685
pca = PCA9685(i2c_bus)
pca.frequency = 50  # 50Hz para servos

canal = 8 # Cambia según el canal que uses

# Función para mover el servo
def mover_servo_ms(pulse_ms): 
    duty_cycle = int(pulse_ms * 65535 / 20.0)
    pca.channels[canal].duty_cycle = duty_cycle

# Función para mover el servo suavemente entre dos posiciones
def mover_suavemente(desde, hasta, pasos=20, retardo=0.05): #
    paso = (hasta - desde) / pasos 
    for i in range(pasos + 1): 
        pulso_actual = desde + paso * i
        mover_servo_ms(pulso_actual) 
        time.sleep(retardo)

# Movimiento completo de 0° a -90° y regreso con suavidad
def movimiento_suave_ida_y_vuelta():
    print("Moviendo suavemente a -90°...")
    mover_suavemente(1.5, 0.5, pasos=30, retardo=0.03)
    #mover_suavemente(1.5, 0.7, pasos=30, retardo=0.03)

    time.sleep(5.5)

    print("Volviendo suavemente a 0°...")
    mover_suavemente(0.5, 1.5, pasos=30, retardo=0.03)
    #mover_suavemente(0.7, 1.5, pasos=30, retardo=0.03)

# Ejecución
try:
    movimiento_suave_ida_y_vuelta() # Llamar a la función de movimiento
except KeyboardInterrupt:
    print("\nInterrumpido por el usuario.")
finally:
    pca.deinit()
    print("Programa finalizado.")
