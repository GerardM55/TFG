import time
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685

#SDA=GPIO 2 y SCL=GPIO3. 
# Crear el bus I2C
i2c_bus = busio.I2C(SCL, SDA)

# Crear el objeto del controlador PCA9685
pca = PCA9685(i2c_bus)
pca.frequency = 50  # 50Hz para servos MG996R

# Canal del servo
canal = 0 # Cambia según el canal que uses

# Función para mover el servo directamente con ms
def mover_servo_ms(pulse_ms):
    duty_cycle = int(pulse_ms * 65535 / 20.0)  # 20 ms de periodo (50Hz)
    pca.channels[canal].duty_cycle = duty_cycle # Convertir ms a duty cycle

# Calibrar con dos posiciones estándar
def calibrar_servo_estandar():
    posiciones = {
        "-90° (izquierda total)": 0.9, # Pulso en ms para -90°
        "+90° (derecha total)": 2.5 # Pulso en ms para +90°
    }

    for descripcion, pulse in posiciones.items(): 
        input(f"\nPulsa Enter para mover el servo a {descripcion} (pulso: {pulse} ms)...")
        mover_servo_ms(pulse)
        time.sleep(1)

    print("\n✓ Calibración básica completada.")

# Ejecución
try:
    calibrar_servo_estandar() # Llamar a la función de calibración
except KeyboardInterrupt:
    print("\nInterrumpido por el usuario.")
finally:
    pca.deinit() # Liberar el PCA9685
    print("Programa finalizado.")
