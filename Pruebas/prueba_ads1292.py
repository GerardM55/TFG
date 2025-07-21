import lgpio
import spidev
import time

# Comandas del ADS1292
CMD_READ_REG = 0x20
CMD_STOP = 0x0A
CMD_SDATAC = 0x11
CMD_RESET = 0x06

# Pines GPIO (BCM)
ADS1292_CS_PIN = 8     # CS - reservado por hardware SPI
ADS1292_DRDY_PIN = 5   # DRDY - entrada
ADS1292_START_PIN = 27 # START - salida
ADS1292_PWDN_PIN = 17  # PWDN - salida
ADS1292_CLKSEL_PIN = 4 # CLKSEL - salida

# Inicializa GPIO
chip = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(chip, ADS1292_START_PIN)
lgpio.gpio_claim_output(chip, ADS1292_PWDN_PIN)
lgpio.gpio_claim_output(chip, ADS1292_CLKSEL_PIN)
lgpio.gpio_claim_input(chip, ADS1292_DRDY_PIN)

# Inicializa SPI
spi = spidev.SpiDev()
spi.open(0, 0)  # Bus 0, CE0
spi.max_speed_hz = 500000
spi.mode = 0b01

def ads1292_write_command(cmd): # Escribe un comando al ADS1292
    spi.xfer2([cmd])

def ads1292_read_register(reg_address): # Lee un registro del ADS1292
    spi.xfer2([CMD_READ_REG | reg_address, 0x00])
    return spi.xfer2([0x00])[0]

def reinicialitza_ads1292(): # Reinicializa el ADS1292
    lgpio.gpio_write(chip, ADS1292_PWDN_PIN, 0) # Pone PWDN a 0
    time.sleep(0.002)
    lgpio.gpio_write(chip, ADS1292_PWDN_PIN, 1) # Pone PWDN a 1
    time.sleep(0.1)
    ads1292_write_command(CMD_RESET) # Envía el comando de reset
    time.sleep(0.01)
    ads1292_write_command(CMD_SDATAC) # Envía el comando SDATAC
    time.sleep(0.01)
    lgpio.gpio_write(chip, ADS1292_START_PIN, 1) # Pone START a 1

try:
    # Inicialización de pines
    lgpio.gpio_write(chip, ADS1292_START_PIN, 0)
    lgpio.gpio_write(chip, ADS1292_PWDN_PIN, 0)
    lgpio.gpio_write(chip, ADS1292_CLKSEL_PIN, 1)
    time.sleep(0.005)
    lgpio.gpio_write(chip, ADS1292_PWDN_PIN, 1)
    time.sleep(0.1)
    lgpio.gpio_write(chip, ADS1292_START_PIN, 1)
    time.sleep(0.01)

    print("Iniciando SPI...")
    ads1292_write_command(CMD_RESET)
    time.sleep(0.01)
    ads1292_write_command(CMD_SDATAC)
    time.sleep(0.01)
    lgpio.gpio_write(chip, ADS1292_START_PIN, 1)
    id_val = 0
    while id_val!=0x53: # Verifica el ID del ADS1292
        
        intentos = 0

        while intentos < 3:
            id_val = ads1292_read_register(0x00) 
            print(f"Intentos {intentos + 1}: Registro ID (0x00)= 0x{id_val:02X}")

            if id_val == 0x53:
                break

            intentos += 1
            reinicialitza_ads1292()
            time.sleep(0.1)

        if id_val != 0x53:
            print("⚠️ Error: No se ha detectado ID válido después de 3 intentos.")
        else:
            print("✅ ID válido detectado.")

        time.sleep(2)

except KeyboardInterrupt:
    print("\nPrograma terminado por el usuario.")

finally:
    spi.close()
    lgpio.gpiochip_close(chip)
    print("SPI cerrado y GPIO limpiado correctamente.")
