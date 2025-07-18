import lgpio
import spidev
import time

# Comandes de l'ADS1292R
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

def ads1292r_write_command(cmd):
    spi.xfer2([cmd])

def ads1292r_read_register(reg_address):
    spi.xfer2([CMD_READ_REG | reg_address, 0x00])
    return spi.xfer2([0x00])[0]

def reinicialitza_ads1292r():
    lgpio.gpio_write(chip, ADS1292_PWDN_PIN, 0)
    time.sleep(0.002)
    lgpio.gpio_write(chip, ADS1292_PWDN_PIN, 1)
    time.sleep(0.1)
    ads1292r_write_command(CMD_RESET)
    time.sleep(0.01)
    ads1292r_write_command(CMD_SDATAC)
    time.sleep(0.01)
    lgpio.gpio_write(chip, ADS1292_START_PIN, 1)

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

    print("Iniciant SPI...")
    ads1292r_write_command(CMD_RESET)
    time.sleep(0.01)
    ads1292r_write_command(CMD_SDATAC)
    time.sleep(0.01)
    lgpio.gpio_write(chip, ADS1292_START_PIN, 1)
    id_val = 0
    while id_val!=0x53:
        
        intents = 0

        while intents < 3:
            id_val = ads1292r_read_register(0x00)
            print(f"Intent {intents + 1}: Registre ID (0x00)= 0x{id_val:02X}")

            if id_val == 0x53:
                break

            intents += 1
            reinicialitza_ads1292r()
            time.sleep(0.1)

        if id_val != 0x53:
            print("⚠️ Error: No s'ha detectat ID vàlid després de 3 intents.")
        else:
            print("✅ ID vàlid detectat.")

        time.sleep(2)

except KeyboardInterrupt:
    print("\nPrograma terminado por el usuario.")

finally:
    spi.close()
    lgpio.gpiochip_close(chip)
    print("SPI cerrado y GPIO limpiado correctamente.")
