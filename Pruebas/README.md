# Descripción del directorio
En este directorio estan disponibles todos los archivos utilizados para las pruebas del proyecto. 
A continuación, se describe el funcionamiento y los requisitos necesarios para el uso de cada script. 
## calibracion_agarres.py
La función de este script es calibrar los rangos de los servomotores en función del agarre deseado. 

### Descripción
El script realiza las siguientes acciones: 
* Inicializa el bus I2C y el módulo PCA9685.

* Define los canales y movimientos correspondientes a cada dedo.
* Ejecuta el agarre "stop".
* Ejecuta el agarre deseado.
* Regresa a la posición de reposo mediante la ejecución del aggare "stop".

### Requisitos
Las líbrerias necesarias son las siguientes: 
````
pip install adafruit-circuitpython-pca9685 adafruit-blinka
````
### Salida
La salida del script es la ejecución del agarre. El usuario debe modificar manualmente la variable 
**agarre** con un agarre dentro del diccionario **agarre_config**. 
También debe modificar los valores numéricos de caga agarre del diccionario **agarre_config** en función del resultado
del agarre que observe y del rango de movimiento permitido por cada servomotor. 

## calibracion_agarres.py
Este script es el encargado de calibrar el rango de movimientos de cada servomotor por individual. 

### Descripción
El script realiza lo siguiente: 
* Inicializa el bus I2C.
* Configura la frecuencia a 50 Hz.
* Mueve el servo a dos posiciones estándar:
  *  -90° (izquierda): pulso de X ms
  *  +90° (derecha): pulso de X ms
* Espera que el usuario presione Enter antes de mover a cada posición.

### Requisitos 
Las líbrerias necesarias son las siguientes: 
````
pip install adafruit-circuitpython-pca9685 adafruit-blinka
````
### Salida
La salida del script es el movimiento de cada servomotor. Para ajustar este movimiento, el usuario debe modificar
manualmente los valores de la variable **posiciones**. 

## comunicacionserieads.zip
El archivo zip contiene el proyecto de PlatformIO utilizado para obtener la señal EMG y hacer las pruebas con el ESP32

### Descripción 
El archivo **main.cpp** realiza las siguientes funciones: 
* Inicia la comunicación SPI del ADS1292 con el ESP32.
* Configura los registros del ADS1292
* Espera 6 segundos antes de iniciar la calibración de picos
* Adquiere 200 picos y calcula la mediana y la MAD. 
* Aplica unos rangos de detección de picos en función de los valores anteriores
* Adquiere 30 picos y calcula un threshold en función de la media y la desviación estándar
* Adquiere la señal y calcula si los picos son superiores al threshold para determinar si hay contracción muscular

### Requisitos
Las dependencias utilizadas son las siguientes: 
````
#include <Arduino.h>
#include <SPI.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <stdio.h>
#include <Filters.h>
#include <AH/Timing/MillisMicrosTimer.hpp>
#include <Filters/Notch.hpp>
#include <vector>
#include <Filters/MedianFilter.hpp> 
#include <Filters/Butterworth.hpp>
#include <AH/Timing/MillisMicrosTimer.hpp>
````
### Salida
La salida del archivo es la indicación de la terminal del momento en que ha detectado que un pico es superior al threshold. 
El usuario debe ajustar manualmente los siguientes valores:
* El número que múltiplica la variable **mad** en las variables **PEAK_MIN_HEIGHT** y **PEAK_MAX_HEIGHT**. 
* El número que múltplica la variable **peakStdDev** para obtener la variable **threshold**.
* Descomentar la línea 430 y 431 y comentar desde la linea 432 a la 541 en función de si quiere
* ver la señal por puerto serie o ejecutar el algoritmo de detección de la contracción.

## movimiento_sg90.py
Este script es el encargado de la calibración del servomotor SG90 que realiza el movimiento de abducción 
y aducción del dedo pulgar. 
### Descripción
El script realiza las siguientes acciones:  
* Inicializa el bus I2C y el módulo PCA9685.  
* Define el canal PWM que controla el servo.  
* Ejecuta un movimiento suave del servo desde la posición inicial a la final y regresa.  
* Libera los recursos del PCA9685.

### Requisitos
Las librerías necesarias son las siguientes:
````
pip install adafruit-circuitpython-pca9685 adafruit-blinka
````
### Salida
El output del script es el movimiento de aducción/abducción del servomotor SG90. El usuario debe 
editar manualmente los dos primeros parámetros de la función **mover_suavemente()** en función de los rangos de movimiento. 

## prueba_ads1292.py
Este script es el encargado de verificar la correcta comunicación SPI entre el ADS1292 y la Raspberry Pi Zero 2W. 

### Descripción
El script realiza las siguientes acciones:  
* Inicializa los pines GPIO necesarios para la comunicación con el ADS1292.  
* Inicializa la interfaz SPI para establecer comunicación con el chip.  
* Envía comandos de reinicio y configuración al ADS1292.  
* Lee el registro de ID del chip y verifica si la conexión es correcta.  
* Si no se detecta el ID válido (`0x53`), reintenta hasta 3 veces.  
* Finaliza cerrando SPI y liberando los recursos GPIO

### Requisitos
Las librerías necesarias son las siguientes: 
````
pip install lgpio spidev
````
### Salida
La salida del script es la verificación por consola de la correcta comunicación por SPI. Si se establece comunicación
se muestra por pantalla: ✅ ID válido detectado. Contrariamente, se muestra: ⚠️ Error: No se ha detectado ID válido después de 3 intentos.

## prueba_audio_sinpulsador.py y prueba_vosk+ics43434.py
Estos dos scripts se ejecutan conjuntamente para realizar la misma prueba. Esta consiste en verificar el tiempo de ejecución de ambos scripts. 

### Descripción

El conjunto de scripts realiza las siguientes acciones:

* Inicializa el micrófono conectado por I2S para capturar audio cuando se detecta una señal en el pin GPIO 26.
* Graba audio en formato WAV mientras se mantiene la señal activa.
* Una vez terminada la grabación, guarda el archivo en disco y ejecuta un segundo script de análisis.
* El segundo script convierte el audio a un formato compatible con Vosk y realiza transcripción automática del contenido hablado.
* La transcripción es clasificada.
* El resultado final muestra por consola la **transcripción** y la **etiqueta de clasificación** correspondiente.
* Finalmente, el archivo de audio temporal se elimina del sistema.

### Requisitos

Las librerías necesarias para ambos scripts son las siguientes:
````
pip install lgpio spidev pydub vosk joblib
````
### Salida
La salida del script es la transcripción del audio grabado junto con la duración del proceso. 
Esta segunda salida es el resultado que se busca al realizar esta prueba. 

## transcriptor_audio.py
Este script realiza la misma función que los scripts anteriores, la diferencia es que este realiza la transcripción
sin realizar un **subprocess**. 

### Descripción
El script realiza las siguientes acciones:
Captura de audio controlada por eventos GPIO (pin 26).

* Inicializa el micrófono conectado por I2S para capturar audio cuando se detecta una señal en el pin GPIO 26.

* Graba el audio en formato WAV mientras la señal se mantiene activa.

* Guarda el archivo de audio.

* Añade el archivo a una cola para procesarlo en segundo plano.

* Convierte el audio a un formato compatible con Vosk (mono, 16 kHz, 16 bits).

* Realiza transcripción automática del contenido con el transcriptor Vosk.

* Clasifica la transcripción.

* Muestra en consola la etiqueta resultante de la clasificación.

### Requisitos
Las librerías necesarias para ejecutar el script son:
````
pip install alsaaudio lgpio joblib vosk pydub
````
### Salida
El resultado del script es la impresión por consola de la transcripción junto con la duración del proceso. 

## prueba_conceptos_previos.ino
En este script hay el código utilizado durante las pruebas de conceptos previos. Para su ejecución se necesita una **Arduino Shield** y 
el dispositivo **MySignals hw2 libelium**. 

### Descripción
El script realiza las siguientes acciones:

* Captura la señal EMG mediante la placa MySignals.

* Aplica un filtro pasa banda (10-500 Hz).

* Almacena los datos filtrados en un buffer circular para calcular continuamente el umbral de ruido.

* Calcula el umbral de ruido como la media más dos desviaciones estándar de la señal reciente.

* Detecta la activación y desactivación muscular comparando la señal filtrada con el umbral de ruido, considerando un tiempo mínimo de estabilidad (20 ms) para evitar falsos positivos.

* Reporta por serial el estado del músculo: "Músculo ON" o "Músculo OFF".

### Requisitos
Las dependencias utilizadas son las siguientes: 
````
#include <MySignals.h>
#include "Wire.h"
#include "SPI.h"
````
### Salida
A través del Serial Plot de Arduino se puede observar la forma de la señal y el estado del músculo. 

## prueba_micropin.py
Este script realiza la prueba para observar el cambio de estado de la GPIO 14 asociada al conmutador SPST del cambio de modo EMG a voz y vicerversa. 

### Descripción 
El script realiza las siguientes acciones:

* Monitorea cambios en el pin GPIO 14, configurado como entrada.

* Cada vez que se detecta un cambio de nivel (flanco ascendente o descendente) en ese pin, alterna el modo entre "emg" y "voz".

* Imprime en consola el modo actual tras cada cambio.

### Requisitos 
Las librerías necesarias para ejecutar el script son:
````
pip install lgpio spidev
````
### Salida
El script muestra por consola el estado de la GPIO y cada vez que hay un cambio de estado. 

## prueba_retroalimentacion.py
Este script es usado para verificar los valores devueltos por el ADS1292 referentes a la retroalimentación de los servomotores. 
La función principal del script es realizar el movimiento de flexión de un dedo y obtener los valores
de intensidad del ADS1292 mediante el script **ADS1292.py** (disponible en el archivo Scripts finales)

### Descripción 
El script realiza las siguientes acciones:

* Inicializar el bus I2C y el controlador PCA9685.

* Configurar el MUX para seleccionar la señal de control del dedo deseado.

* Mover el servos suavemente por pasos en función del dedo deseado.

* Leer la señal de realimentación del ADS1292.

### Requisitos 
Las librerías necesarias para ejecutar el script son:
````
pip install adafruit-circuitpython-pca9685 lgpio adafruit-blinka
````
### Salida
La salida del script es la impresión por pantalla de los valores devueltos por el ADS1292, junto con el movimiento
del dedo deseado. 




