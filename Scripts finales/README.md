# Descripción del directorio
En este directorio se encuentran todos los scripts necesarios para la ejecución del sistema. A continuación, 
se explican detalladamente. 

## apagado_script.sh 
El script **apagado_script.sh** es un script de shell que su función es registrar en un archivo de log con 
el inicio y final de su ejecución, junto con la ejecución del script **apagar_ads1292.py**. 
El archivo es ejecutado cuando el pulsador de la PiJuice Zero es presionado dos veces seguidas. 

## apagar_ads1292.py
El script tiene la función de apagar el sistema operativo, concretamente la alimentación a la Raspberry Pi Zero 2W
a través de la PiJuice. 

### Descripción

El script realiza las siguientes acciones:

* Inicializa la comunicación con el dispositivo PiJuice vía I2C.

* Registra en un archivo de log el inicio del proceso de apagado.

*Espera 2 segundos para asegurar el cierre limpio de conexiones.

* Envía la orden de corte de energía para que el PiJuice apague el sistema en 15 segundos.

* Registra en el log la confirmación de la orden enviada o cualquier error ocurrido.

### Requisitos
Las librerías de Python necesarias para la ejecución del script son: 
````
pip install pijuice
````
### Salida 
El script genera un archivo de log en _/home/raspberry/logs/apagar_ads1292.log_ donde se registran los eventos y posibles errores del proceso de apagado.

## inicio_script.sh
Este script de shell es el encargado de la inicialización del sistema. Esta tiene lugar al presionar una vez el pulsador de la Pijuice Zero. 

### Descripción
El archivo realiza las siguientes tareas: 

* Crea los directorios necesarios para logs y notificaciones.

* Finaliza cualquier proceso lgd previo y lanza uno nuevo con la ruta de notificación personalizada.

* Exporta la variable de entorno LGPIO_DIR para asegurar el funcionamiento correcto de lgpio.

* Activa el entorno virtual de Python.

* Carga el overlay de sonido para Google Voice HAT.

* Recarga ALSA para detectar dispositivos de audio.

* Ejecuta el script **script_principal.py** y lo redirige a un archivo de log.

### Requisitos
No se necesita ninguna librería en particular para ejecutar el archivo. 

### Salida 
Se generan dos archivos de log en /home/raspberry/logs:

* **servicio_arranque.log**: Registra los eventos generales del arranque del sistema y la ejecución del script.

* **script_principal.out**: Captura la salida del script Python principal **script_principal.py**, incluyendo posibles errores de ejecución.

## script_principal.py
Es el script encargado de controlar la ejecución principal del sistema y de llamar a los distintos scripts en función de las acciones a realizar. 

### Descripción
El script realiza las siguientes acciones:
* Inicialización de GPIO y recursos compartidos
* Lectura del modo actual
* Modo EMG:
    * Se activa la interfaz SPI del módulo ADS1292 para comenzar la adquisición de señales EMG.
    * Se capta la contracción muscular
    * Si la mano está abierta, se ejecuta el agarre "general".
    * Si la mano está cerrada, se ejecuta "stop".
* Modo voz:
    * Se llama a la función principal del módulo grabar_audio que graba el audio y devuelve una etiqueta con el agarre deseado
    * Si el agarre previo era "stop", se ejecuta el agarre deseado.
    * Si el agarre previo era diferente a "stop", se ejecuta el agarre "stop".
    * El sistema espera 3 segundos por si el usuario quiero cambiar de modo.

### Requisitos
Las librerías necesarias para la ejecución del script son:
````
pip install lgpio
pip install flask
pip install pydub
pip install joblib
pip install vosk
````
### Salida
El script muestra a través de prints el estado de la ejecución. 

## grabar_audio.py
Este script es el encargado de la grabación y transcripción del audio. A continuación, se detalla su funcionamiento. 

### Descripción
El script realiza las siguientes acciones:
* Configuración de GPIO e interrupciones
* Grabación del audio a través de I2S con ALSA.
* El audio se guarda en un archivo wav
* Conversión y transcripción del audio con el transcriptor Vosk
* Limpieza y clasificación del agarre.
* Se devuelve la etiqueta del agarre al script principal.
* Se cierra el I2S y se liberan las GPIOs correspondientes.

### Requisitos
Las librerías necesarias para la ejecución del script son:
````
pip install lgpio
pip install alsaaudio
pip install pydub
pip install joblib
pip install vosk
pip install scikit-learn
pip install unicodedata
````
### Salida
La salida de este script es la devolución del resultado de la transcripción del audio a través de la variable **etiqueta** en la función **main()**. 

## PCA9685.py 
Este script es el encargado de establecer conexión con el módulo PCA9685 y gestionar el movimiento de los servomotores en función del agarre seleccionado. 

### Descripción
El script realiza las siguientes acciones:
* Configura el bus I2C
* Según el tipo de agarre solicitado, determina qué dedos mover, en qué dirección y con qué tipo de movimiento:
  * **R**: Movimiento sin realimentación.
  * **S**: Movimento con realimentación.
* Configura el MUX para gestionar la realimentación en caso de que el agarre no sea ni "stop" ni "general".
* Mueve cada servo al valor definido en el agarre.
* Finaliza liberando recursos (I2C, GPIOs).

### Requisitos
Las librerías necesarias para la ejecución del script son:
````
pip install adafruit-circuitpython-pca9685 lgpio
````

### Salida
El script muestra su estado a través de prints. La salida es visual con los movimientos de los servomotores. 

## ADS1292.py
Este script es el encargado de controlar el ADS1292, el cual tiene dos funciones principales:
* Lectura de la señal EMG.
* Lectura de la realimentación de los servomotores.

### Descripción 
A continuación, se detalla el proceso de ejecución en función del modo utilizado
* Inicializa el bus SPI
* Configura los registros del ADS1292 en función del modo.
* Modo EMG:
    * Espera 6 segundos.
    * Realiza la adquisición de picos para establecer unos margenes para la calibración de picos.
    * Realiza la calibración de picos para establecer un threshold para la detección de la contracción muscular.
    * Adquiere muestras hasta detectar una contracción muscular.
* Modo voz:
    * Calcula el valor de la corriente del servomotor.
    * Devuelve si el servomotor ha superado el límite de corriente.
* Finaliza el bus SPI y libera las GPIOS correspondientes 

### Requisitos
Las librerías necesarias para la ejecución del script son:
````
pip install lgpio numpy scipy
````
### Salida
El script muestra a través de prints los valores adquiridos con el ADS1292. 

