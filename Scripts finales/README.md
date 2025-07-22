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
