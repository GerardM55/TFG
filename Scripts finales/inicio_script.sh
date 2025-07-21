#!/usr/bin/env bash

LOGFILE="/home/raspberry/logs/servicio_arranque.log"
SCRIPT_LOG="/home/raspberry/logs/script_principal.out"
NOTIFY_DIR="/home/raspberry/lgpio_notify"

mkdir -p "$NOTIFY_DIR"
chmod 700 "$NOTIFY_DIR"
mkdir -p /home/raspberry/logs

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iniciando ads1292.service" >> "$LOGFILE"

# Matar lgd por si hay uno en marcha
pkill lgd 2>/dev/null
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iniciando lgd con directorio $NOTIFY_DIR" >> "$LOGFILE"
lgd -d "$NOTIFY_DIR" &

sleep 1

# Exportar variable para que lgpio use el directorio correcto
export LGPIO_DIR="$NOTIFY_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exportado LGPIO_DIR=$LGPIO_DIR" >> "$LOGFILE"

# Cambiar al directorio home para evitar crear archivos .lgd-nfy en lugares no deseados
cd /home/raspberry

# Activar entorno virtual
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Activando entorno virtual..." >> "$LOGFILE"
source /home/raspberry/myenv/bin/activate

# Cargar overlay de sonido
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cargando overlay de sonido..." >> "$LOGFILE"
sudo dtoverlay googlevoicehat-soundcard >> "$LOGFILE" 2>&1
sleep 3

# Recargar ALSA para refrescar los dispositivos de sonido
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Recargando ALSA..." >> "$LOGFILE"
sudo alsa force-reload >> "$LOGFILE" 2>&1
sleep 2

# Mostrar dispositivos de sonido
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dispositivos de sonido:" | tee -a "$LOGFILE"
aplay -l | tee -a "$LOGFILE"

# Crear o vaciar el log del script principal para evitar errores con tail
: > "$SCRIPT_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log del script principal iniciado" >> "$SCRIPT_LOG"

# Ejecutar script principal en background, redirigiendo stdout y stderr al log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ejecutando script_principal.py..." >> "$LOGFILE"
/home/raspberry/myenv/bin/python -u /home/raspberry/script_principal.py >> "$SCRIPT_LOG" 2>&1 &


echo "[$(date '+%Y-%m-%d %H:%M:%S')] ads1292.service finalizado" >> "$LOGFILE"
