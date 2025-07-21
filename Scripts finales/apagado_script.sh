#!/usr/bin/env bash

LOGFILE="/home/raspberry/logs/servicio_parada.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Parando ads1292.service" >> "$LOGFILE"
/home/raspberry/myenv/bin/python /home/raspberry/apagar_ads1292.py
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ads1292.service parado" >> "$LOGFILE"
