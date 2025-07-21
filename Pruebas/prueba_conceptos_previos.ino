#include <MySignals.h>
#include "Wire.h"
#include "SPI.h"

#define SAMPLE_SIZE 100  // Tamaño de la ventana para análisis de ruido
#define THRESHOLD_TIME 20 // Tiempo mínimo de activación/desactivación en ms

float emgData[SAMPLE_SIZE];
uint16_t index = 0;
bool muscleActive = false;
unsigned long activationStart = 0;
float noiseThreshold = 0;

void setup() {
    Serial.begin(115200);
    MySignals.begin();
}

void loop() {
    // Captura la señal EMG
    float emg = MySignals.getEMG();
    
    // Aplicar filtro pasa banda 10-500 Hz (implementación simple con media móvil)
    float filteredEMG = bandPassFilter(emg);
    
    // Almacenar valores en buffer para análisis de ruido
    emgData[index] = filteredEMG;
    index = (index + 1) % SAMPLE_SIZE;
    
    // Calcular umbral de ruido cada vez que el buffer se llena
    if (index == 0) {
        noiseThreshold = calculateNoiseThreshold();
    }
    
    // Detección de activación/desactivación muscular
    detectActivation(filteredEMG);
    
    Serial.println(filteredEMG);
    delay(5);
}

// Función para aplicar un filtro pasa banda 10-500Hz (simplificado)
float bandPassFilter(float input) {
    static float prevInput = 0, prevOutput = 0;
    float alpha = 0.9; // Factor de suavizado (ajustable)
    float highPass = input - prevInput + alpha * prevOutput;
    prevInput = input;
    prevOutput = highPass;
    return highPass;
}

// Calcula el umbral de ruido como la media más dos desviaciones estándar
float calculateNoiseThreshold() {
    float sum = 0, sumSq = 0;
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        sum += emgData[i];
        sumSq += emgData[i] * emgData[i];
    }
    float mean = sum / SAMPLE_SIZE;
    float variance = (sumSq / SAMPLE_SIZE) - (mean * mean);
    float stddev = sqrt(variance);
    return mean + 2 * stddev; // Umbral de ruido
}

// Detección de activación y desactivación del músculo
void detectActivation(float emgValue) {
    unsigned long currentTime = millis();
    if (emgValue > noiseThreshold) {
        if (!muscleActive) {
            if (activationStart == 0) {
                activationStart = currentTime; // Inicia el conteo del tiempo de activación
            } else if (currentTime - activationStart >= THRESHOLD_TIME) {
                muscleActive = true;
                Serial.println("Músculo ON");
            }
        }
    } else {
        if (muscleActive) {
            if (activationStart == 0) {
                activationStart = currentTime;
            } else if (currentTime - activationStart >= THRESHOLD_TIME) {
                muscleActive = false;
                Serial.println("Músculo OFF");
            }
        }
        activationStart = 0; // Reiniciar conteo si la señal baja antes del umbral
    }
}



