import os
import librosa
import soundfile as sf

# Ruta de entrada y salida
input_folder = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre/agarre_obl_aug"
output_folder = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/agarre_obliquo"

# Crear el directorio de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Función para normalizar audios a 16 kHz
def normalizar_audio(input_file, output_file, target_sr=16000):
    try:
        # Cargar audio
        audio, sr = librosa.load(input_file, sr=None)
        
        # Si la frecuencia es diferente, la convertimos
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Guardar el archivo convertido
        sf.write(output_file, audio, target_sr)
        print(f" Archivo normalizado: {input_file} -> {output_file}")
    except Exception as e:
        print(f" Error procesando {input_file}: {e}")

# Procesar todos los archivos .wav
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".wav"):
            input_file = os.path.join(root, file)
            output_file = os.path.join(output_folder, file)
            normalizar_audio(input_file, output_file)

print("¡Todos los audios han sido normalizados!")
