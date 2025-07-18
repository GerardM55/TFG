import os
import librosa
import numpy as np
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain

# Directorios de entrada y salida
input_dir = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audios_deteccio_veu/altres_veu_aug"
output_dir = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audios_deteccio_veu/altres_veu_aug_2"
os.makedirs(output_dir, exist_ok=True)

# Verificar si el directorio existe
if not os.path.exists(input_dir):
    print(f"¡Error! El directorio {input_dir} no existe.")
else:
    print(f"Directorio encontrado: {input_dir}")

# Definir las técnicas de data augmentation con parámetros suaves
augment = Compose([
    PitchShift(min_semitones=-1, max_semitones=1, p=0.5),  # Cambio de tono suave (±1 semitono)
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),  # Cambio de velocidad suave (90%-110%)
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=0.5),  # Añadir ruido suave
    Gain(min_gain_db=-1.0, max_gain_db=1.0, p=0.5),  # Cambio de volumen suave (-1.0 dB a 1.0 dB)
])

# Función para cargar y guardar audios con augmentación
def augment_and_save(audio_file, output_folder, num_augmentations=6):
    # Cargar el audio original
    audio, sr = librosa.load(audio_file, sr=16000)
    
    # Guardar el audio original
    base_filename = os.path.basename(audio_file)
    sf.write(os.path.join(output_folder, base_filename), audio, sr)
    
    for i in range(num_augmentations):
        # Aplicar augmentación
        augmented_audio = augment(samples=audio, sample_rate=sr)
        
        # Guardar el audio aumentado
        augmented_filename = f"{os.path.splitext(base_filename)[0]}_aug_{i}.wav"
        sf.write(os.path.join(output_folder, augmented_filename), augmented_audio, sr)

# Recorrer los archivos de audio y aplicar augmentaciones
for root, _, files in os.walk(input_dir):  # Usamos os.walk() para recorrer subdirectorios
    for file in files:
        if file.endswith(".wav"):
            augment_and_save(os.path.join(root, file), output_dir)

print("Aumento de datos completado. Archivos guardados en:", output_dir)
