import librosa
import numpy as np
import os
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

# Directorio con los audios originales
audio_dir = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre/agarre_obl"
output_dir = "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre/agarre_obl_aug"
os.makedirs(output_dir, exist_ok=True)

# Verificar si el directorio existe
if not os.path.exists(audio_dir):
    print(f"¡Error! El directorio {audio_dir} no existe.")
else:
    print(f"Directorio encontrado: {audio_dir}")

# Ejemplo de augmentaciones que se van a aplicar
augment = Compose([
    AddGaussianNoise(min_amplitude=0.02, max_amplitude=0.1, p=0.5),  # Añadir ruido
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),  # Estiramiento temporal
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)  # Desplazamiento de tono
])

# Función para cargar y guardar audios
def augment_and_save(audio_file, output_folder, num_augmentations=5):
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

# Recorrer los archivos de audio
for root, _, files in os.walk(audio_dir):  # Usamos os.walk() para recorrer subdirectorios
    for file in files:
        if file.endswith(".wav"):
            augment_and_save(os.path.join(root, file), output_dir)
